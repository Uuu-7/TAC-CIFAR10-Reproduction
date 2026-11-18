"""Reproduce the training-free CIFAR-10 comparison from TAC (ICML 2024).

The two evaluated representations are:
  1. normalized CLIP ViT-B/32 image embeddings;
  2. image embeddings concatenated with TAC's retrieved WordNet text counterpart.

Expensive intermediate embeddings are cached under data/cache, so interrupted CPU
runs can be resumed by launching the same command again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


PROMPTS = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)

PAPER = {
    "clip_image_kmeans": {"ACC": 74.2, "NMI": 70.3, "ARI": 61.6},
    "image_text_kmeans": {"ACC": 90.1, "NMI": 80.8, "ARI": 79.8},
}

PAPER_TOP30_NOUNS = [
    "spadefoot toad", "field sparrow", "curassow", "Lipizzan", "sable antelope",
    "black fronted bush shrike", "chameleon tree frog", "whitetail deer", "fire truck",
    "waterbuck", "emu", "hartebeest", "dressage", "containership", "pratincole",
    "woodland caribou", "hydroplane racing", "banana boat", "yacht race",
    "yellow breasted chat", "pen tailed tree shrew", "elk", "wagtail", "stonefish",
    "trucking rig", "clipper ship", "Texas horned lizard", "stealth bomber",
    "articulated lorry", "fire department",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def encode_text_without_padding(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Discard positions after the last EOT; CLIP attention is causal.

    Only the EOT representation is returned by CLIP, so later padding tokens
    cannot influence it. The mask is restored after each call.
    """
    length = int(tokens.argmax(dim=-1).max().item()) + 1
    if length == tokens.shape[1]:
        return model.encode_text(tokens)
    blocks = tuple(model.transformer.resblocks)
    original_masks = [block.attn_mask for block in blocks]
    try:
        for block, mask in zip(blocks, original_masks):
            block.attn_mask = mask[:length, :length]
        shortened = tokens[:, :length]
        x = model.token_embedding(shortened).type(model.dtype)
        x = x + model.positional_embedding[:length].type(model.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x).type(model.dtype)
        row = torch.arange(x.shape[0], device=x.device)
        return x[row, shortened.argmax(dim=-1)] @ model.text_projection
    finally:
        for block, mask in zip(blocks, original_masks):
            block.attn_mask = mask


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    contingency = np.zeros((10, 10), dtype=np.int64)
    np.add.at(contingency, (y_true.astype(int), y_pred.astype(int)), 1)
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    acc = contingency[rows, cols].sum() / y_true.size
    return {
        "ACC": 100.0 * float(acc),
        "NMI": 100.0 * float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": 100.0 * float(adjusted_rand_score(y_true, y_pred)),
    }


def image_embeddings(
    model: torch.nn.Module,
    preprocess,
    root: Path,
    cache: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, train in (("train", True), ("test", False)):
        feature_path = cache / f"cifar10_image_{split}.npy"
        label_path = cache / f"cifar10_labels_{split}.npy"
        if feature_path.exists() and label_path.exists():
            outputs[split] = (np.load(feature_path), np.load(label_path))
            continue

        dataset = CIFAR10(root=root, train=train, download=True, transform=preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        features, labels = [], []
        started = time.perf_counter()
        for step, (images, target) in enumerate(loader, 1):
            with torch.inference_mode():
                encoded = model.encode_image(images.to(device)).float()
            features.append(encoded.cpu().numpy())
            labels.append(target.numpy())
            if step == 1 or step % 20 == 0 or step == len(loader):
                elapsed = time.perf_counter() - started
                done = min(step * batch_size, len(dataset))
                print(f"image/{split}: {done}/{len(dataset)} ({elapsed:.1f}s)", flush=True)
        features_array = np.concatenate(features).astype(np.float32)
        labels_array = np.concatenate(labels).astype(np.int64)
        np.save(feature_path, features_array)
        np.save(label_path, labels_array)
        outputs[split] = (features_array, labels_array)
    return outputs["train"][0], outputs["test"][0], outputs["test"][1]


def noun_embeddings(
    model: torch.nn.Module,
    nouns: list[str],
    cache: Path,
    device: torch.device,
    batch_size: int,
    cache_key: str,
    prompts: tuple[str, ...],
    trim_text_padding: bool = False,
) -> np.ndarray:
    ensemble_path = cache / f"{cache_key}_{len(prompts)}prompt_ensemble.npy"
    if ensemble_path.exists():
        return np.load(ensemble_path)

    prompt_features = []
    for template_index, template in enumerate(prompts):
        prompt_path = cache / f"{cache_key}_{len(prompts)}prompt_{template_index}.npy"
        if prompt_path.exists():
            prompt_features.append(np.load(prompt_path))
            continue
        partial_path = prompt_path.with_name(prompt_path.stem + ".partial.npy")
        progress_path = prompt_path.with_name(prompt_path.stem + ".progress.json")
        if partial_path.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed = int(progress["completed"])
            prompt_array = np.lib.format.open_memmap(partial_path, mode="r+")
            if prompt_array.shape != (len(nouns), 512):
                raise RuntimeError(f"Partial cache has an unexpected shape: {partial_path}")
            print(f"Resuming {partial_path.name} at noun {completed}.", flush=True)
        else:
            completed = 0
            prompt_array = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float32, shape=(len(nouns), 512)
            )
            progress_path.write_text(
                json.dumps({"completed": 0}) + "\n", encoding="utf-8"
            )
        started = time.perf_counter()
        for start in range(completed, len(nouns), batch_size):
            batch = [template.format(noun) for noun in nouns[start : start + batch_size]]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            with torch.inference_mode():
                encoded = (
                    encode_text_without_padding(model, tokens)
                    if trim_text_padding else model.encode_text(tokens)
                ).float()
            done = min(start + batch_size, len(nouns))
            prompt_array[start:done] = encoded.cpu().numpy()
            prompt_array.flush()
            progress_path.write_text(
                json.dumps({"completed": done}) + "\n", encoding="utf-8"
            )
            if start == 0 or done % (batch_size * 20) == 0 or done == len(nouns):
                elapsed = time.perf_counter() - started
                print(
                    f"text/prompt-{template_index}: {done}/{len(nouns)} ({elapsed:.1f}s)",
                    flush=True,
                )
        del prompt_array
        partial_path.replace(prompt_path)
        progress_path.unlink(missing_ok=True)
        prompt_features.append(np.load(prompt_path))

    ensemble = np.mean(prompt_features, axis=0, dtype=np.float32)
    np.save(ensemble_path, ensemble)
    return ensemble


def cluster_kmeans(
    x: np.ndarray, n_clusters: int, seed: int, n_init: int, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster features using either the historical proxy or spherical FAISS CPU."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if backend == "faiss-cpu":
        import faiss

        estimator = faiss.Kmeans(
            x.shape[1], n_clusters, spherical=True, niter=300,
            nredo=n_init, seed=seed, gpu=False, verbose=False,
        )
        estimator.train(x)
        _, labels = estimator.index.search(x, 1)
        return labels.reshape(-1), l2_normalize(estimator.centroids)
    if backend != "sklearn":
        raise ValueError(f"Unknown K-means backend: {backend}")
    estimator = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=seed,
        algorithm="lloyd",
        verbose=0,
    )
    labels = estimator.fit_predict(x)
    centers = l2_normalize(estimator.cluster_centers_.astype(np.float32))
    return labels, centers


def select_discriminative_nouns(
    train_images: np.ndarray,
    nouns: list[str],
    noun_features: np.ndarray,
    cache: Path,
    seed: int,
    cache_key: str,
    backend: str,
) -> tuple[np.ndarray, list[str]]:
    selected_path = cache / f"cifar10_selected_noun_features_{cache_key}.npy"
    selected_words_path = cache / f"cifar10_selected_nouns_{cache_key}.txt"
    if selected_path.exists() and selected_words_path.exists():
        selected_words = selected_words_path.read_text(encoding="utf-8").splitlines()
        return np.load(selected_path), selected_words

    train_unit = l2_normalize(train_images)
    noun_unit = l2_normalize(noun_features)
    print("Fitting 167 semantic centers on CIFAR-10 train embeddings...", flush=True)
    assignments, centers = cluster_kmeans(train_unit, 167, seed, n_init=10, backend=backend)
    # Recompute means exactly as in the authors' filter_nouns.py, then normalize.
    centers = np.stack([train_unit[assignments == k].mean(0) for k in range(167)])
    centers = l2_normalize(centers)

    logits = centers @ noun_unit.T
    logits -= logits.max(axis=0, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    winner = probabilities.argmax(axis=0)
    confidence = probabilities.max(axis=0)

    selected_indices: list[int] = []
    for center_index in range(167):
        candidates = np.flatnonzero(winner == center_index)
        if candidates.size:
            order = np.argsort(-confidence[candidates], kind="stable")
            selected_indices.extend(candidates[order[:5]].tolist())
    selected_indices = sorted(set(selected_indices))
    selected_features = noun_unit[selected_indices].astype(np.float32)
    selected_words = [nouns[index] for index in selected_indices]
    np.save(selected_path, selected_features)
    selected_words_path.write_text("\n".join(selected_words) + "\n", encoding="utf-8")
    print(f"Selected {len(selected_words)} discriminative WordNet nouns.", flush=True)
    return selected_features, selected_words


def retrieve_text_counterparts(
    test_images: np.ndarray, selected_nouns: np.ndarray, tau: float = 0.005
) -> np.ndarray:
    image_unit = l2_normalize(test_images)
    logits = image_unit @ l2_normalize(selected_nouns).T / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return l2_normalize(weights @ selected_nouns).astype(np.float32)


def evaluate(
    y_true: np.ndarray,
    image_features: np.ndarray,
    text_features: np.ndarray,
    seeds: list[int],
    n_init: int,
    backend: str,
) -> tuple[list[dict], dict]:
    representations = {
        "clip_image_kmeans": l2_normalize(image_features),
        "image_text_kmeans": np.concatenate(
            [l2_normalize(image_features), l2_normalize(text_features)], axis=1
        ),
    }
    rows = []
    for method, representation in representations.items():
        for seed in seeds:
            started = time.perf_counter()
            prediction, _ = cluster_kmeans(representation, 10, seed, n_init, backend)
            metric = clustering_metrics(y_true, prediction)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **metric,
                    "seconds": time.perf_counter() - started,
                }
            )
            print(method, seed, metric, flush=True)

    summary = {}
    for method in representations:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for metric in ("ACC", "NMI", "ARI"):
            values = np.array([row[metric] for row in subset])
            summary[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "paper": PAPER[method][metric],
                "delta_vs_paper": float(values.mean() - PAPER[method][metric]),
            }
    summary["improvement"] = {
        metric: summary["image_text_kmeans"][metric]["mean"]
        - summary["clip_image_kmeans"][metric]["mean"]
        for metric in ("ACC", "NMI", "ARI")
    }
    summary["paper_improvement"] = {
        metric: PAPER["image_text_kmeans"][metric] - PAPER["clip_image_kmeans"][metric]
        for metric in ("ACC", "NMI", "ARI")
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--kmeans-backend", choices=("sklearn", "faiss-cpu"), default="sklearn")
    parser.add_argument("--trim-text-padding", action="store_true",
                        help="Skip text positions after EOT; verify numerical equivalence before use.")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--noun-source",
        choices=("full-wordnet", "paper-top30"),
        default="full-wordnet",
        help="Use the exact full vocabulary or the bounded Table 7 guidance set.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("authors-seven", "paper-single"),
        default="paper-single",
        help="Use the released code's seven prompts or the paper's single 'a photo of' prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.kmeans_backend == "faiss-cpu":
        import faiss

        faiss.omp_set_num_threads(args.threads)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache = args.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False, download_root=args.data_dir / "models"
    )
    model.eval()
    train_images, test_images, test_labels = image_embeddings(
        model,
        preprocess,
        args.data_dir,
        cache,
        device,
        args.image_batch_size,
        args.workers,
    )
    prompts = PROMPTS if args.prompt_set == "authors-seven" else ("a photo of a {}.",)
    if args.noun_source == "full-wordnet":
        noun_table = pd.read_csv(args.data_dir / "WordNetNouns.csv")
        nouns = noun_table["word"].astype(str).tolist()
        noun_features = noun_embeddings(
            model, nouns, cache, device, args.text_batch_size,
            "wordnet_nouns_trimmed" if args.trim_text_padding else "wordnet_nouns",
            prompts, args.trim_text_padding,
        )
        selected_features, selected_words = select_discriminative_nouns(
            train_images, nouns, noun_features, cache, args.seeds[0],
            f"{args.prompt_set}{'_trimmed' if args.trim_text_padding else ''}{'_faiss' if args.kmeans_backend == 'faiss-cpu' else ''}_seed{args.seeds[0]}",
            args.kmeans_backend,
        )
        protocol_variant = "authors' full WordNet selection protocol"
    else:
        selected_words = PAPER_TOP30_NOUNS
        selected_features = l2_normalize(
            noun_embeddings(
                model,
                selected_words,
                cache,
                device,
                args.text_batch_size,
                "paper_table7_top30_trimmed" if args.trim_text_padding else "paper_table7_top30",
                prompts,
                args.trim_text_padding,
            )
        )
        protocol_variant = "bounded run using the paper's published CIFAR-10 Table 7 Top-30 nouns"
    retrieved = retrieve_text_counterparts(test_images, selected_features)
    np.save(cache / f"cifar10_retrieved_text_test_{args.noun_source}_{args.prompt_set}{'_trimmed' if args.trim_text_padding else ''}{'_faiss' if args.kmeans_backend == 'faiss-cpu' else ''}.npy", retrieved)
    rows, summary = evaluate(
        test_labels, test_images, retrieved, args.seeds, args.n_init, args.kmeans_backend
    )

    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "target": "CIFAR-10 test split, K=10",
        "protocol": {
            "model": "OpenAI CLIP ViT-B/32",
            "train_images_for_text_space": 50000,
            "test_images_for_evaluation": 10000,
            "semantic_centers": 167,
            "nouns_per_center": 5,
            "prompts": list(prompts),
            "prompt_set": args.prompt_set,
            "trim_text_padding": args.trim_text_padding,
            "retrieval_temperature": 0.005,
            "selected_nouns": len(selected_words),
            "noun_source": args.noun_source,
            "protocol_variant": protocol_variant,
            "kmeans_backend": args.kmeans_backend,
            "kmeans_n_init": args.n_init,
            "seeds": args.seeds,
        },
        "environment": {
            "device": str(device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
