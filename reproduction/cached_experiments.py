"""Label-free protocol ablations on cached CIFAR-10 CLIP/WordNet embeddings.

Labels are used only to calculate metrics, never for feature or noun selection. Report
every requested configuration; do not select a hyperparameter on test scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reproduce_cifar10 import cluster_kmeans, clustering_metrics, l2_normalize


PROMPT_FILES = {
    "single": "wordnet_nouns_1prompt_ensemble.npy",
    "seven": "wordnet_nouns_trimmed_7prompt_ensemble.npy",
}


def load_arrays(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_train = l2_normalize(np.load(cache / "cifar10_image_train.npy").astype(np.float32))
    image_test = l2_normalize(np.load(cache / "cifar10_image_test.npy").astype(np.float32))
    labels_test = np.load(cache / "cifar10_labels_test.npy")
    if image_train.shape != (50000, 512) or image_test.shape != (10000, 512):
        raise ValueError("Expected CIFAR-10 CLIP ViT-B/32 cache shapes")
    return image_train, image_test, labels_test


def selected_noun_indices(
    image_train: np.ndarray,
    nouns: np.ndarray,
    centers_count: int,
    gamma: int,
    backend: str,
    center_seed: int,
    center_restarts: int,
) -> np.ndarray:
    """Follow the paper's center-softmax and top-gamma selection rule."""
    assignments, _ = cluster_kmeans(
        image_train, centers_count, center_seed, center_restarts, backend
    )
    centers = np.stack(
        [image_train[assignments == i].mean(axis=0) for i in range(centers_count)]
    )
    if not np.isfinite(centers).all():
        raise RuntimeError("An empty semantic center was produced")
    centers = l2_normalize(centers)
    noun_unit = l2_normalize(nouns.astype(np.float32, copy=False))
    logits = centers @ noun_unit.T
    logits -= logits.max(axis=0, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    winner = probabilities.argmax(axis=0)
    confidence = probabilities.max(axis=0)
    chosen = []
    for i in range(centers_count):
        candidates = np.flatnonzero(winner == i)
        order = np.argsort(-confidence[candidates], kind="stable")
        chosen.extend(candidates[order[:gamma]].tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def retrieve(image: np.ndarray, nouns: np.ndarray, tau: float, mode: str) -> np.ndarray:
    noun_unit = l2_normalize(nouns)
    out = np.empty_like(image)
    batch_size = 64 if len(noun_unit) > 5000 else 256
    for start in range(0, len(image), batch_size):
        sims = image[start : start + batch_size] @ noun_unit.T
        if mode == "hard":
            out[start : start + batch_size] = noun_unit[sims.argmax(axis=1)]
        else:
            logits = sims / tau
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1, keepdims=True)
            out[start : start + batch_size] = weights @ noun_unit
    return l2_normalize(out)


def evaluate(
    image: np.ndarray,
    text: np.ndarray | None,
    labels: np.ndarray,
    backend: str,
    seeds: list[int],
    restarts: int,
    fields: dict,
    max_points_per_centroid: int | None = None,
) -> list[dict]:
    representation = image if text is None else np.concatenate((image, text), axis=1)
    rows = []
    for seed in seeds:
        prediction, centers = cluster_kmeans(representation, 10, seed, restarts, backend,
                                             max_points_per_centroid)
        sizes = np.bincount(prediction, minlength=10)
        row = {
            **fields,
            "eval_backend": backend,
            "seed": seed,
            "max_points_per_centroid": max_points_per_centroid or "default",
            "nouns_selected": fields.get("nouns_selected", 0),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "mean_assigned_cosine": float(
                np.mean(np.sum(l2_normalize(representation) * centers[prediction], axis=1))
            ),
            **clustering_metrics(labels, prediction),
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def run(args: argparse.Namespace) -> list[dict]:
    if args.tau <= 0 or args.gamma < 1 or args.centers < 10:
        raise ValueError("tau > 0, gamma >= 1, and centers >= 10 are required")
    if "faiss-cpu" in (*args.selection_backends, args.eval_backend):
        import faiss

        faiss.omp_set_num_threads(args.threads)
    image_train, image_test, labels_test = load_arrays(args.cache)
    rows = []
    if args.image_only:
        rows += evaluate(image_test, None, labels_test, args.eval_backend,
                         args.seeds, args.restarts, {"experiment": args.experiment,
                                                      "retrieval": "image-only"},
                         args.max_points_per_centroid)
    if args.experiment == "stability":
        for prompt in args.prompts:
            path = args.cache / (
                "cifar10_retrieved_text_test_full-wordnet_"
                + ("paper-single" if prompt == "single" else "authors-seven_trimmed")
                + "_faiss.npy"
            )
            text = np.load(path).astype(np.float32)
            rows += evaluate(image_test, text, labels_test, args.eval_backend,
                             args.seeds, args.restarts, {"experiment": "stability", "prompt": prompt},
                             args.max_points_per_centroid)
        return rows

    if args.all_nouns:
        for prompt in args.prompts:
            nouns = np.load(args.cache / PROMPT_FILES[prompt], mmap_mode="r")
            text = retrieve(image_test, nouns, args.tau, "soft")
            rows += evaluate(image_test, text, labels_test, args.eval_backend,
                             args.seeds, args.restarts,
                             {"experiment": args.experiment, "retrieval": "all-nouns",
                              "retrieval_prompt": prompt, "tau": args.tau,
                              "nouns_selected": len(nouns)}, args.max_points_per_centroid)

    if args.no_selected:
        return rows
    for selection_prompt in args.prompts:
        nouns = np.load(args.cache / PROMPT_FILES[selection_prompt], mmap_mode="r")
        for selection_backend in args.selection_backends:
            indices = selected_noun_indices(image_train, nouns, args.centers, args.gamma,
                                            selection_backend, args.center_seed, args.center_restarts)
            retrieval_prompts = args.retrieval_prompts if args.experiment == "factorial" else [selection_prompt]
            for retrieval_prompt in retrieval_prompts:
                retrieval_nouns = np.load(args.cache / PROMPT_FILES[retrieval_prompt], mmap_mode="r")
                selected = l2_normalize(np.asarray(retrieval_nouns[indices], dtype=np.float32))
                for mode in args.modes:
                    text = retrieve(image_test, selected, args.tau, mode)
                    fields = {
                        "experiment": args.experiment,
                        "selection_prompt": selection_prompt,
                        "retrieval_prompt": retrieval_prompt,
                        "selection_backend": selection_backend,
                        "retrieval": mode,
                        "centers": args.centers,
                        "gamma": args.gamma,
                        "tau": args.tau,
                        "nouns_selected": len(selected),
                    }
                    rows += evaluate(image_test, text, labels_test, args.eval_backend,
                                     args.seeds, args.restarts, fields,
                                     args.max_points_per_centroid)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("stability", "factorial", "ablation"))
    parser.add_argument("--cache", type=Path, default=root / "data" / "cache")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompts", nargs="+", choices=("single", "seven"), default=["single", "seven"])
    parser.add_argument("--retrieval-prompts", nargs="+", choices=("single", "seven"),
                        default=["single", "seven"])
    parser.add_argument("--selection-backends", nargs="+", choices=("sklearn", "faiss-cpu"),
                        default=["sklearn", "faiss-cpu"])
    parser.add_argument("--eval-backend", choices=("sklearn", "faiss-cpu"), default="faiss-cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--center-seed", type=int, default=0)
    parser.add_argument("--center-restarts", type=int, default=10)
    parser.add_argument("--centers", type=int, default=167)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--modes", nargs="+", choices=("soft", "hard"), default=["soft"])
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--all-nouns", action="store_true",
                        help="Retrieve from every WordNet noun; expensive on CPU.")
    parser.add_argument("--no-selected", action="store_true",
                        help="Skip selected-noun variants (use with --all-nouns).")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-points-per-centroid", type=int, default=None,
                        help="Override FAISS's default training sample cap; diagnostic only.")
    args = parser.parse_args()
    if args.no_selected and not (args.all_nouns or args.image_only):
        parser.error("--no-selected requires --all-nouns or --image-only")
    if args.restarts < 1 or args.center_restarts < 1 or args.threads < 1:
        parser.error("restarts, center-restarts, and threads must be positive")
    if args.max_points_per_centroid is not None and args.max_points_per_centroid < 1:
        parser.error("max-points-per-centroid must be positive")
    output = args.output or root / "results" / f"{args.experiment}.csv"
    rows = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
