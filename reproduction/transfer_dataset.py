"""Training-free TAC transfer to CIFAR-20 or STL-10 using cached WordNet text.

Long CLIP image extraction is resumable batch by batch. Text weight 0.75 was
chosen on CIFAR-10 and is evaluated here without target-dataset tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import time
from pathlib import Path

import clip
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100, STL10

from cached_experiments import PROMPT_FILES, retrieve, selected_noun_indices
from reproduce_cifar10 import cluster_kmeans, l2_normalize


class CoarseCIFAR100(CIFAR100):
    """CIFAR-100 images with the 20 official coarse labels as targets."""

    def __init__(self, root: str, train: bool, transform):
        super().__init__(root=root, train=train, download=True, transform=transform)
        file = Path(root) / self.base_folder / ("train" if train else "test")
        with file.open("rb") as stream:
            self.targets = pickle.load(stream, encoding="latin1")["coarse_labels"]


def dataset_for(name: str, root: Path, split: str, preprocess):
    if name == "cifar20":
        return CoarseCIFAR100(str(root), train=split == "train", transform=preprocess)
    return STL10(root=str(root), split=split, download=True, transform=preprocess)


def embedding_split(name: str, split: str, root: Path, cache: Path, model, preprocess,
                    device: torch.device, batch_size: int, workers: int):
    feature_path = cache / f"{name}_image_{split}.npy"
    label_path = cache / f"{name}_labels_{split}.npy"
    if feature_path.exists() and label_path.exists():
        return np.load(feature_path), np.load(label_path)
    dataset = dataset_for(name, root, split, preprocess)
    n = len(dataset)
    partial_features = cache / f"{name}_image_{split}.partial.npy"
    partial_labels = cache / f"{name}_labels_{split}.partial.npy"
    progress_path = cache / f"{name}_image_{split}.progress.json"
    if progress_path.exists():
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
        if not partial_features.exists() or not partial_labels.exists() or not 0 <= completed <= n:
            raise RuntimeError(f"Incomplete checkpoint for {name}/{split}")
        features = np.lib.format.open_memmap(partial_features, mode="r+")
        labels = np.lib.format.open_memmap(partial_labels, mode="r+")
        if features.shape != (n, 512) or labels.shape != (n,):
            raise RuntimeError("Cached image array shape does not match the dataset")
    else:
        completed = 0
        features = np.lib.format.open_memmap(partial_features, mode="w+", dtype=np.float32,
                                              shape=(n, 512))
        labels = np.lib.format.open_memmap(partial_labels, mode="w+", dtype=np.int64,
                                            shape=(n,))
        progress_path.write_text(json.dumps({"completed": 0}) + "\n", encoding="utf-8")
    print(f"{name}/{split}: resuming at {completed}/{n}", flush=True)
    loader = DataLoader(Subset(dataset, range(completed, n)), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    for images, targets in loader:
        with torch.inference_mode():
            encoded = model.encode_image(images.to(device)).float().cpu().numpy()
        end = completed + len(targets)
        features[completed:end] = encoded
        labels[completed:end] = targets.numpy()
        features.flush()
        labels.flush()
        progress_path.write_text(json.dumps({"completed": end}) + "\n", encoding="utf-8")
        completed = end
        if completed == n or completed % (batch_size * 20) == 0:
            print(f"{name}/{split}: {completed}/{n}", flush=True)
    del features, labels
    partial_features.replace(feature_path)
    partial_labels.replace(label_path)
    progress_path.unlink()
    return np.load(feature_path), np.load(label_path)


def metrics(labels: np.ndarray, pred: np.ndarray, clusters: int) -> dict[str, float]:
    counts = np.zeros((clusters, clusters), dtype=np.int64)
    np.add.at(counts, (labels.astype(int), pred.astype(int)), 1)
    rows, cols = linear_sum_assignment(-counts)
    return {
        "ACC": float(100 * counts[rows, cols].sum() / len(labels)),
        "NMI": float(100 * normalized_mutual_info_score(labels, pred)),
        "ARI": float(100 * adjusted_rand_score(labels, pred)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cifar20", "stl10"))
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-set", choices=("single", "seven"), default="seven")
    parser.add_argument("--backend", choices=("sklearn", "faiss-cpu"), default="faiss-cpu")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.75])
    args = parser.parse_args()
    if any(weight < 0 for weight in args.weights):
        raise ValueError("Weights must be nonnegative")
    torch.set_num_threads(args.threads)
    if args.backend == "faiss-cpu":
        import faiss

        faiss.omp_set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache = args.data_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    output = args.output_dir or root / "results" / f"{args.dataset}_transfer"
    output.mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False,
                                  download_root=args.data_dir / "models")
    model.eval()
    started = time.perf_counter()
    train_image, _ = embedding_split(args.dataset, "train", args.data_dir, cache,
                                     model, preprocess, device, args.batch_size, args.workers)
    test_image, test_labels = embedding_split(args.dataset, "test", args.data_dir, cache,
                                              model, preprocess, device, args.batch_size, args.workers)
    clusters = 20 if args.dataset == "cifar20" else 10
    semantic_centers = 167 if args.dataset == "cifar20" else 17
    noun_features = np.load(cache / PROMPT_FILES[args.prompt_set], mmap_mode="r")
    indices = selected_noun_indices(l2_normalize(train_image), noun_features,
                                    semantic_centers, 5, args.backend, args.seeds[0], 10)
    selected = l2_normalize(np.asarray(noun_features[indices], dtype=np.float32))
    image_unit = l2_normalize(test_image)
    retrieved = retrieve(image_unit, selected, tau=0.005, mode="soft")
    rows = []
    for weight in args.weights:
        fused = np.concatenate((image_unit, weight * retrieved), axis=1)
        for seed in args.seeds:
            pred, _ = cluster_kmeans(fused, clusters, seed, args.restarts, args.backend)
            row = {"dataset": args.dataset, "weight": weight, "seed": seed,
                   **metrics(test_labels, pred, clusters)}
            rows.append(row)
            print(row, flush=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": args.dataset, "prompt_set": args.prompt_set, "backend": args.backend,
        "device": str(device), "python": platform.python_version(),
        "clusters": clusters, "semantic_centers": semantic_centers,
        "selected_nouns": len(selected), "seeds": args.seeds, "restarts": args.restarts,
        "weights": args.weights, "elapsed_seconds": time.perf_counter() - started,
        "summary": {
            str(weight): {metric: {
                "mean": float(np.mean([r[metric] for r in rows if r["weight"] == weight])),
                "std": float(np.std([r[metric] for r in rows if r["weight"] == weight], ddof=1))
                if len(args.seeds) > 1 else 0.0,
            } for metric in ("ACC", "NMI", "ARI")}
            for weight in args.weights
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results in {output}", flush=True)


if __name__ == "__main__":
    main()
