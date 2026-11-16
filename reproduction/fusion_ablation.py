"""Exploratory text-weight ablation using cached CIFAR-10 embeddings.

This script reports every requested weight. Test labels are used only for
evaluation; do not select a weight on the test set and call it a validated gain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    counts = np.zeros((10, 10), dtype=np.int64)
    np.add.at(counts, (y.astype(int), pred.astype(int)), 1)
    row, col = linear_sum_assignment(-counts)
    return {
        "ACC": 100 * counts[row, col].sum() / len(y),
        "NMI": 100 * normalized_mutual_info_score(y, pred),
        "ARI": 100 * adjusted_rand_score(y, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--cache", type=Path, default=root / "data" / "cache")
    parser.add_argument("--output", type=Path, default=root / "results" / "fusion_ablation.csv")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-init", type=int, default=20)
    args = parser.parse_args()

    image = unit(np.load(args.cache / "cifar10_image_test.npy").astype(np.float32))
    text_path = args.cache / "cifar10_retrieved_text_test_full-wordnet_paper-single.npy"
    text = unit(np.load(text_path).astype(np.float32))
    labels = np.load(args.cache / "cifar10_labels_test.npy")
    if image.shape != text.shape or image.shape[0] != labels.size:
        raise ValueError("Cached embeddings and labels have inconsistent shapes")

    rows = []
    for weight in args.weights:
        if weight < 0:
            raise ValueError("Text weights must be nonnegative")
        # Equal weight (1.0) is the paper's feature concatenation.
        features = np.concatenate((image, weight * text), axis=1)
        for seed in args.seeds:
            prediction = KMeans(
                n_clusters=10, init="k-means++", n_init=args.n_init,
                max_iter=300, random_state=seed, algorithm="lloyd",
            ).fit_predict(features)
            row = {"text_weight": weight, "seed": seed, **scores(labels, prediction)}
            rows.append(row)
            print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
