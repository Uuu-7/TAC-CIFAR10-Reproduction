"""Train the CIFAR-10 TAC cluster heads from cached CLIP features.

Implements the authors' two heads, cross-modal neighbor distillation, confidence
and balance losses. Training and nearest-neighbor mining are resumable; test
labels are used only for final evaluation after each seed's training.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reproduce_cifar10 import clustering_metrics, l2_normalize, retrieve_text_counterparts


PAPER_TAC = {"ACC": 91.9, "NMI": 83.3, "ARI": 83.1}


class TACHeads(nn.Module):
    def __init__(self, clusters: int = 10):
        super().__init__()

        def head():
            layers = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                   nn.Linear(512, clusters), nn.Softmax(dim=1))
            nn.init.trunc_normal_(layers[0].weight, std=0.02)
            nn.init.trunc_normal_(layers[3].weight, std=0.02)
            return layers

        self.text_head = head()
        self.image_head = head()

    def forward(self, text: torch.Tensor, image: torch.Tensor):
        return self.text_head(text), self.image_head(image)


def cluster_distillation(anchor: torch.Tensor, neighbor: torch.Tensor,
                         temperature: float) -> torch.Tensor:
    """InfoNCE across K cluster-assignment columns, matching paper Eq. 9-11."""
    k = anchor.shape[1]
    columns = F.normalize(torch.cat((anchor.T, neighbor.T), dim=0), dim=1)
    similarity = columns @ columns.T / temperature
    positive = torch.cat((similarity.diag(k), similarity.diag(-k))).unsqueeze(1)
    mask = torch.ones((2 * k, 2 * k), dtype=torch.bool, device=anchor.device)
    mask.fill_diagonal_(False)
    idx = torch.arange(k, device=anchor.device)
    mask[idx, idx + k] = False
    mask[idx + k, idx] = False
    negative = similarity[mask].reshape(2 * k, -1)
    logits = torch.cat((positive, negative), dim=1)
    return F.cross_entropy(logits, torch.zeros(2 * k, dtype=torch.long, device=anchor.device))


def batch_loss(model: TACHeads, text: torch.Tensor, image: torch.Tensor,
               neighbor_text: torch.Tensor, neighbor_image: torch.Tensor,
               temperature: float, balance_weight: float):
    p_text, p_image = model(text, image)
    pn_text, pn_image = model(neighbor_text, neighbor_image)
    distill = (cluster_distillation(p_image, pn_text, temperature)
               + cluster_distillation(p_text, pn_image, temperature))
    agreement = (p_text * p_image).sum(dim=1).clamp(1e-9, 1 - 1e-9)
    confidence = F.binary_cross_entropy(agreement, torch.ones_like(agreement))
    entropy = sum(-torch.sum(p.mean(dim=0) * torch.log(p.mean(dim=0).clamp_min(1e-9)))
                  for p in (p_text, p_image))
    return distill + confidence - balance_weight * entropy, {
        "distill": float(distill.detach()),
        "confidence": float(confidence.detach()),
        "entropy": float(entropy.detach()),
    }


def neighbors_for(name: str, features: np.ndarray, cache: Path, topk: int) -> np.ndarray:
    path = cache / f"tac_{name}_neighbors{topk}.npy"
    if path.exists():
        result = np.load(path)
        if result.shape != (len(features), topk):
            raise RuntimeError(f"Neighbor cache shape mismatch: {path}")
        return result
    features = np.ascontiguousarray(l2_normalize(features).astype(np.float32))
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    result = np.empty((len(features), topk), dtype=np.int32)
    for start in range(0, len(features), 1024):
        end = min(start + 1024, len(features))
        _, candidates = index.search(features[start:end], topk + 1)
        for row, own in enumerate(range(start, end)):
            result[own] = candidates[row][candidates[row] != own][:topk]
        print(f"neighbors/{name}: {end}/{len(features)}", flush=True)
    np.save(path, result)
    return result


def train_seed(seed: int, train_text: np.ndarray, train_image: np.ndarray,
               test_image: np.ndarray, test_labels: np.ndarray,
               text_neighbors: np.ndarray, image_neighbors: np.ndarray,
               args, checkpoints: Path, device: torch.device):
    torch.manual_seed(seed)
    model = TACHeads().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    checkpoint = checkpoints / f"seed{seed}.pt"
    start_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["completed_epochs"])
        if start_epoch > args.epochs:
            raise RuntimeError("Checkpoint has more epochs than requested; use a new checkpoint directory")
        print(f"seed {seed}: resuming at epoch {start_epoch}", flush=True)
    train_text_t = torch.from_numpy(train_text)
    train_image_t = torch.from_numpy(train_image)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 100_000 + epoch)
        order = torch.randperm(len(train_image), generator=generator).numpy()
        rng = np.random.default_rng(seed * 100_000 + epoch)
        totals = {"distill": 0.0, "confidence": 0.0, "entropy": 0.0}
        batches = 0
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            ids = order[start : start + args.batch_size]
            text_ids = text_neighbors[ids, rng.integers(args.topk, size=len(ids))]
            image_ids = image_neighbors[ids, rng.integers(args.topk, size=len(ids))]
            tensors = [array[index].to(device) for array, index in (
                (train_text_t, ids), (train_image_t, ids),
                (train_text_t, text_ids), (train_image_t, image_ids))]
            loss, terms = batch_loss(model, *tensors, args.temperature, args.balance_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name, value in terms.items():
                totals[name] += value
            batches += 1
        print(f"seed {seed} epoch {epoch + 1}/{args.epochs}: "
              + ", ".join(f"{name}={value / batches:.4f}" for name, value in totals.items()),
              flush=True)
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "completed_epochs": epoch + 1}, tmp)
        tmp.replace(checkpoint)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(test_image), args.batch_size):
            x = torch.from_numpy(test_image[start : start + args.batch_size]).to(device)
            predictions.append(model.image_head(x).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(predictions)
    scores = clustering_metrics(test_labels, pred)
    print(f"seed {seed} final: {scores}", flush=True)
    return {"seed": seed, **scores}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "tac_training")
    parser.add_argument("--prompt-set", choices=("single", "seven"), default="seven")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--balance-weight", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.topk < 1:
        raise ValueError("epochs >= 1, batch-size >= 2, and topk >= 1 are required")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    train_image = l2_normalize(np.load(args.cache / "cifar10_image_train.npy").astype(np.float32))
    test_image = l2_normalize(np.load(args.cache / "cifar10_image_test.npy").astype(np.float32))
    test_labels = np.load(args.cache / "cifar10_labels_test.npy")
    suffix = "paper-single_faiss" if args.prompt_set == "single" else "authors-seven_trimmed_faiss"
    selected = np.load(args.cache / f"cifar10_selected_noun_features_{suffix}_seed0.npy")
    text_path = args.cache / f"cifar10_retrieved_text_train_{suffix}.npy"
    if text_path.exists():
        train_text = np.load(text_path)
        if train_text.shape != train_image.shape:
            raise RuntimeError(f"Cached text feature shape mismatch: {text_path}")
    else:
        train_text = retrieve_text_counterparts(train_image, selected)
        np.save(text_path, train_text)
    train_text = l2_normalize(train_text.astype(np.float32))
    checkpoint_dir = args.cache / (
        f"tac_checkpoints_{args.prompt_set}_b{args.batch_size}_k{args.topk}"
        f"_t{args.temperature:g}_bal{args.balance_weight:g}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    text_neighbors = neighbors_for(f"text_{args.prompt_set}", train_text, args.cache, args.topk)
    image_neighbors = neighbors_for("image", train_image, args.cache, args.topk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [train_seed(seed, train_text, train_image, test_image, test_labels,
                       text_neighbors, image_neighbors, args, checkpoint_dir, device)
            for seed in args.seeds]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": {"prompt_set": args.prompt_set, "epochs": args.epochs,
                     "batch_size": args.batch_size, "topk": args.topk,
                     "temperature": args.temperature, "balance_weight": args.balance_weight,
                     "seeds": args.seeds, "device": str(device)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "faiss": faiss.__version__, "elapsed_seconds": time.perf_counter() - started},
        "summary": {metric: {"mean": float(np.mean([row[metric] for row in rows])),
                             "std": float(np.std([row[metric] for row in rows], ddof=1))
                             if len(rows) > 1 else 0.0,
                             "paper": PAPER_TAC[metric]}
                    for metric in ("ACC", "NMI", "ARI")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"Saved training results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
