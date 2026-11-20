"""Generate README/report figures from committed results; no experiments run."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "figures"
METRICS = ("ACC", "NMI", "ARI")
NAVY, ORANGE, TEAL, GRAY = "#17366F", "#E98438", "#25877D", "#667085"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.titleweight": "bold", "savefig.facecolor": "white"})


def rows(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summary(name):
    return json.loads((ROOT / name / "summary.json").read_text(encoding="utf-8"))["summary"]


def stats(data, metric):
    values = np.array([float(r[metric]) for r in data])
    return values.mean(), values.std(ddof=1)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E9F0", linewidth=0.8)


def save(fig, name, title, subtitle):
    fig.suptitle(title, x=0.04, y=0.99, ha="left", fontsize=18,
                 color=NAVY, fontweight="bold")
    fig.text(0.04, 0.915, subtitle, color=GRAY, fontsize=10)
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.88), w_pad=2.5)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=190)
    plt.close(fig)
    print(f"Saved {name}.png / .svg", flush=True)


def stability():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    data = [(rows("stability_default.csv"), "Default cap", NAVY),
            (rows("stability_full_data.csv"), "Full-data cap", ORANGE)]
    for ax, metric in zip(axes, METRICS):
        for offset, (all_rows, label, color) in zip((-0.15, 0.15), data):
            for i, prompt in enumerate(("single", "seven")):
                subset = [r for r in all_rows if r["prompt"] == prompt]
                values = [float(r[metric]) for r in subset]
                center, sd = stats(subset, metric)
                x = i + offset
                ax.scatter(x + np.linspace(-0.055, 0.055, len(values)), values,
                           color=color, alpha=0.45, s=19)
                ax.errorbar(x, center, yerr=sd, fmt="D", color=color,
                            capsize=5, markersize=6,
                            label=label if i == 0 else None)
        ax.set(xticks=[0, 1], xticklabels=["Single prompt", "Seven prompts"],
               title=metric, ylabel="Score (%)")
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    save(fig, "sampling_stability", "FAISS sampling and seed stability",
         "Dots: individual seeds; diamonds: mean ± sample SD. Default cap: n=20; full-data cap: n=5.")


def factorial():
    data = rows("factorial_faiss.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, metric in zip(axes, METRICS):
        cells = [[stats([r for r in data if r["selection_prompt"] == a and
                   r["retrieval_prompt"] == b], metric)
                  for b in ("single", "seven")] for a in ("single", "seven")]
        means = np.array([[c[0] for c in row] for row in cells])
        ax.imshow(means, cmap="Blues", vmin=means.min()-3, vmax=means.max()+4,
                  aspect="auto")
        for i in range(2):
            for j in range(2):
                m, s = cells[i][j]
                ax.text(j, i, f"{m:.2f}\n± {s:.2f}", ha="center", va="center",
                        fontsize=13, color=NAVY)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Single", "Seven"],
               yticklabels=["Single", "Seven"], xlabel="Retrieval encoding",
               ylabel="Noun selection", title=f"{metric} (%)")
        ax.tick_params(length=0)
    save(fig, "prompt_factorial", "Where does the prompt ensemble act?",
         "Mean ± sample SD, n=5 seeds per cell. FAISS CPU, default sampling cap; exploratory CIFAR-10 analysis.")


def ablations():
    data = rows("ablation_grid/summary.csv")
    image = rows("ablation_image_only.csv")
    labels = {"baseline": "Baseline", "centers_50": "Centers: 50",
        "centers_100": "Centers: 100", "centers_300": "Centers: 300",
        "gamma_1": "Nouns/center: 1", "gamma_3": "Nouns/center: 3",
        "gamma_10": "Nouns/center: 10", "gamma_20": "Nouns/center: 20",
        "tau_002": "Temperature: 0.002", "tau_010": "Temperature: 0.010",
        "tau_020": "Temperature: 0.020", "hard_retrieval": "Hard retrieval"}
    order = ["baseline", "centers_50", "centers_100", "centers_300", "gamma_1",
             "gamma_3", "gamma_10", "gamma_20", "tau_002", "tau_010", "tau_020", "hard_retrieval"]
    by_name = {r["configuration"]: r for r in data}
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.4), sharey=True)
    ys = np.arange(13)
    for ax, metric in zip(axes, METRICS):
        means = [stats(image, metric)[0]] + [float(by_name[k][metric+"_mean"]) for k in order]
        sds = [stats(image, metric)[1]] + [float(by_name[k][metric+"_std"]) for k in order]
        for i, (mean, sd) in enumerate(zip(means, sds)):
            color = TEAL if i == 8 else NAVY if i == 1 else GRAY
            ax.errorbar(mean, i, xerr=sd, fmt="o", color=color, capsize=3, markersize=5)
        ax.axvline(means[1], color=NAVY, linestyle="--", alpha=0.3)
        ax.set(title=metric, xlabel="Score (%)", yticks=ys,
               yticklabels=["Image only"] + [labels[k] for k in order])
        ax.grid(axis="x", color="#E5E9F0")
    axes[0].invert_yaxis()
    save(fig, "component_ablations", "Text guidance and parameter sensitivity",
         "Mean ± sample SD, n=5. Dashed line: baseline; green: gamma=20. FAISS CPU, full-data cap; exploratory.")


def transfer():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (dataset, label) in enumerate((("cifar20", "CIFAR-20"), ("stl10", "STL-10"))):
        base = summary(dataset + "_transfer")
        changed = summary(dataset + "_transfer_gamma20")
        for ax, metric in zip(axes[row], METRICS):
            for i, weight in enumerate(("1.0", "0.75")):
                for offset, s, color, name in ((-0.13, base, NAVY, "Gamma 5"),
                                             (0.13, changed, ORANGE, "Gamma 20")):
                    v = s[weight][metric]
                    ax.errorbar(i+offset, v["mean"], yerr=v["std"], fmt="o",
                                color=color, capsize=5, markersize=6,
                                label=name if i == 0 else None)
                    ax.annotate(f"{v['mean']:.2f}", (i+offset, v["mean"]),
                                xytext=(0, 9 if offset > 0 else -17),
                                textcoords="offset points", ha="center", fontsize=8, color=color)
            ax.set(title=f"{label} · {metric}", xticks=[0, 1],
                   xticklabels=["Weight 1.0", "Weight 0.75"], ylabel="Score (%)")
            ax.margins(x=0.35, y=0.35)
            grid(ax)
    axes[0, 0].legend(fontsize=9, loc="upper left")
    save(fig, "transfer_validation", "Does a larger noun vocabulary transfer?",
         "Fixed gamma=5 vs 20, mean ± sample SD over the same 5 seeds; both text weights shown. Axes vary by panel.")


def training():
    data = rows("tac_training_20seeds/runs.csv")
    s = summary("tac_training_20seeds")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    seeds = [int(r["seed"]) for r in data]
    acc = [float(r["ACC"]) for r in data]
    axes[0].scatter(seeds, acc, color=[ORANGE if v < 85 else NAVY for v in acc], s=45, zorder=3)
    axes[0].axhline(s["ACC"]["paper"], color=TEAL, ls="--", label="Paper ACC: 91.90")
    axes[0].axhline(s["ACC"]["mean"], color=NAVY, ls=":", label=f"20-seed mean: {s['ACC']['mean']:.2f}")
    for seed, v in zip(seeds, acc):
        if v < 85:
            axes[0].annotate(f"Seed {seed}: {v:.2f}", (seed, v), xytext=(5, 10), textcoords="offset points", fontsize=9)
    axes[0].set(title="Every training seed is retained", xlabel="Training seed", ylabel="ACC (%)",
                xticks=range(0, 20, 2), ylim=(80, 94))
    axes[0].legend(loc="lower right", fontsize=9)
    for i, metric in enumerate(METRICS):
        v = s[metric]
        axes[1].errorbar(i-0.1, v["mean"], yerr=v["std"], fmt="o", color=NAVY,
                         capsize=5, label="Reproduction ± SD" if i == 0 else None)
        axes[1].scatter(i+0.1, v["paper"], marker="D", color=TEAL,
                         label="Paper reference" if i == 0 else None)
    axes[1].set(title="Full TAC vs paper", xticks=range(3), xticklabels=METRICS, ylabel="Score (%)", ylim=(76, 97))
    axes[1].legend(fontsize=9, loc="upper right")
    for ax in axes:
        grid(ax)
    save(fig, "training_stability", "Full TAC: 20-seed training reproduction",
         "20 epochs per seed; frozen feature/text caches. Two lower-performing runs remain in all reported means.")


def fusion():
    data = rows("fusion_ablation.csv")
    weights = sorted({float(r["text_weight"]) for r in data})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, metric in zip(axes, METRICS):
        values = [stats([r for r in data if float(r["text_weight"]) == w], metric) for w in weights]
        ax.errorbar(weights, [v[0] for v in values], yerr=[v[1] for v in values],
                    fmt="o-", color=NAVY, capsize=4)
        ax.axvline(1.0, color=GRAY, ls="--", alpha=0.5)
        ax.set(title=metric, xlabel="Text weight", ylabel="Score (%)", xticks=weights)
        grid(ax)
    save(fig, "fusion_weight", "Exploratory text-weight sweep on CIFAR-10",
         "Mean ± sample SD, n=5; scikit-learn Euclidean proxy. Dashed line: equal-weight fusion (1.0).")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for plot in (training, stability, factorial, ablations, transfer, fusion):
        plot()
