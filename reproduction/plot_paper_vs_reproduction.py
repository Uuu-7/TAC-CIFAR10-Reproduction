"""Visualize CIFAR-10 results reported by the paper and reproduced locally.

The script reads ``results/summary.json`` so a new experiment can refresh the
figure without copying numbers into the plotting code.

Examples
--------
python plot_paper_vs_reproduction.py
python plot_paper_vs_reproduction.py --output results/comparison.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("ACC", "NMI", "ARI")
NAVY = "#17366F"
LIGHT_NAVY = "#8EA6D0"
ORANGE = "#F07C2B"
LIGHT_ORANGE = "#F8BE91"
GREEN = "#65A844"
GRAY = "#667085"
GRID = "#D9DEE8"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare the paper's CIFAR-10 scores with the reproduction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "results" / "summary.json",
        help="Path to the reproduction summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "paper_vs_reproduction.png",
        help="Output path. PNG, PDF and SVG are supported by Matplotlib.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster output DPI.")
    parser.add_argument(
        "--show", action="store_true", help="Open an interactive window after saving."
    )
    return parser.parse_args()


def load_values(summary_path: Path) -> dict[str, np.ndarray]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload["summary"]
    image_only = summary["clip_image_kmeans"]
    image_text = summary["image_text_kmeans"]

    return {
        "paper_image": np.array([image_only[m]["paper"] for m in METRICS]),
        "ours_image": np.array([image_only[m]["mean"] for m in METRICS]),
        "ours_image_std": np.array([image_only[m]["std"] for m in METRICS]),
        "paper_text": np.array([image_text[m]["paper"] for m in METRICS]),
        "ours_text": np.array([image_text[m]["mean"] for m in METRICS]),
        "ours_text_std": np.array([image_text[m]["std"] for m in METRICS]),
        "paper_gain": np.array([summary["paper_improvement"][m] for m in METRICS]),
        "ours_gain": np.array([summary["improvement"][m] for m in METRICS]),
    }


def label_bars(ax: plt.Axes, bars, decimals: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#303846",
            fontweight="semibold",
        )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=GRAY)


def draw(values: dict[str, np.ndarray], output_path: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        }
    )

    fig, (ax_scores, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.3),
        gridspec_kw={"width_ratios": [1.75, 1]},
    )
    fig.patch.set_facecolor("white")

    # Panel A: four scores per metric make the baseline and guided comparison explicit.
    x = np.arange(len(METRICS))
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    bars = [
        ax_scores.bar(
            x + offsets[0], values["paper_image"], width,
            label="Paper · image-only", color=LIGHT_NAVY,
        ),
        ax_scores.bar(
            x + offsets[1], values["ours_image"], width,
            label="Reproduced · image-only", color=NAVY,
            yerr=values["ours_image_std"], capsize=2,
        ),
        ax_scores.bar(
            x + offsets[2], values["paper_text"], width,
            label="Paper · image-text", color=LIGHT_ORANGE,
        ),
        ax_scores.bar(
            x + offsets[3], values["ours_text"], width,
            label="Reproduced · image-text", color=ORANGE,
            yerr=values["ours_text_std"], capsize=2,
        ),
    ]
    for group in bars:
        label_bars(ax_scores, group)

    ax_scores.set_title("A. Absolute clustering scores", loc="left", pad=14)
    ax_scores.set_ylabel("Score (%)", color=GRAY)
    ax_scores.set_xticks(x, METRICS, fontweight="bold", color="#23304A")
    ax_scores.set_ylim(50, 98)
    ax_scores.set_yticks(np.arange(50, 101, 10))
    ax_scores.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    style_axis(ax_scores)

    # Panel B: compare the improvement caused by adding textual guidance.
    width_gain = 0.32
    paper_gain_bars = ax_gain.bar(
        x - width_gain / 2,
        values["paper_gain"],
        width_gain,
        label="Paper gain",
        color=LIGHT_ORANGE,
    )
    ours_gain_bars = ax_gain.bar(
        x + width_gain / 2,
        values["ours_gain"],
        width_gain,
        label="Reproduced gain",
        color=GREEN,
    )
    label_bars(ax_gain, paper_gain_bars)
    label_bars(ax_gain, ours_gain_bars)

    ax_gain.set_title("B. Improvement from text guidance", loc="left", pad=14)
    ax_gain.set_ylabel("Gain (percentage points)", color=GRAY)
    ax_gain.set_xticks(x, METRICS, fontweight="bold", color="#23304A")
    ax_gain.set_ylim(0, 22)
    ax_gain.set_yticks(np.arange(0, 23, 5))
    ax_gain.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    style_axis(ax_gain)

    fig.suptitle(
        "CIFAR-10: Paper Results vs. Our Reproduction",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#102F6B",
    )
    fig.text(
        0.055,
        0.925,
        "CLIP ViT-B/32 · K = 10 · Higher is better · Reproduction mean over five K-means seeds",
        ha="left",
        fontsize=10.5,
        color=GRAY,
    )
    fig.text(
        0.055,
        0.018,
        "Paper: Li et al. (ICML 2024). Reproduction: full WordNet, one prompt, scikit-learn CPU K-means proxy.",
        ha="left",
        fontsize=8.5,
        color=GRAY,
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved comparison figure to: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    values = load_values(args.input.resolve())
    draw(values, args.output.resolve(), args.dpi)
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
