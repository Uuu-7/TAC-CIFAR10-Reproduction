"""Run a prespecified one-factor-at-a-time TAC ablation grid.

Each configuration writes its own CSV. Restarting skips completed files, so
long CPU retrieval from all WordNet nouns can be resumed at the config level.
No parameter is chosen from test metrics by this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def configurations(include_all_nouns: bool):
    baseline = {"centers": 167, "gamma": 5, "tau": 0.005, "mode": "soft"}
    yield "baseline", baseline
    for centers in (50, 100, 300):
        yield f"centers_{centers}", {**baseline, "centers": centers}
    for gamma in (1, 3, 10, 20):
        yield f"gamma_{gamma}", {**baseline, "gamma": gamma}
    for tau, label in ((0.002, "002"), (0.01, "010"), (0.02, "020")):
        yield f"tau_{label}", {**baseline, "tau": tau}
    yield "hard_retrieval", {**baseline, "mode": "hard"}
    if include_all_nouns:
        yield "all_nouns", {**baseline, "all_nouns": True}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "ablation_grid")
    parser.add_argument("--include-all-nouns", action="store_true",
                        help="Enable the most expensive WordNet-wide retrieval control.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--restarts", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, config in configurations(args.include_all_nouns):
        output = args.output_dir / f"{label}.csv"
        if output.exists() and output.stat().st_size > 0 and output.with_suffix(".config.json").exists() and not args.force:
            print(f"Skipping completed {label}", flush=True)
            continue
        command = [
            sys.executable, str(root / "cached_experiments.py"), "ablation",
            "--prompts", "seven", "--selection-backends", "faiss-cpu",
            "--eval-backend", "faiss-cpu", "--centers", str(config["centers"]),
            "--gamma", str(config["gamma"]), "--tau", str(config["tau"]),
            "--threads", str(args.threads), "--restarts", str(args.restarts),
            "--max-points-per-centroid", "1000", "--seeds", *map(str, args.seeds),
            "--output", str(output),
        ]
        if config.get("all_nouns"):
            command += ["--all-nouns", "--no-selected"]
        else:
            command += ["--modes", config["mode"]]
        print(f"Running {label}", flush=True)
        log = args.output_dir / f"{label}.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
