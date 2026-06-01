#!/usr/bin/env python3
"""Visualize 5 metrics × N datasets comparing mp_vne vs il_mp_vne_pso.

Reads metrics.json from results/<scenario>/<algorithm>/metrics.json
and produces a grouped-bar figure with 5 subplots (one per metric).

Expected layout:
  results/
    scenario_100nodes/
      mp_vne/metrics.json
      il_mp_vne_pso/metrics.json
    scenario_200nodes/
      ...
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("acceptance_rate",   "Acceptance Rate",      "ratio"),
    ("revenue_rate",      "Long-term Revenue Rate", "rev/time"),
    ("avg_cost",          "Average Embedding Cost", "cost"),
    ("revenue_cost_ratio", "Revenue / Cost Ratio",  "ratio"),
    ("avg_delay",         "Average Delay",          "delay"),
]


def load_metrics(results_dir: Path, scenarios: list, algorithms: list):
    """Returns dict: data[algorithm][scenario] -> metrics dict."""
    data = {a: {} for a in algorithms}
    for scenario in scenarios:
        for algo in algorithms:
            mpath = results_dir / scenario / algo / "metrics.json"
            if not mpath.exists():
                print(f"WARN: missing {mpath}", file=sys.stderr)
                continue
            with open(mpath) as f:
                data[algo][scenario] = json.load(f)
    return data


def plot(data, scenarios, algorithms, output_path: Path):
    n_metrics = len(METRICS)
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(
        nrows=(n_metrics + 1) // 2, ncols=2,
        figsize=(13, 4 * ((n_metrics + 1) // 2)),
    )
    axes = axes.flatten()

    colors = {"mp_vne": "#888888", "il_mp_vne_pso": "#2E86AB"}
    labels = {"mp_vne": "MP-VNE (heuristic)", "il_mp_vne_pso": "IL-MP-VNE (proposed)"}

    x = np.arange(n_scenarios)
    width = 0.35

    for idx, (key, title, unit) in enumerate(METRICS):
        ax = axes[idx]
        for i, algo in enumerate(algorithms):
            vals = [data.get(algo, {}).get(s, {}).get(key, 0) for s in scenarios]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=labels.get(algo, algo),
                          color=colors.get(algo, None), edgecolor="black", linewidth=0.5)
            # Annotate values on bars
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f"{v:.2g}" if v >= 0.01 else f"{v:.3g}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Substrate size")
        ax.set_ylabel(unit)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("scenario_", "").replace("nodes", " nodes") for s in scenarios])
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9, loc="best")

    # Hide extra subplot if odd count
    if n_metrics < len(axes):
        for j in range(n_metrics, len(axes)):
            axes[j].axis("off")

    fig.suptitle("MP-VNE vs IL-MP-VNE (proposed) — 5 metrics × 4 scenarios",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--scenarios", nargs="+",
                   default=["scenario_100nodes", "scenario_200nodes",
                            "scenario_500nodes", "scenario_1000nodes"])
    p.add_argument("--algorithms", nargs="+",
                   default=["mp_vne", "il_mp_vne_pso"])
    p.add_argument("--output", default="results/comparison.png")
    args = p.parse_args()

    data = load_metrics(Path(args.results_dir), args.scenarios, args.algorithms)
    plot(data, args.scenarios, args.algorithms, Path(args.output))


if __name__ == "__main__":
    main()
