"""Visualize metric comparison: mp_vne vs v10 vs v14 on scenario_100nodes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = Path("results/scenario_100nodes")
OUTPUT_DIR = Path("docs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = [
    ("mp_vne_v2",         "mp_vne (paper)",      "#aaaaaa"),
    ("mp_vne",            "mp_vne (per-dom)",    "#666666"),
    ("il_mp_vne_v10_pso", "V10 (multi-restart)", "#2F7DC1"),
    ("il_mp_vne_v14_pso", "V14 (hybrid top-K)",  "#E08020"),
    ("il_mp_vne_v16_pso", "V16 (3 fixes)",       "#CC3344"),
]

METRICS = [
    ("acceptance_rate",     "Acceptance Rate",       "higher",  lambda v: f"{v*100:.2f}%"),
    ("avg_cost",            "Avg Embedding Cost",    "lower",   lambda v: f"{v:.1f}"),
    ("revenue_cost_ratio",  "Revenue/Cost Ratio",    "higher",  lambda v: f"{v:.4f}"),
    ("avg_delay",           "Avg Delay",             "lower",   lambda v: f"{v:.2f}"),
]


def load_metrics():
    data = {}
    for algo_id, label, color in ALGORITHMS:
        path = RESULT_DIR / algo_id / "metrics.json"
        with open(path) as f:
            data[algo_id] = json.load(f)
    return data


def main():
    data = load_metrics()

    # --- Figure with 4 subplots, one per metric ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Algorithm Comparison on scenario_100nodes (3000 VNRs)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    for ax, (key, title, direction, fmt) in zip(axes.flat, METRICS):
        labels = [a[1] for a in ALGORITHMS]
        colors = [a[2] for a in ALGORITHMS]
        values = [data[a[0]][key] for a in ALGORITHMS]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)

        # Annotate
        for bar, v in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                fmt(v),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        # Highlight best
        best_idx = int(np.argmax(values)) if direction == "higher" else int(np.argmin(values))
        bars[best_idx].set_edgecolor("#117733")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(f"{title}  (↑ better)" if direction == "higher" else f"{title}  (↓ better)",
                     fontsize=12)
        ax.set_ylabel(title, fontsize=10)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add some headroom above bars
        ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "comparison_v10_v14_vs_mpvne.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # --- Per-metric normalized improvement chart (% vs mp_vne) ---
    fig2, ax2 = plt.subplots(figsize=(11, 5.5))
    width = 0.27
    x = np.arange(len(METRICS))

    # Compare each NN variant against mp_vne_v2 (paper-accurate baseline)
    mp_vals = [data["mp_vne_v2"][m[0]] for m in METRICS]

    for offset, (algo_id, label, color) in enumerate([
        ALGORITHMS[2], ALGORITHMS[3], ALGORITHMS[4],
    ]):
        deltas = []
        for (key, title, direction, _), mp_v in zip(METRICS, mp_vals):
            v = data[algo_id][key]
            if direction == "higher":
                delta_pct = (v - mp_v) / mp_v * 100
            else:
                # Lower is better → invert so positive means improvement
                delta_pct = (mp_v - v) / mp_v * 100
            deltas.append(delta_pct)

        bars = ax2.bar(x + (offset - 1) * width, deltas, width,
                       label=label, color=color, edgecolor="black", linewidth=0.6)
        for bar, d in zip(bars, deltas):
            ax2.annotate(
                f"{d:+.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if d >= 0 else -14),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold",
                color="#117733" if d > 0 else "#cc3333",
            )

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[1] for m in METRICS], fontsize=10)
    ax2.set_ylabel("% improvement over mp_vne (+ = better)", fontsize=11)
    ax2.set_title("Relative Improvement vs mp_vne (paper) Baseline",
                  fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "improvement_over_mpvne.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved: {out2}")

    # --- Print summary table ---
    print()
    print("=" * 78)
    headers = [a[1].split()[0] if "mp_vne" not in a[1] else a[1] for a in ALGORITHMS]
    hdr = " ".join(f"{h:>15}" for h in headers)
    print(f"{'Metric':<20}{hdr}{'Best':>15}")
    print("-" * (20 + 15 * (len(ALGORITHMS) + 1)))
    for key, title, direction, fmt in METRICS:
        row_vals = [data[a[0]][key] for a in ALGORITHMS]
        best_idx = int(np.argmax(row_vals)) if direction == "higher" else int(np.argmin(row_vals))
        best_label = ALGORITHMS[best_idx][1].split()[0]
        cells = " ".join(f"{fmt(v):>15}" for v in row_vals)
        print(f"{title:<20}{cells}{best_label:>15}")
    print("=" * (20 + 15 * (len(ALGORITHMS) + 1)))


if __name__ == "__main__":
    main()
