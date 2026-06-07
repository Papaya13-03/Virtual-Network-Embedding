"""Compare cf-peak ckpt (cf_cont_e21 = global cf ep31) vs V19 best (ep19, normal reward) vs mp_vne_v4.

Reads metrics.json from results/scenario_50nodes/{il_mp_vne_v19_cf_peak, il_mp_vne_v19_e19, mp_vne_v4}.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

SOURCES = [
    ("mp_vne_v4 (heuristic)", "mp_vne_v4", "tab:red"),
    ("Normal ep19 (V19-best)", "il_mp_vne_v19_e19", "tab:green"),
    ("Normal ep189", "il_mp_vne_v19_normal_peak", "darkgreen"),
    ("Normal ep200 (final)", "il_mp_vne_v19_normal_ep200", "olive"),
    ("CF cf ep31", "il_mp_vne_v19_cf_peak", "tab:blue"),
    ("CF cf ep78", "il_mp_vne_v19_cf_peak2", "tab:cyan"),
    ("CF cf ep130", "il_mp_vne_v19_cf_ep130", "steelblue"),
    ("CF cf ep195 ★", "il_mp_vne_v19_cf_ep195", "gold"),
    ("CF cf ep198", "il_mp_vne_v19_cf_peak3", "tab:purple"),
    ("CF cf ep200 (final)", "il_mp_vne_v19_cf_ep200", "indigo"),
]

METRICS = [
    ("acceptance_rate", "Acceptance rate (%)", 100, "↑ better"),
    ("avg_cost", "Avg cost (↓ better)", 1, "↓ better"),
    ("revenue_cost_ratio", "Revenue / Cost (↑ better)", 1, "↑ better"),
    ("avg_delay", "Avg delay (↓ better)", 1, "↓ better"),
]


def main():
    rows = []
    for label, name, color in SOURCES:
        p = ROOT / f"results/scenario_50nodes/{name}/metrics.json"
        if not p.exists():
            print(f"missing: {p}")
            continue
        m = json.loads(p.read_text())
        rows.append({"label": label, "name": name, "color": color, "m": m})

    # Print summary table.
    print(f"{'Algo':45} {'Acc%':>7} {'Cost':>8} {'Rev/Cost':>9} {'Delay':>7}")
    print("-" * 80)
    for r in rows:
        m = r["m"]
        print(f"{r['label']:45} {m['acceptance_rate']*100:7.2f} "
              f"{m['avg_cost']:8.2f} {m['revenue_cost_ratio']:9.4f} "
              f"{m['avg_delay']:7.2f}")

    # Bar grid: 4 metrics × N sources.
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    labels_short = [r["label"].split(" (")[0] for r in rows]
    colors = [r["color"] for r in rows]
    x = np.arange(len(rows))

    for i, (key, ylabel, scale, direction) in enumerate(METRICS):
        ax = axes[i]
        vals = np.array([r["m"][key] * scale for r in rows])
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{key.replace('_', ' ')}  ({direction})", fontsize=11,
                     fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        # Annotate bars.
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.2f}" if key != "revenue_cost_ratio" else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Trim y-range so differences are visible.
        lo = min(vals); hi = max(vals)
        pad = (hi - lo) * 0.18 + 1e-3
        ax.set_ylim(max(0, lo - pad), hi + pad * 3)

    fig.suptitle(
        "50-node EVAL (test set, 3000 VNRs) — all trained ckpts vs MP_VNE baseline\n"
        "★ CF cf ep195 = NEW BEST overall (acc 34.23%, rc 0.317, delay 24.19)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_cf_peak_compare_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
