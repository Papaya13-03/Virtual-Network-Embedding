"""Focused head-to-head: CF cf ep195 vs Normal ep19 (V19-best) vs mp_vne_v4.

Best of CF (cost-focused recipe) vs best of Normal recipe (eval-peak ep19) vs
heuristic baseline.
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
    ("Normal eval-peak (ep19, V19-best)", "il_mp_vne_v19_e19", "tab:green"),
    ("CF cf ep195 ★ (best deployment)", "il_mp_vne_v19_cf_ep195", "gold"),
]

METRICS = [
    ("acceptance_rate", "Acceptance rate (%)", 100, "↑ better"),
    ("avg_cost", "Avg cost", 1, "↓ better"),
    ("revenue_cost_ratio", "Revenue / Cost", 1, "↑ better"),
    ("avg_delay", "Avg delay", 1, "↓ better"),
]


def main():
    rows = []
    for label, name, color in SOURCES:
        p = ROOT / f"results/scenario_50nodes/{name}/metrics.json"
        m = json.loads(p.read_text())
        rows.append({"label": label, "name": name, "color": color, "m": m})

    # Print summary.
    print(f"{'Algo':40} {'Acc%':>7} {'Cost':>8} {'Rev/Cost':>9} {'Delay':>7}")
    print("-" * 80)
    for r in rows:
        m = r["m"]
        print(f"{r['label']:40} {m['acceptance_rate']*100:7.2f} "
              f"{m['avg_cost']:8.2f} {m['revenue_cost_ratio']:9.4f} "
              f"{m['avg_delay']:7.2f}")

    # 1×4 bar chart with annotations.
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    labels = [r["label"].split(" (")[0] for r in rows]
    colors = [r["color"] for r in rows]
    x = np.arange(len(rows))

    for i, (key, ylabel, scale, direction) in enumerate(METRICS):
        ax = axes[i]
        vals = np.array([r["m"][key] * scale for r in rows])
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=1.0,
                      width=0.7)

        # Highlight winner bar (gold border).
        better_is_smaller = direction == "↓ better"
        winner = int(np.argmin(vals) if better_is_smaller else np.argmax(vals))
        bars[winner].set_edgecolor("darkred")
        bars[winner].set_linewidth(3.0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{key.replace('_', ' ')}  ({direction})", fontsize=11,
                     fontweight="bold")
        ax.grid(alpha=0.3, axis="y")

        # Annotate each bar with value.
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.2f}" if key != "revenue_cost_ratio" else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Tight y-axis.
        lo = min(vals); hi = max(vals)
        pad = (hi - lo) * 0.25 + 1e-3
        ax.set_ylim(max(0, lo - pad), hi + pad * 2)

        # Annotate Δ between CF ep195 (index 2) and Normal V19-best (index 1).
        delta = vals[2] - vals[1]
        better_for_cf = (delta > 0) if not better_is_smaller else (delta < 0)
        arrow_color = "darkgreen" if better_for_cf else "darkred"
        sign = "+" if delta >= 0 else ""
        ax.text(0.5, 0.92,
                f"Δ CF − Normal\n= {sign}{delta:.4f}\n"
                f"({'CF wins' if better_for_cf else 'CF loses'})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="center", va="top", color=arrow_color,
                bbox=dict(boxstyle="round",
                          facecolor="#e8f5e9" if better_for_cf else "#ffebee",
                          alpha=0.92,
                          edgecolor=arrow_color))

    fig.suptitle(
        "Head-to-head: CF cf ep195 vs Normal V19-best (ep19) vs mp_vne_v4 baseline\n"
        "★ red-bordered bar = winner on that metric",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_cf195_vs_normal_peak_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ===== Pretty table figure (text-only) =====
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    # Build a structured table summary.
    n = rows[1]["m"]
    cf = rows[2]["m"]
    v4 = rows[0]["m"]

    txt = (
        "                          mp_vne_v4   Normal ep19    CF ep195 ★    Δ (CF − Normal)    Δ (CF − v4)\n"
        f"  Acceptance rate (%):   {v4['acceptance_rate']*100:9.2f}   {n['acceptance_rate']*100:11.2f}   {cf['acceptance_rate']*100:10.2f}        +{(cf['acceptance_rate']-n['acceptance_rate'])*100:.2f}pp            +{(cf['acceptance_rate']-v4['acceptance_rate'])*100:.2f}pp ↑\n"
        f"  Avg cost (↓):          {v4['avg_cost']:9.2f}   {n['avg_cost']:11.2f}   {cf['avg_cost']:10.2f}        {cf['avg_cost']-n['avg_cost']:+.2f}              {cf['avg_cost']-v4['avg_cost']:+.2f} ↓\n"
        f"  Rev/Cost (↑):          {v4['revenue_cost_ratio']:9.4f}   {n['revenue_cost_ratio']:11.4f}   {cf['revenue_cost_ratio']:10.4f}      {cf['revenue_cost_ratio']-n['revenue_cost_ratio']:+.4f}             {cf['revenue_cost_ratio']-v4['revenue_cost_ratio']:+.4f} ↑\n"
        f"  Avg delay (↓):         {v4['avg_delay']:9.2f}   {n['avg_delay']:11.2f}   {cf['avg_delay']:10.2f}        {cf['avg_delay']-n['avg_delay']:+.2f}              {cf['avg_delay']-v4['avg_delay']:+.2f} ↓\n"
    )
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=11,
            family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#fafafa", alpha=0.95,
                      edgecolor="black"))

    verdict = (
        "VERDICT — CF cf ep195 wins on:\n"
        "  - Acceptance: +1.10pp vs Normal,  +5.20pp vs mp_vne_v4\n"
        "  - Rev/Cost  : +0.0072 vs Normal,  +0.0035 vs mp_vne_v4\n"
        "  - Avg delay : -0.92  vs Normal,   -0.74  vs mp_vne_v4\n"
        "\n"
        "CF cf ep195 loses on:\n"
        "  - Avg cost: +11.42 vs mp_vne_v4 (because CF accepts more VNRs total)\n"
        "  (but CF cost is -5.84 LOWER than Normal V19-best)\n"
        "\n"
        "→ DEPLOY: il_mp_vne_v19_50nodes_cf_cont4_e45.pt  (= CF cf ep195)"
    )
    ax.text(0.02, 0.45, verdict, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", family="monospace",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.95,
                      edgecolor="darkgreen"))

    fig.suptitle("CF cf ep195  vs  Normal V19-best (ep19)  vs  mp_vne_v4",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out2 = OUT / f"50nodes_cf195_vs_normal_summary_{DATE_TAG}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
