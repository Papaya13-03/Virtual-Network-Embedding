"""Eval acceptance per epoch (true test-set performance over training).

Reads results/scenario_50nodes/il_mp_vne_v19_e{1..20}/metrics.json and plots:
  - Eval acceptance per epoch (test-set, not online).
  - Cost / rev-cost / delay across epochs.

Saves results/figures/50nodes_v19_eval_per_epoch_<ts>.png.
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

N_EPOCHS = 20
V4_BASELINE = 29.0   # mp_vne_v4 on 50-node


def main():
    rows = []
    for ep in range(1, N_EPOCHS + 1):
        p = ROOT / f"results/scenario_50nodes/il_mp_vne_v19_e{ep}/metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        rows.append({
            "epoch": ep,
            "acc": m["acceptance_rate"] * 100,
            "n_succ": m["n_success"],
            "cost": m["avg_cost"],
            "rev_cost": m["revenue_cost_ratio"],
            "delay": m["avg_delay"],
        })
    if not rows:
        print("no data")
        return

    epochs = np.array([r["epoch"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    cost = np.array([r["cost"] for r in rows])
    rev_cost = np.array([r["rev_cost"] for r in rows])
    delay = np.array([r["delay"] for r in rows])

    best_idx = int(np.argmax(acc))
    best_ep = epochs[best_idx]
    best_acc = acc[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def phase_boundary(ax):
        ax.axvline(10.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
                   label="continuation start")

    # (0,0) Acceptance ↑
    ax = axes[0, 0]
    ax.plot(epochs, acc, "o-", color="tab:green", linewidth=2.5, markersize=8,
            label="V19 per-epoch eval")
    ax.axhline(V4_BASELINE, color="tab:red", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"mp_vne_v4 baseline ({V4_BASELINE}%)")
    phase_boundary(ax)
    ax.scatter([best_ep], [best_acc], s=300, marker="*", color="gold",
               edgecolors="black", linewidths=1.5, zorder=10,
               label=f"best: ep {best_ep} = {best_acc:.2f}%")
    ax.set_title("Eval acceptance per epoch (test set, 3000 VNRs)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Acceptance rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.96,
            f"ep1: {acc[0]:.2f}%\n"
            f"ep20: {acc[-1]:.2f}%\n"
            f"best ep{best_ep}: {best_acc:.2f}%\n"
            f"Δ vs v4: +{best_acc-V4_BASELINE:.1f}pp",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.9))

    # (0,1) Cost ↓
    ax = axes[0, 1]
    ax.plot(epochs, cost, "o-", color="tab:blue", linewidth=2.5, markersize=8,
            label="V19 per-epoch eval")
    ax.axhline(216.5, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="mp_vne_v4 cost (216.5)")
    phase_boundary(ax)
    ax.set_title("Eval avg cost per epoch (↓ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg cost")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,0) Rev/Cost ↑
    ax = axes[1, 0]
    ax.plot(epochs, rev_cost, "o-", color="tab:purple", linewidth=2.5, markersize=8,
            label="V19 per-epoch eval")
    ax.axhline(0.3135, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="mp_vne_v4 rev/cost (0.314)")
    phase_boundary(ax)
    ax.set_title("Eval revenue / cost per epoch (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Revenue / Cost")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,1) Delay ↓
    ax = axes[1, 1]
    ax.plot(epochs, delay, "o-", color="tab:orange", linewidth=2.5, markersize=8,
            label="V19 per-epoch eval")
    ax.axhline(24.93, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="mp_vne_v4 delay (24.93)")
    phase_boundary(ax)
    ax.set_title("Eval avg delay per epoch (↓ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg delay")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(
        f"V19 on 50-node — 20 epochs eval (best: ep {best_ep} = {best_acc:.2f}%, "
        f"+{best_acc-V4_BASELINE:.1f}pp vs v4)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_v19_eval_per_epoch_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
