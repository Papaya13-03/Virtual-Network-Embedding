"""Visualize IL pretrain (cross-entropy supervision from MP_VNE expert)
on 50-node before PPO.

Reads logs/imitation_50nodes.csv (10k episodes, ~620 batches).
"""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    rows = []
    with open(ROOT / "logs/imitation_50nodes.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "batch": int(r["batch"]),
                "loss": float(r["avg_loss"]),
                "expert_succ": float(r["expert_succ"]),
                "match": float(r["matched_rate"]),
            })

    batch = np.array([r["batch"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    expert_succ = np.array([r["expert_succ"] for r in rows]) * 100
    match = np.array([r["match"] for r in rows]) * 100

    # Rolling smoothing for trend.
    def smooth(a, w=5):
        return np.array([a[max(0, i - w + 1):i + 1].mean() for i in range(len(a))])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) Loss
    ax = axes[0]
    ax.plot(batch, loss, "o-", color="tab:red", linewidth=1.5, markersize=5,
            alpha=0.55, label="per-batch avg loss")
    ax.plot(batch, smooth(loss), "-", color="darkred", linewidth=2.5,
            label="rolling avg (w=5)")
    ax.set_title("(A) IL cross-entropy loss\n(↓ better — learning expert's snode choices)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch (batch_size=16)")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.98, 0.97,
            f"first 50 batches: {loss[:50].mean():.3f}\n"
            f"last 50 batches : {loss[-50:].mean():.3f}\n"
            f"Δ: {loss[-50:].mean() - loss[:50].mean():+.3f}\n"
            f"(loss > 0 vì candidate pool có {{5+ snodes/domain × 10 domains}}\n"
            f" ⇒ random log_prob ≈ −log(1/50) ≈ 3.9)",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.9))

    # (1) Expert success rate (cumulative through episodes)
    ax = axes[1]
    ax.plot(batch, expert_succ, "o-", color="tab:green", linewidth=1.5,
            markersize=5, alpha=0.55, label="per-batch")
    ax.plot(batch, smooth(expert_succ), "-", color="darkgreen", linewidth=2.5,
            label="rolling avg")
    ax.set_title("(B) Expert success rate\n(MP_VNE solving rate on synthetic VNRs)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Expert success (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.97,
            f"Final: {expert_succ[-1]:.1f}%\n"
            f"(MP_VNE chỉ solve được ~45% VNR\n"
            f" trên 50-node training substrate)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.9))

    # (2) Match rate
    ax = axes[2]
    ax.plot(batch, match, "o-", color="tab:blue", linewidth=1.5, markersize=5,
            alpha=0.7, label="per-batch")
    ax.set_ylim(85, 102)
    ax.set_title("(C) Target match rate\n(expert's snode IN candidate pool)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Match rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.04,
            f"100% throughout — expert's chosen snode\n"
            f"luôn nằm trong cand_pool (top-K=5 / domain).\n"
            f"⇒ CE loss có gradient signal hợp lệ.",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.9))

    fig.suptitle(
        "V19 IL pretrain on 50-node — Cross-entropy supervision from MP_VNE expert "
        f"({len(rows)} batch points, 10000 episodes total)\n"
        "Goal: warm-start policy với behaviour của heuristic MP_VNE trước khi PPO",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_il_pretrain_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
