"""Per-epoch learning curves from epoch_summary.csv.

Each epoch trains the same VNRs but on RESET substrate state. Mean metrics
per epoch are directly comparable across epochs → clean learning curves.

Saves results/figures/{name}_epoch_curves_<ts>.png.
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

CSV_PATH = ROOT / "logs" / "ppo_v19_50nodes_10ep_epoch_summary.csv"
OUT_NAME = "50nodes_v19_10ep"


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def main():
    d = load_csv(CSV_PATH)
    epochs = d["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (0,0) Success rate (↑)
    ax = axes[0, 0]
    ax.plot(epochs, d["succ_rate"] * 100, "o-", color="tab:green",
            linewidth=2.5, markersize=8)
    delta = (d["succ_rate"][-1] - d["succ_rate"][0]) * 100
    peak_idx = int(np.argmax(d["succ_rate"]))
    peak = d["succ_rate"][peak_idx] * 100
    ax.set_title("Online success rate per epoch  (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Success rate (%)")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.96,
            f"start: {d['succ_rate'][0]*100:.1f}%\n"
            f"end:   {d['succ_rate'][-1]*100:.1f}%\n"
            f"peak:  {peak:.1f}% (epoch {peak_idx+1})\n"
            f"Δ: +{delta:.1f}pp" if delta > 0 else
            f"start: {d['succ_rate'][0]*100:.1f}%\n"
            f"end:   {d['succ_rate'][-1]*100:.1f}%\n"
            f"peak:  {peak:.1f}% (epoch {peak_idx+1})\n"
            f"Δ: {delta:.1f}pp",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.9))

    # (0,1) Mean reward (↑)
    ax = axes[0, 1]
    ax.plot(epochs, d["mean_reward"], "o-", color="tab:red",
            linewidth=2.5, markersize=8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    rd = d["mean_reward"][-1] - d["mean_reward"][0]
    ax.set_title("Mean reward per epoch  (↑ better, toward 0)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.04,
            f"start: {d['mean_reward'][0]:+.3f}\n"
            f"end:   {d['mean_reward'][-1]:+.3f}\n"
            f"Δ: {rd:+.3f}",
            transform=ax.transAxes, fontsize=10, va="bottom", fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.9))

    # (0,2) Mean loss (→ 0 from below)
    ax = axes[0, 2]
    ax.plot(epochs, d["mean_loss"], "o-", color="tab:blue",
            linewidth=2.5, markersize=8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_title("Mean total loss per epoch  (→ 0 = converged)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss")
    ax.grid(alpha=0.3)
    abs_drop = abs(d["mean_loss"][0]) - abs(d["mean_loss"][-1])
    ax.text(0.02, 0.04,
            f"start: {d['mean_loss'][0]:+.3f}\n"
            f"end:   {d['mean_loss'][-1]:+.3f}\n"
            f"|loss| Δ: −{abs_drop:.3f}",
            transform=ax.transAxes, fontsize=10, va="bottom", fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.9))

    # (1,0) Mean policy_loss (→ 0)
    ax = axes[1, 0]
    ax.plot(epochs, d["mean_policy_loss"], "o-", color="tab:orange",
            linewidth=2.5, markersize=8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_title("Mean policy loss per epoch", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean policy_loss")
    ax.grid(alpha=0.3)

    # (1,1) Mean KL (↑ — model moving)
    ax = axes[1, 1]
    ax.plot(epochs, d["mean_kl"], "o-", color="tab:purple",
            linewidth=2.5, markersize=8)
    kd = d["mean_kl"][-1] - d["mean_kl"][0]
    ax.set_title("Mean KL(π‖π_ref) per epoch  (↑ — policy diverging from R2)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean KL")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.96,
            f"start: {d['mean_kl'][0]:.3f}\n"
            f"end:   {d['mean_kl'][-1]:.3f}\n"
            f"Δ: +{kd:.3f}",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="purple",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.9))

    # (1,2) Mean entropy (↓ — sharpening)
    ax = axes[1, 2]
    ax.plot(epochs, d["mean_entropy"], "o-", color="tab:cyan",
            linewidth=2.5, markersize=8)
    ed = d["mean_entropy"][-1] - d["mean_entropy"][0]
    ax.set_title("Mean entropy per epoch  (↓ — cand_head confident)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean entropy")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.96,
            f"start: {d['mean_entropy'][0]:.2f}\n"
            f"end:   {d['mean_entropy'][-1]:.2f}\n"
            f"Δ: {ed:+.2f}",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkcyan",
            bbox=dict(boxstyle="round", facecolor="#e0f7fa", alpha=0.9))

    fig.suptitle(
        "PPO learning curves on 50-node — substrate reset each epoch (10 epochs)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"{OUT_NAME}_epoch_curves_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
