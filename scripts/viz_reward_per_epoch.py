"""Reward per epoch — focused single-panel plot across 20 epochs."""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_csv(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["epoch"] = data["epoch"] + offset
    return data


def main():
    first = load_csv(ROOT / "logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0)
    cont = load_csv(ROOT / "logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10)
    epochs = np.concatenate([first["epoch"], cont["epoch"]])
    reward = np.concatenate([first["mean_reward"], cont["mean_reward"]])

    fig, ax = plt.subplots(figsize=(12, 6))
    # Continuous line through all 20 epochs (gray underline → no gap at ep10↔11).
    ax.plot(epochs, reward, "-", color="dimgray", linewidth=2.0, alpha=0.5,
            zorder=1)
    # Phase-coloured markers on top.
    ax.plot(first["epoch"], first["mean_reward"], "o",
            color="tab:red", markersize=10, markeredgecolor="darkred",
            markeredgewidth=1.0, label="first 10 ep", zorder=3)
    ax.plot(cont["epoch"], cont["mean_reward"], "s",
            color="tab:pink", markersize=10, markeredgecolor="darkred",
            markeredgewidth=1.0, label="continuation (11-20)", zorder=3)
    ax.axvline(10.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    # Zoom y-axis to show the actual range tightly so the small upward trend is
    # visible (reward range ~0.07, not the theoretical [-1, +1]).
    lo = reward.min()
    hi = reward.max()
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad)

    # Best epoch (least negative reward).
    best = int(np.argmax(reward))
    ax.scatter([epochs[best]], [reward[best]], s=300, marker="*",
               color="gold", edgecolors="black", linewidths=1.5, zorder=10,
               label=f"peak: ep {int(epochs[best])} = {reward[best]:+.3f}")

    # Trend annotation.
    delta = reward[-1] - reward[0]
    ax.text(0.02, 0.96,
            f"ep1: {reward[0]:+.3f}\n"
            f"ep20: {reward[-1]:+.3f}\n"
            f"peak ep{int(epochs[best])}: {reward[best]:+.3f}\n"
            f"Δ (ep1 → ep20): {delta:+.3f}  ({'↑ better' if delta > 0 else '↓ worse'})",
            transform=ax.transAxes, fontsize=11, va="top", fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.92,
                      edgecolor="darkred"))

    # Mark phase regions.
    ax.axvspan(0.5, 10.5, alpha=0.05, color="tab:red", label=None)
    ax.axvspan(10.5, 20.5, alpha=0.05, color="tab:pink", label=None)
    yr = ax.get_ylim()
    ytext = yr[0] + (yr[1] - yr[0]) * 0.04
    ax.text(5.5, ytext, "Phase 1: ref = R2 IL", ha="center", fontsize=9,
            color="tab:red", style="italic")
    ax.text(15.5, ytext, "Phase 2: ref = ep10 ckpt", ha="center", fontsize=9,
            color="tab:pink", style="italic")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean reward per epoch", fontsize=12)
    ax.set_title("V19 PPO on 50-node — mean reward per epoch (20 epochs)\n"
                 "Reward = +1+0.3·rel_cost (success) | −1 (fail), per-VNR averaged",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xticks(np.arange(1, 21))

    fig.tight_layout()
    out = OUT / f"50nodes_v19_reward_per_epoch_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
