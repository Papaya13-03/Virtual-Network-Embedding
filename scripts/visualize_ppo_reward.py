"""PPO reward EMA per batch — V19 cand-RL direct on 100-node.

Single-panel line plot: raw avg_reward + smoothed curve. Reward formula:
    success → +1 + 0.3·(cost_baseline − cost)/cost_baseline
    fail    → −1
EMA per batch is the running mean used as the policy-gradient baseline.

Saves results/figures/ppo_reward_<YYYYMMDD_HHMMSS>.png.
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

PPO_CSV = ROOT / "logs" / "ppo_v17_direct.csv"


def load(path):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except ValueError:
                continue
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def smooth(y, win=7):
    if len(y) < win:
        return y, np.arange(len(y))
    kernel = np.ones(win) / win
    s = np.convolve(y, kernel, mode="valid")
    x = np.arange(win - 1, len(y))
    return s, x


def main():
    d = load(PPO_CSV)
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(d["batch"], d["avg_reward"],
            color="tab:red", linewidth=1.4, alpha=0.4, label="raw")
    s, xi = smooth(d["avg_reward"], 7)
    ax.plot(d["batch"][xi], s,
            color="tab:red", linewidth=2.6, label="smoothed (window=7)")

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(-1.0, color="tab:gray", linewidth=0.8, linestyle="--", alpha=0.5,
               label="all-fail floor (−1)")
    ax.axhline(1.0, color="tab:green", linewidth=0.8, linestyle="--", alpha=0.5,
               label="all-success ceiling (~+1)")

    # Endpoint annotation.
    if len(d["avg_reward"]):
        ax.annotate(
            f"end EMA = {d['avg_reward'][-1]:.3f}",
            xy=(d["batch"][-1], d["avg_reward"][-1]),
            xytext=(-110, -28), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1),
            fontsize=11, color="tab:red", fontweight="bold",
        )

    ax.set_xlabel("Batch")
    ax.set_ylabel("Avg reward EMA")
    ax.set_title("V19 PPO (cand-RL direct, 100-node) — reward EMA across training",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # Stats text-box.
    rewards = d["avg_reward"]
    text = (f"start: {rewards[0]:.3f}\n"
            f"min:   {rewards.min():.3f}\n"
            f"max:   {rewards.max():.3f}\n"
            f"final: {rewards[-1]:.3f}\n"
            f"mean:  {rewards.mean():.3f}")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="lightgray"))

    fig.tight_layout()
    out = OUT / f"ppo_reward_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
