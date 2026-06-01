"""Training curves for R2 (IL pretrain) and V19 PPO (cand-RL direct).

R2 is IL (supervised) — has cross-entropy loss + expert_succ + match_rate
(no RL reward). V19 PPO has policy/total loss + per-batch reward EMA.

2x2 layout, IL on the left column, PPO on the right column:
  (0,0) R2 IL loss
  (0,1) V19 PPO loss
  (1,0) R2 IL match_rate (success indicator for IL)
  (1,1) V19 PPO avg reward

Saves results/figures/r2_vs_v19_loss_reward_<YYYYMMDD_HHMMSS>.png.
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

R2_IL_CSV = ROOT / "logs" / "imitation_v6_r2_100nodes.csv"
V19_PPO_CSV = ROOT / "logs" / "ppo_v17_direct.csv"


def load_csv(path):
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


def smooth(y, win=5):
    if len(y) < win:
        return y
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="valid")


def main():
    r2 = load_csv(R2_IL_CSV)
    ppo = load_csv(V19_PPO_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")

    # ---- (0,0) R2 IL loss ----
    ax = axes[0, 0]
    ax.plot(r2["batch"], r2["avg_loss"], color="tab:gray", linewidth=1.4,
            alpha=0.4, label="raw")
    sm = smooth(r2["avg_loss"], 7)
    if len(sm):
        ax.plot(r2["batch"][len(r2["batch"]) - len(sm):], sm,
                color="tab:gray", linewidth=2.2, label="smoothed (win=7)")
    ax.set_title("R2 — IL pretrain loss\n(cross-entropy on expert snode)")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    if len(r2["avg_loss"]):
        ax.text(0.02, 0.04,
                f"start={r2['avg_loss'][0]:.3f}   end={r2['avg_loss'][-1]:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # ---- (0,1) V19 PPO loss ----
    ax = axes[0, 1]
    ax.plot(ppo["batch"], ppo["loss"], color="tab:blue", linewidth=1.4,
            alpha=0.4, label="total loss (raw)")
    sm = smooth(ppo["loss"], 7)
    if len(sm):
        ax.plot(ppo["batch"][len(ppo["batch"]) - len(sm):], sm,
                color="tab:blue", linewidth=2.2, label="smoothed (win=7)")
    if "policy_loss" in ppo:
        ax.plot(ppo["batch"], ppo["policy_loss"],
                color="tab:orange", linewidth=1.0, alpha=0.5, label="policy loss")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_title("V19 — PPO (cand-RL direct) loss\n(policy + KL − entropy)")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # ---- (1,0) R2 IL match_rate (proxy "signal" for IL) ----
    ax = axes[1, 0]
    ax.plot(r2["batch"], r2["matched_rate"] * 100,
            color="tab:green", linewidth=2.2, label="match rate (expert snode in pool)")
    if "expert_succ" in r2:
        ax.plot(r2["batch"], r2["expert_succ"] * 100,
                color="tab:gray", linewidth=1.8, linestyle="--",
                label="expert success rate (mp_vne online)")
    ax.set_title("R2 IL — match rate (proxy 'reward')\nIL is supervised → no RL reward")
    ax.set_xlabel("Batch")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # ---- (1,1) V19 PPO avg reward + KL ----
    ax = axes[1, 1]
    ax.plot(ppo["batch"], ppo["avg_reward"],
            color="tab:red", linewidth=1.4, alpha=0.4, label="reward EMA (raw)")
    sm = smooth(ppo["avg_reward"], 7)
    if len(sm):
        ax.plot(ppo["batch"][len(ppo["batch"]) - len(sm):], sm,
                color="tab:red", linewidth=2.2, label="reward EMA (smoothed)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Avg reward (success bonus + 0.3·cost term)")
    ax.set_title("V19 PPO — reward EMA\n(success bonus + cost term; fail = −1)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    # KL on twin axis
    if "kl" in ppo:
        ax2 = ax.twinx()
        ax2.plot(ppo["batch"], ppo["kl"], color="tab:cyan",
                 linewidth=1.6, alpha=0.85, label="KL(π‖π_ref)")
        ax2.set_ylabel("KL", color="tab:cyan")
        ax2.tick_params(axis="y", labelcolor="tab:cyan")
        ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Training curves: R2 (IL) vs V19 (PPO direct cand-RL)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"r2_vs_v19_loss_reward_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
