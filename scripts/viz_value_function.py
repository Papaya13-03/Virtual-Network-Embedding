"""Visualize the value function V(s) — critic in PPO training.

For normal-reward training (200 epochs), shows:
  - mean_value_loss per epoch (MSE between V(s) predictions and observed rewards)
  - mean_advantage per epoch (reward − V(s)) — should center around 0
  - Comparison to mean_reward to show V(s) tracks reward target
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

PHASES = [
    ("logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0),
    ("logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10),
    ("logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20),
    ("logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30),
    ("logs/ppo_v19_50nodes_10ep_cont4_epoch_summary.csv", 50),
    ("logs/ppo_v19_50nodes_10ep_cont5_epoch_summary.csv", 80),
    ("logs/ppo_v19_50nodes_10ep_cont6_epoch_summary.csv", 130),
    ("logs/ppo_v19_50nodes_10ep_cont7_epoch_summary.csv", 150),
]


def load(path, off=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["epoch"] = data["epoch"] + off
    return data


def smooth(a, w=10):
    return np.array([a[max(0, i - w + 1):i + 1].mean() for i in range(len(a))])


def main():
    parts = []
    for p, off in PHASES:
        f = ROOT / p
        if f.exists():
            parts.append(load(f, off))

    epochs = np.concatenate([d["epoch"] for d in parts])
    # mean_value_loss is column #7 (0-indexed 6) — appeared after we extended
    # logging. Phases 1-5 (ep 1-80) used OLD csv schema without it. Just guard.
    vloss_list = []
    for d in parts:
        if "mean_value_loss" in d:
            vloss_list.append(d["mean_value_loss"])
        else:
            # Old schema: no value_loss column → fill with NaN.
            vloss_list.append(np.full_like(d["mean_reward"], np.nan))
    vloss = np.concatenate(vloss_list)
    reward = np.concatenate([d["mean_reward"] for d in parts])
    # Advantage column may also be missing in old runs.
    adv_list = []
    for d in parts:
        if "mean_advantage" in d:
            adv_list.append(d["mean_advantage"])
        else:
            adv_list.append(np.full_like(d["mean_reward"], np.nan))
    advantage = np.concatenate(adv_list)
    succ = np.concatenate([d["succ_rate"] for d in parts]) * 100

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    # (0,0) Value loss
    ax = axes[0, 0]
    mask = ~np.isnan(vloss)
    ax.plot(epochs[mask], vloss[mask], "o", color="tab:orange", markersize=3,
            alpha=0.4, label="per-epoch")
    if mask.sum() > 5:
        sm = smooth(vloss[mask], 10)
        ax.plot(epochs[mask], sm, "-", color="darkorange", linewidth=2.5,
                label="rolling avg (w=10)")
    ax.set_title("(A) Value loss — MSE( V(s), reward )",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean value loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "↓ better — critic dần predict đúng reward thật.\n"
            "Plateau ≠ 0 vì reward có noise (fail = −1 vs success = +1.x).",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.92))

    # (0,1) Reward vs V(s) — sanity check
    ax = axes[0, 1]
    ax.plot(epochs, reward, "-", color="tab:purple", linewidth=2.0,
            label="mean reward (target)")
    # Estimate V(s) from advantage = reward - V(s)  ⇒  V(s) = reward - adv
    v_est = reward - advantage
    if (~np.isnan(advantage)).sum() > 5:
        ax.plot(epochs, v_est, "--", color="tab:cyan", linewidth=2.0,
                label="V(s) (= reward − advantage)")
    ax.set_title("(B) Reward vs V(s) prediction  — critic tracks target",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "V(s) bám sát reward → advantage ≈ 0 trung bình.\n"
            "→ Critic làm baseline tốt, giảm variance gradient.",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.92))

    # (1,0) Advantage = reward - V(s)
    ax = axes[1, 0]
    ax.plot(epochs, advantage, "o", color="tab:blue", markersize=3, alpha=0.4,
            label="per-epoch advantage mean")
    if (~np.isnan(advantage)).sum() > 5:
        ax.plot(epochs, smooth(np.where(np.isnan(advantage), 0, advantage), 10),
                "-", color="darkblue", linewidth=2.5, label="rolling avg")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_title("(C) Advantage = reward − V(s)\n"
                 "(per-batch mean; ~0 = critic well-calibrated)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Advantage (per-batch mean)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "advantage được NORMALIZE trước backward\n"
            "→ mean ≈ 0, std ≈ 1 trong loss thực.\n"
            "Pre-normalize value gần 0 = critic chính xác.",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.92))

    # (1,1) Architecture diagram (text)
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("(D) Critic in PPO total loss",
                 fontsize=12, fontweight="bold")
    ax.text(0.01, 0.95,
            "--- ARCHITECTURE ---\n"
            "shared encoder\n"
            "    |\n"
            "    v\n"
            "value_head:  MLP( 64 -> 32 -> 1 )  ->  V(s) in R\n\n"
            "--- ROLE IN LOSS ---\n"
            "advantage  = reward - V(s).detach()\n"
            "policy_loss = - E[ log pi(a|s) * normalized(advantage) ]\n"
            "value_loss  = MSE( V(s), reward )\n"
            "total_loss  = policy_loss\n"
            "            + 0.5  * value_loss      <- critic weight\n"
            "            - 0.01 * entropy\n"
            "            + 0.1  * KL(pi || pi_ref)\n\n"
            "--- INTUITION ---\n"
            "* V(s) = baseline = expected reward estimate\n"
            "  -> subtract baseline -> reduce gradient variance\n"
            "     (Williams 1992)\n"
            "* value_loss trains V(s) by SUPERVISED MSE\n"
            "  -> no TD bootstrap (single-step return)\n"
            "* value_coef = 0.5 is the PPO paper default",
            transform=ax.transAxes, fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.92))

    fig.suptitle(
        "V19 Normal-reward PPO — Value function V(s) diagnostics (200 epochs)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_value_function_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
