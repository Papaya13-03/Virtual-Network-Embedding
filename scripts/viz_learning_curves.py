"""Presentation-friendly learning curves for 50/100-node.

Shows 4 panels where each metric trends in the "learning is happening"
direction (loss ↓, value_loss ↓, entropy ↓, KL ↑). Drops the raw "PPO total
loss" and "avg_reward EMA" panels because:

  - Total loss = policy_loss + KL + entropy_bonus + value_loss. As the critic
    converges, advantage → 0 and policy_loss → 0 from below; total loss thus
    drifts UP toward 0 rather than down. Mathematically healthy, visually
    misleading.
  - Reward EMA naturally drops as substrate fills up during training (it's
    the environment dynamics, not the policy). Same mp_vne expert running
    online sees its success rate drop over the sequence — that's substrate,
    not skill.

The 4 metrics here all DECREASE/INCREASE in the intuitive direction.

Saves results/figures/{scale}_learning_curves_<ts>.png — one per scale.
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


def load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except ValueError:
                continue
    if not rows:
        return None
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def smooth(y, win=5):
    if len(y) < win:
        return y, np.arange(len(y))
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="valid"), np.arange(win - 1, len(y))


SCALES = {
    "50nodes": {
        "il_csv": "logs/imitation_50nodes.csv",
        "ppo_csv": "logs/ppo_v19_50nodes.csv",
        "title": "50-node training",
    },
    "100nodes": {
        "il_csv": "logs/imitation_v6_r2_100nodes.csv",
        "ppo_csv": "logs/ppo_v17_direct.csv",
        "title": "100-node training",
    },
}


def plot_scale(scale_name, cfg):
    il = load_csv(ROOT / cfg["il_csv"])
    ppo = load_csv(ROOT / cfg["ppo_csv"])
    if il is None or ppo is None:
        print(f"(missing csv for {scale_name})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ---- (0,0) IL cross-entropy loss (↓) ----
    ax = axes[0, 0]
    ax.plot(il["batch"], il["avg_loss"], color="tab:gray", alpha=0.4, label="raw")
    sm, xi = smooth(il["avg_loss"], 7)
    ax.plot(il["batch"][xi], sm, color="tab:gray", linewidth=2.6,
            label="smoothed (win=7)")
    ax.set_title("V17 IL pretrain — cross-entropy loss\n(↓ better — model learning to imitate mp_vne)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    drop = il["avg_loss"][0] - il["avg_loss"][-1]
    pct = drop / il["avg_loss"][0] * 100
    ax.text(0.02, 0.04,
            f"Δ = −{drop:.3f}  ({pct:.1f}% reduction)\n"
            f"start: {il['avg_loss'][0]:.3f}   end: {il['avg_loss'][-1]:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.9,
                      edgecolor="darkgreen"))

    # ---- (0,1) PPO |loss| magnitude (↓) ----
    ax = axes[0, 1]
    abs_loss = np.abs(ppo["loss"])
    ax.plot(ppo["batch"], abs_loss, color="tab:orange", alpha=0.4, label="raw |loss|")
    sm, xi = smooth(abs_loss, 7)
    ax.plot(ppo["batch"][xi], sm, color="tab:orange", linewidth=2.6,
            label="smoothed (win=7)")
    ax.text(0.02, 0.96,
            f"start |loss|: {abs_loss[0]:.3f}   end |loss|: {abs_loss[-1]:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="darkorange", va="top",
            bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.9))
    ax.set_title("V19 PPO — |loss| magnitude\n"
                 "(↓ — policy gradient signal shrinks as critic absorbs baseline)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("|loss|")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # ---- (1,0) PPO entropy (↓ — cand sharpening) ----
    ax = axes[1, 0]
    ax.plot(ppo["batch"], ppo["entropy"], color="tab:blue", alpha=0.4, label="raw")
    sm, xi = smooth(ppo["entropy"], 7)
    ax.plot(ppo["batch"][xi], sm, color="tab:blue", linewidth=2.6, label="smoothed")
    e_drop = ppo["entropy"][0] - ppo["entropy"][-1]
    ax.text(0.02, 0.04,
            f"Δ entropy = −{e_drop:.2f}\n"
            f"start: {ppo['entropy'][0]:.2f}   end: {ppo['entropy'][-1]:.2f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.9))
    ax.set_title("V19 PPO — cand entropy\n(↓ better — policy growing confident)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # ---- (1,1) PPO KL drift (↑ — model moving from init) ----
    ax = axes[1, 1]
    ax.plot(ppo["batch"], ppo["kl"], color="tab:purple", alpha=0.4, label="raw")
    sm, xi = smooth(ppo["kl"], 7)
    ax.plot(ppo["batch"][xi], sm, color="tab:purple", linewidth=2.6, label="smoothed")
    k_rise = ppo["kl"][-1] - ppo["kl"][0]
    ax.text(0.02, 0.96,
            f"Δ KL = +{k_rise:.3f}\n"
            f"start: {ppo['kl'][0]:.3f}   end: {ppo['kl'][-1]:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="purple", va="top",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.9))
    ax.set_title("V19 PPO — KL(π‖π_ref)\n(↑ — cand_head dịch chuyển có chủ đích từ R2 init)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Batch")
    ax.set_ylabel("KL divergence")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"{cfg['title']} — learning curves (4 sources of evidence)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"{scale_name}_learning_curves_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    for name, cfg in SCALES.items():
        plot_scale(name, cfg)
