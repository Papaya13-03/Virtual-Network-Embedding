"""130-epoch normal-reward training: convergence analysis + learning curves.

Stitches 6 phases (10+10+10+20+30+50 = 130 epochs) of normal-reward PPO
(REINFORCE+critic+KL). Shows per-epoch succ/reward/KL/entropy/loss and a
rolling-window convergence diagnostic.

Saves results/figures/50nodes_normal_130ep_<ts>.png.
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
    ("Phase 1: ep 1-10 (ref=R2 IL)",
     "logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0, "o", "tab:red"),
    ("Phase 2: ep 11-20 (ref=ep10)",
     "logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10, "s", "tab:pink"),
    ("Phase 3: ep 21-30 (ref=ep20)",
     "logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20, "^", "tab:orange"),
    ("Phase 4: ep 31-50 (ref=ep30)",
     "logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30, "D", "tab:purple"),
    ("Phase 5: ep 51-80 (ref=ep50)",
     "logs/ppo_v19_50nodes_10ep_cont4_epoch_summary.csv", 50, "v", "tab:cyan"),
    ("Phase 6: ep 81-130 (ref=ep80)",
     "logs/ppo_v19_50nodes_10ep_cont5_epoch_summary.csv", 80, "X", "tab:blue"),
]

V4_ACC = 29.03  # mp_vne_v4 eval acc
V19_BEST_EVAL = 33.13  # V19 best ckpt eval (= ep19 of phase 1+2)


def load(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    keys = rows[0].keys()
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data["epoch"] = data["epoch"] + offset
    return data


def main():
    phases = []
    for label, p, off, mk, col in PHASES:
        if (ROOT / p).exists():
            phases.append((label, load(ROOT / p, off), mk, col))
        else:
            print(f"skip missing: {p}")

    all_ep = np.concatenate([d["epoch"] for _, d, _, _ in phases])
    all_succ = np.concatenate([d["succ_rate"] for _, d, _, _ in phases]) * 100
    all_reward = np.concatenate([d["mean_reward"] for _, d, _, _ in phases])
    all_kl = np.concatenate([d["mean_kl"] for _, d, _, _ in phases])
    all_ent = np.concatenate([d["mean_entropy"] for _, d, _, _ in phases])
    all_loss = np.concatenate([d["mean_loss"] for _, d, _, _ in phases])
    all_pol = np.concatenate([d["mean_policy_loss"] for _, d, _, _ in phases])

    N = len(all_ep)

    # Rolling mean / std (window 10).
    W = 10
    roll_mean = np.array([all_succ[max(0, i-W+1):i+1].mean() for i in range(N)])
    roll_std = np.array([all_succ[max(0, i-W+1):i+1].std() for i in range(N)])

    # Linear regression on last 50 epochs.
    last50 = all_succ[-50:]
    x = np.arange(50)
    slope50, intercept50 = np.polyfit(x, last50, 1)
    # Slope arrow over last 50 ep on the chart.
    fit_y = slope50 * x + intercept50

    peak_idx = int(np.argmax(all_succ))
    peak_ep = int(all_ep[peak_idx])
    peak_acc = all_succ[peak_idx]

    def boundaries(ax):
        for k in range(1, len(phases)):
            x = phases[k][1]["epoch"][0] - 0.5
            ax.axvline(x, color="gray", linestyle=":", linewidth=1.0, alpha=0.45)

    def plot_phases(ax, key, scale=1):
        ax.plot(all_ep, np.concatenate([d[key] for _, d, _, _ in phases]) * scale,
                "-", color="dimgray", linewidth=1.4, alpha=0.4, zorder=1)
        for label, d, mk, col in phases:
            ax.plot(d["epoch"], d[key] * scale, mk, color=col, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=label, zorder=3)

    fig, axes = plt.subplots(3, 2, figsize=(18, 13))

    # (0,0) succ_rate with rolling mean band — convergence diagnostic
    ax = axes[0, 0]
    plot_phases(ax, "succ_rate", scale=100)
    boundaries(ax)
    # Rolling mean band.
    ax.fill_between(all_ep, roll_mean - roll_std, roll_mean + roll_std,
                    color="black", alpha=0.10, zorder=2,
                    label=f"rolling ±1σ (window={W})")
    ax.plot(all_ep, roll_mean, "-", color="black", linewidth=2.0, alpha=0.6,
            zorder=3, label="rolling mean")
    # Linear fit on last 50.
    ax.plot(all_ep[-50:], fit_y, "--", color="darkred", linewidth=2.2,
            zorder=4,
            label=f"last 50ep slope={slope50:+.4f} pp/ep (≈ {slope50*50:+.2f}pp)")
    # Peak star.
    ax.scatter([peak_ep], [peak_acc], s=300, marker="*", color="gold",
               edgecolors="black", linewidths=1.3, zorder=10,
               label=f"online peak ep{peak_ep}: {peak_acc:.2f}%")
    # v4 + V19-best refs.
    ax.axhline(V4_ACC, color="tab:red", linestyle="--", linewidth=1.3,
               alpha=0.7, label=f"mp_vne_v4 eval={V4_ACC}%")
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle="--", linewidth=1.3,
               alpha=0.7, label=f"V19-best eval (ep19)={V19_BEST_EVAL}%")
    ax.set_title("Online success rate per epoch — convergence diagnostic",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # (0,1) Convergence indicator: rolling std (variance)
    ax = axes[0, 1]
    ax.plot(all_ep, roll_std, "-", color="tab:purple", linewidth=2.0)
    boundaries(ax)
    ax.set_title(f"Rolling σ of succ_rate (window={W}) — variance trend",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("σ (pp)")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.96,
            "↓ trend = bouncing shrinks\nflat = noise floor reached",
            transform=ax.transAxes, fontsize=10, va="top", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.9))

    # (1,0) reward
    ax = axes[1, 0]
    plot_phases(ax, "mean_reward")
    boundaries(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean reward per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")

    # (1,1) KL
    ax = axes[1, 1]
    plot_phases(ax, "mean_kl")
    boundaries(ax)
    ax.set_title("Mean KL(π‖π_ref) per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean KL")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")

    # (2,0) loss
    ax = axes[2, 0]
    plot_phases(ax, "mean_loss")
    boundaries(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean total loss per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (2,1) entropy
    ax = axes[2, 1]
    plot_phases(ax, "mean_entropy")
    boundaries(ax)
    ax.set_title("Mean entropy per epoch (cand sharpening)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean entropy")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(
        f"V19 normal-reward PPO on 50-node — {N} epochs total "
        f"(succ peak ep{peak_ep} = {peak_acc:.2f}%, slope on last 50ep "
        f"= {slope50:+.4f} pp/ep)\n"
        f"Reward = +1.0 + 0.3·rel_cost (success) | −1.0 (fail)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_normal_{N}ep_convergence_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ----- Focused zoomed succ_rate plot (sole panel) -----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(all_ep, all_succ, "-", color="dimgray", linewidth=1.4, alpha=0.4,
            zorder=1)
    for label, d, mk, col in phases:
        ax.plot(d["epoch"], d["succ_rate"] * 100, mk, color=col, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5,
                label=label, zorder=3)
    for k in range(1, len(phases)):
        x = phases[k][1]["epoch"][0] - 0.5
        ax.axvline(x, color="gray", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.fill_between(all_ep, roll_mean - roll_std, roll_mean + roll_std,
                    color="black", alpha=0.10, zorder=2)
    ax.plot(all_ep, roll_mean, "-", color="black", linewidth=2.0, alpha=0.7,
            zorder=3, label="rolling mean (w=10)")
    ax.plot(all_ep[-50:], fit_y, "--", color="darkred", linewidth=2.2,
            zorder=4,
            label=f"last-50ep slope = {slope50:+.4f} pp/ep")
    ax.scatter([peak_ep], [peak_acc], s=350, marker="*", color="gold",
               edgecolors="black", linewidths=1.4, zorder=10,
               label=f"online peak: ep{peak_ep} = {peak_acc:.2f}%")
    ax.axhline(V4_ACC, color="tab:red", linestyle="--", linewidth=1.3,
               alpha=0.7, label=f"mp_vne_v4 eval = {V4_ACC}%")
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle="--", linewidth=1.3,
               alpha=0.7, label=f"V19-best eval (ep19) = {V19_BEST_EVAL}%")

    ax.text(0.02, 0.96,
            f"Total epochs: {N}\n"
            f"Last-50 slope: {slope50:+.4f} pp/ep ({slope50*50:+.2f}pp / 50ep)\n"
            f"First-50 mean / std: {all_succ[:50].mean():.2f}% / {all_succ[:50].std():.2f}pp\n"
            f"Last-50  mean / std: {all_succ[-50:].mean():.2f}% / {all_succ[-50:].std():.2f}pp\n"
            f"→ mean ↑, std ↓, slope ≈ 0  ⇒  converged (stochastic equilibrium)",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkblue",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.92))

    ax.set_title(
        f"V19 normal-reward PPO on 50-node — {N} epochs (online succ_rate)",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Online success rate (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    out2 = OUT / f"50nodes_normal_{N}ep_succ_only_{DATE_TAG}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
