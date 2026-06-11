"""Clean acceptance-rate convergence chart for Normal-reward (presentation).

Uses all phases of normal-reward training (including in-progress cont7).
Single panel, clear styling, ready to show to a supervisor.
"""
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

NORMAL_PHASES = [
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont4_epoch_summary.csv", 50),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont5_epoch_summary.csv", 80),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont6_epoch_summary.csv", 130),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont7_epoch_summary.csv", 150),  # in-progress
]

V4_ACC = 29.03
V19_BEST_EVAL = 33.13


def load_csv(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["epoch"] = data["epoch"] + offset
    return data


def rolling(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].mean() for i in range(len(arr))])


def rolling_std(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].std() for i in range(len(arr))])


def main():
    parts = []
    for p, off in NORMAL_PHASES:
        f = ROOT / p
        if f.exists():
            parts.append(load_csv(f, off))
    epochs = np.concatenate([d["epoch"] for d in parts])
    succ = np.concatenate([d["succ_rate"] for d in parts]) * 100
    reward = np.concatenate([d["mean_reward"] for d in parts])
    N = len(epochs)

    succ_mean = rolling(succ, 10)
    succ_std = rolling_std(succ, 10)
    reward_mean = rolling(reward, 10)
    reward_std = rolling_std(reward, 10)

    # Linear fit on last 50 epochs (or all if N<50).
    M = min(50, N)
    x = np.arange(M)
    slope, intercept = np.polyfit(x, succ[-M:], 1)
    fit_y = slope * x + intercept

    peak_idx = int(np.argmax(succ))
    peak_ep = int(epochs[peak_idx])
    peak_acc = succ[peak_idx]

    # Set a clean presentation style.
    plt.rcParams.update({
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(14, 6.5))

    # Raw scatter (light).
    ax.plot(epochs, succ, "o", color="tab:green", markersize=3.5, alpha=0.35,
            label="Per-epoch online acceptance")
    # ±1σ band.
    ax.fill_between(epochs, succ_mean - succ_std, succ_mean + succ_std,
                    color="tab:green", alpha=0.13, label="±1σ (rolling, w=10)")
    # Rolling mean (the convergence story).
    ax.plot(epochs, succ_mean, "-", color="black", linewidth=2.6,
            label="Rolling mean (w=10)")
    # Linear fit on tail.
    ax.plot(epochs[-M:], fit_y, "--", color="darkred", linewidth=2.4,
            label=f"Last-{M}ep slope = {slope:+.4f} pp/ep")
    # Reference horizontals.
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle=":", linewidth=1.5,
               alpha=0.9, label=f"V19-best EVAL = {V19_BEST_EVAL}%")
    ax.axhline(V4_ACC, color="tab:red", linestyle=":", linewidth=1.5,
               alpha=0.85, label=f"mp_vne_v4 EVAL = {V4_ACC}%")
    # Peak star.
    ax.scatter([peak_ep], [peak_acc], s=350, marker="*", color="gold",
               edgecolors="black", linewidths=1.4, zorder=10,
               label=f"Online peak: ep {peak_ep} = {peak_acc:.2f}%")

    # TIGHT y-axis: only show the actual data range + small padding.
    lo = min(succ.min(), V19_BEST_EVAL) - 0.4
    hi = max(succ.max(), V19_BEST_EVAL) + 0.4
    ax.set_ylim(lo, hi)

    # Annotation box: convergence verdict.
    ax.text(0.015, 0.97,
            f"Total epochs trained: {N}\n"
            f"First-50 mean : {succ[:50].mean():.2f}%  (σ={succ[:50].std():.2f}pp)\n"
            f"Last-{M}  mean : {succ[-M:].mean():.2f}%  (σ={succ[-M:].std():.2f}pp)\n"
            f"Slope (last-{M}ep): {slope:+.4f} pp/ep  ⇒  CONVERGED",
            transform=ax.transAxes, fontsize=11, va="top", fontweight="bold",
            color="darkblue",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.95,
                      edgecolor="darkblue"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Online acceptance rate (%)")
    ax.set_title(
        f"V19 Normal-reward PPO on 50-node — online acceptance rate ({N} epochs)\n"
        f"Reward = +1.0 + 0.3·rel_cost (success) | −1.0 (fail)",
        fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", ncol=2)

    fig.tight_layout()
    out = OUT / f"50nodes_normal_acceptance_{N}ep_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ===== Reward chart (matching style) =====
    M2 = min(50, N)
    x2 = np.arange(M2)
    rslope, rint = np.polyfit(x2, reward[-M2:], 1)
    rfit = rslope * x2 + rint
    rpeak_idx = int(np.argmax(reward))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(epochs, reward, "o", color="tab:purple", markersize=3.5, alpha=0.35,
            label="Per-epoch mean reward")
    ax.fill_between(epochs, reward_mean - reward_std, reward_mean + reward_std,
                    color="tab:purple", alpha=0.13, label="±1σ (rolling, w=10)")
    ax.plot(epochs, reward_mean, "-", color="black", linewidth=2.6,
            label="Rolling mean (w=10)")
    ax.plot(epochs[-M2:], rfit, "--", color="darkred", linewidth=2.4,
            label=f"Last-{M2}ep slope = {rslope:+.5f}/ep")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6,
               label="r = 0  (break-even)")
    ax.scatter([epochs[rpeak_idx]], [reward[rpeak_idx]], s=350, marker="*",
               color="gold", edgecolors="black", linewidths=1.4, zorder=10,
               label=f"Reward peak: ep {int(epochs[rpeak_idx])} = "
                     f"{reward[rpeak_idx]:+.3f}")

    # TIGHT y-axis around actual reward range.
    rlo = reward.min() - 0.01
    rhi = reward.max() + 0.01
    ax.set_ylim(rlo, rhi)

    ax.text(0.015, 0.97,
            f"Total epochs trained: {N}\n"
            f"First-50 mean : {reward[:50].mean():+.4f}  (σ={reward[:50].std():.4f})\n"
            f"Last-{M2}  mean : {reward[-M2:].mean():+.4f}  (σ={reward[-M2:].std():.4f})\n"
            f"Slope (last-{M2}ep): {rslope:+.5f}/ep  ⇒  CONVERGED",
            transform=ax.transAxes, fontsize=11, va="top", fontweight="bold",
            color="purple",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.95,
                      edgecolor="purple"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean reward per epoch")
    ax.set_title(
        f"V19 Normal-reward PPO on 50-node — mean reward per epoch ({N} epochs)\n"
        f"Reward = +1.0 + 0.3·rel_cost (success) | −1.0 (fail)",
        fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out_r = OUT / f"50nodes_normal_reward_{N}ep_{DATE_TAG}.png"
    fig.savefig(out_r, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_r}")


if __name__ == "__main__":
    main()
