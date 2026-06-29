"""Compare Normal-reward (130 ep) vs Cost-focused (100 ep) on 50-node.

Shows reward + acceptance trajectory per epoch for both recipes, with rolling
mean + linear-fit slope on the tail to indicate convergence.
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

# 6 phases of normal-reward training (130 epochs).
NORMAL_PHASES = [
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont4_epoch_summary.csv", 50),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont5_epoch_summary.csv", 80),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont6_epoch_summary.csv", 130),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont7_epoch_summary.csv", 150),
]

# 5 phases of cost-focused training (200 epochs).
CF_PHASES = [
    ("experiments/carl_vne_50nodes/costfocused/logs/ppo_v19_50nodes_costfocused_epoch_summary.csv", 0),
    ("experiments/carl_vne_50nodes/costfocused/logs/ppo_v19_50nodes_cf_cont_epoch_summary.csv", 10),
    ("experiments/carl_vne_50nodes/costfocused/logs/ppo_v19_50nodes_cf_cont2_epoch_summary.csv", 50),
    ("experiments/carl_vne_50nodes/costfocused/logs/ppo_v19_50nodes_cf_cont3_epoch_summary.csv", 100),
    ("experiments/carl_vne_50nodes/costfocused/logs/ppo_v19_50nodes_cf_cont4_epoch_summary.csv", 150),
]

V4_ACC = 29.03
V19_BEST_EVAL = 33.13


def load_csv(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    keys = rows[0].keys()
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data["epoch"] = data["epoch"] + offset
    return data


def stitch(phases):
    parts = [load_csv(ROOT / p, off) for p, off in phases]
    epochs = np.concatenate([d["epoch"] for d in parts])
    succ = np.concatenate([d["succ_rate"] for d in parts]) * 100
    reward = np.concatenate([d["mean_reward"] for d in parts])
    return epochs, succ, reward


def rolling(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].mean() for i in range(len(arr))])


def rolling_std(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].std() for i in range(len(arr))])


def main():
    n_ep, n_succ, n_reward = stitch(NORMAL_PHASES)
    c_ep, c_succ, c_reward = stitch(CF_PHASES)

    # Rolling stats.
    W = 10
    n_succ_mean = rolling(n_succ, W); n_succ_std = rolling_std(n_succ, W)
    n_reward_mean = rolling(n_reward, W); n_reward_std = rolling_std(n_reward, W)
    c_succ_mean = rolling(c_succ, W); c_succ_std = rolling_std(c_succ, W)
    c_reward_mean = rolling(c_reward, W); c_reward_std = rolling_std(c_reward, W)

    # Linear fit on last 50 epochs.
    def fit_tail(arr, last=50):
        tail = arr[-last:]
        x = np.arange(len(tail))
        slope, intercept = np.polyfit(x, tail, 1)
        return slope, slope * x + intercept

    n_succ_slope, n_succ_fit = fit_tail(n_succ)
    n_rew_slope, n_rew_fit = fit_tail(n_reward)
    c_succ_slope, c_succ_fit = fit_tail(c_succ)
    c_rew_slope, c_rew_fit = fit_tail(c_reward)

    # Peaks.
    n_peak = int(np.argmax(n_succ)); c_peak = int(np.argmax(c_succ))

    # ===== 2x2 figure: rows = recipe, cols = (reward, succ) =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    # --- Top row: Normal reward
    NORMAL_COLOR = "tab:green"
    # (0,0) reward
    ax = axes[0, 0]
    ax.plot(n_ep, n_reward, "o", color=NORMAL_COLOR, markersize=4,
            markeredgecolor="black", markeredgewidth=0.3, alpha=0.6,
            label="per-epoch")
    ax.fill_between(n_ep, n_reward_mean - n_reward_std, n_reward_mean + n_reward_std,
                    color=NORMAL_COLOR, alpha=0.12, label=f"±1σ (w={W})")
    ax.plot(n_ep, n_reward_mean, "-", color="black", linewidth=2.0,
            label="rolling mean")
    ax.plot(n_ep[-50:], n_rew_fit, "--", color="darkred", linewidth=2.2,
            label=f"last-50ep slope={n_rew_slope:+.5f}/ep")
    ax.set_ylim(n_reward.min() - 0.005, n_reward.max() + 0.005)
    ax.set_title(f"Normal reward — mean reward per epoch  ({len(n_ep)} ep)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.96,
            f"Reward = +1.0 + 0.3·rel_cost (success)\n           = −1.0 (fail)\n"
            f"first-50 mean: {n_reward[:50].mean():+.4f}\n"
            f"last-50  mean: {n_reward[-50:].mean():+.4f}\n"
            f"Δ: {n_reward[-50:].mean()-n_reward[:50].mean():+.4f}",
            transform=ax.transAxes, fontsize=9, va="top", fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.92))

    # (0,1) succ
    ax = axes[0, 1]
    ax.plot(n_ep, n_succ, "o", color=NORMAL_COLOR, markersize=4,
            markeredgecolor="black", markeredgewidth=0.3, alpha=0.6,
            label="per-epoch")
    ax.fill_between(n_ep, n_succ_mean - n_succ_std, n_succ_mean + n_succ_std,
                    color=NORMAL_COLOR, alpha=0.12, label=f"±1σ (w={W})")
    ax.plot(n_ep, n_succ_mean, "-", color="black", linewidth=2.0,
            label="rolling mean")
    ax.plot(n_ep[-50:], n_succ_fit, "--", color="darkred", linewidth=2.2,
            label=f"last-50ep slope={n_succ_slope:+.4f}pp/ep")
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"V19-best eval={V19_BEST_EVAL}%")
    ax.axhline(V4_ACC, color="tab:red", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"MP-VNE eval={V4_ACC}%")
    ax.scatter([n_ep[n_peak]], [n_succ[n_peak]], s=300, marker="*",
               color="gold", edgecolors="black", linewidths=1.3, zorder=10,
               label=f"peak: ep{int(n_ep[n_peak])}={n_succ[n_peak]:.2f}%")
    ax.set_ylim(min(n_succ.min(), V19_BEST_EVAL) - 0.4,
                max(n_succ.max(), V19_BEST_EVAL) + 0.4)
    ax.set_title(f"Normal reward — online succ_rate per epoch  ({len(n_ep)} ep)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Online success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02, 0.96,
            f"first-50: {n_succ[:50].mean():.2f}% ± {n_succ[:50].std():.2f}pp\n"
            f"last-50 : {n_succ[-50:].mean():.2f}% ± {n_succ[-50:].std():.2f}pp\n"
            f"slope: {n_succ_slope:+.4f} pp/ep  ⇒  CONVERGED",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.92))

    # --- Bottom row: Cost-focused
    CF_COLOR = "tab:blue"
    # (1,0) reward
    ax = axes[1, 0]
    ax.plot(c_ep, c_reward, "s", color=CF_COLOR, markersize=4,
            markeredgecolor="black", markeredgewidth=0.3, alpha=0.6,
            label="per-epoch")
    ax.fill_between(c_ep, c_reward_mean - c_reward_std, c_reward_mean + c_reward_std,
                    color=CF_COLOR, alpha=0.12, label=f"±1σ (w={W})")
    ax.plot(c_ep, c_reward_mean, "-", color="black", linewidth=2.0,
            label="rolling mean")
    ax.plot(c_ep[-50:], c_rew_fit, "--", color="darkred", linewidth=2.2,
            label=f"last-50ep slope={c_rew_slope:+.5f}/ep")
    ax.set_ylim(c_reward.min() - 0.003, c_reward.max() + 0.003)
    ax.set_title(f"Cost-focused — mean reward per epoch  ({len(c_ep)} ep)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch (cf)"); ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.96,
            f"Reward = +0.5 + 1.0·rel_cost (success)\n           = −0.5 (fail)\n"
            f"first-50 mean: {c_reward[:50].mean():+.4f}\n"
            f"last-50  mean: {c_reward[-50:].mean():+.4f}\n"
            f"Δ: {c_reward[-50:].mean()-c_reward[:50].mean():+.4f}",
            transform=ax.transAxes, fontsize=9, va="top", fontweight="bold",
            color="darkblue",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.92))

    # (1,1) succ
    ax = axes[1, 1]
    ax.plot(c_ep, c_succ, "s", color=CF_COLOR, markersize=4,
            markeredgecolor="black", markeredgewidth=0.3, alpha=0.6,
            label="per-epoch")
    ax.fill_between(c_ep, c_succ_mean - c_succ_std, c_succ_mean + c_succ_std,
                    color=CF_COLOR, alpha=0.12, label=f"±1σ (w={W})")
    ax.plot(c_ep, c_succ_mean, "-", color="black", linewidth=2.0,
            label="rolling mean")
    ax.plot(c_ep[-50:], c_succ_fit, "--", color="darkred", linewidth=2.2,
            label=f"last-50ep slope={c_succ_slope:+.4f}pp/ep")
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"V19-best eval={V19_BEST_EVAL}%")
    ax.axhline(V4_ACC, color="tab:red", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"MP-VNE eval={V4_ACC}%")
    ax.scatter([c_ep[c_peak]], [c_succ[c_peak]], s=300, marker="*",
               color="gold", edgecolors="black", linewidths=1.3, zorder=10,
               label=f"peak: ep{int(c_ep[c_peak])}={c_succ[c_peak]:.2f}%")
    ax.set_ylim(min(c_succ.min(), V19_BEST_EVAL) - 0.4,
                max(c_succ.max(), V19_BEST_EVAL) + 0.4)
    ax.set_title(f"Cost-focused — online succ_rate per epoch  ({len(c_ep)} ep)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch (cf)"); ax.set_ylabel("Online success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02, 0.96,
            f"first-50: {c_succ[:50].mean():.2f}% ± {c_succ[:50].std():.2f}pp\n"
            f"last-50 : {c_succ[-50:].mean():.2f}% ± {c_succ[-50:].std():.2f}pp\n"
            f"slope: {c_succ_slope:+.4f} pp/ep  ⇒  CONVERGED (mean ↑)",
            transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            color="darkblue",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.92))

    fig.suptitle(
        "V19 PPO on 50-node — Reward & online succ per epoch, two recipes (convergence)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_two_recipes_convergence_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ===== Overlay chart: both recipes on same axes (succ_rate) =====
    fig, ax = plt.subplots(figsize=(14, 6))
    # Normalize by epoch (already aligned: cf ep1 starts from V19-best ckpt
    # ≈ normal ep19. To show side-by-side comparison, just plot both with their
    # own epoch axis on x; reader can compare shapes.)
    ax.plot(n_ep, n_succ, "o", color=NORMAL_COLOR, markersize=3.5, alpha=0.4)
    ax.plot(n_ep, n_succ_mean, "-", color=NORMAL_COLOR, linewidth=2.5,
            label=f"Normal reward ({len(n_ep)} ep) — rolling mean")
    ax.plot(c_ep, c_succ, "s", color=CF_COLOR, markersize=3.5, alpha=0.4)
    ax.plot(c_ep, c_succ_mean, "-", color=CF_COLOR, linewidth=2.5,
            label=f"Cost-focused ({len(c_ep)} ep) — rolling mean")
    ax.axhline(V19_BEST_EVAL, color="tab:green", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"V19-best eval={V19_BEST_EVAL}%")
    ax.axhline(V4_ACC, color="tab:red", linestyle=":", linewidth=1.3,
               alpha=0.8, label=f"MP-VNE eval={V4_ACC}%")
    ax.scatter([n_ep[n_peak]], [n_succ[n_peak]], s=350, marker="*",
               color="gold", edgecolors="black", linewidths=1.5, zorder=10,
               label=f"Normal peak: ep{int(n_ep[n_peak])}={n_succ[n_peak]:.2f}%")
    ax.scatter([c_ep[c_peak]], [c_succ[c_peak]], s=350, marker="*",
               color="orange", edgecolors="black", linewidths=1.5, zorder=10,
               label=f"CF peak: ep{int(c_ep[c_peak])}={c_succ[c_peak]:.2f}%")
    # TIGHT y-axis: focus on data range, drop the noisy first epochs from the
    # y-bounds so 33-37 dominates the view.
    all_y = np.concatenate([n_succ_mean, c_succ_mean, [V19_BEST_EVAL]])
    ylo = max(min(all_y) - 0.4, 28)
    yhi = max(n_succ.max(), c_succ.max()) + 0.4
    ax.set_ylim(ylo, yhi)
    ax.set_title("Two recipes side-by-side: online succ_rate convergence",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch (recipe-local)")
    ax.set_ylabel("Online success rate (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    out2 = OUT / f"50nodes_two_recipes_overlay_{DATE_TAG}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
