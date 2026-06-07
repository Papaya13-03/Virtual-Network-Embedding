"""30-epoch combined learning curves + reward zoomed."""
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
    # (label, csv path, offset, marker, color)
    ("first 10 ep (ref=R2 IL)",
     "logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0, "o", "tab:red"),
    ("cont1 (11-20, ref=ep10)",
     "logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10, "s", "tab:pink"),
    ("cont2 (21-30, ref=ep20)",
     "logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20, "^", "tab:orange"),
    ("cont3 (31-50, ref=ep30)",
     "logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30, "D", "tab:purple"),
]


def load_csv(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["epoch"] = data["epoch"] + offset
    return data


def main():
    phases = []
    for label, p, off, mk, col in PHASES:
        if (ROOT / p).exists():
            phases.append((label, load_csv(ROOT / p, off), mk, col))
        else:
            print(f"skip missing: {p}")

    all_epochs = np.concatenate([d["epoch"] for _, d, _, _ in phases])
    all_reward = np.concatenate([d["mean_reward"] for _, d, _, _ in phases])
    all_succ = np.concatenate([d["succ_rate"] for _, d, _, _ in phases]) * 100
    all_loss = np.concatenate([d["mean_loss"] for _, d, _, _ in phases])
    all_kl = np.concatenate([d["mean_kl"] for _, d, _, _ in phases])
    all_ent = np.concatenate([d["mean_entropy"] for _, d, _, _ in phases])

    def boundaries(ax):
        for k in range(1, len(phases)):
            x = phases[k][1]["epoch"][0] - 0.5
            ax.axvline(x, color="gray", linestyle=":", linewidth=1.2, alpha=0.55)

    def plot_phases(ax, key, *, scale=1):
        # Continuous underline through all points first.
        ax.plot(all_epochs, np.concatenate([d[key] for _, d, _, _ in phases]) * scale,
                "-", color="dimgray", linewidth=1.6, alpha=0.45, zorder=1)
        for label, d, mk, col in phases:
            ax.plot(d["epoch"], d[key] * scale, mk, color=col, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.7,
                    label=label, zorder=3)

    # ----- Combined 6-panel figure -----
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # (0,0) succ_rate (zoomed)
    ax = axes[0, 0]
    plot_phases(ax, "succ_rate", scale=100)
    boundaries(ax)
    peak = int(np.argmax(all_succ))
    ax.scatter([all_epochs[peak]], [all_succ[peak]], s=250, marker="*",
               color="gold", edgecolors="black", linewidths=1.2, zorder=10,
               label=f"peak ep{int(all_epochs[peak])}: {all_succ[peak]:.1f}%")
    pad = (all_succ.max() - all_succ.min()) * 0.25
    ax.set_ylim(all_succ.min() - pad, all_succ.max() + pad)
    ax.set_title("Online success rate per epoch (zoomed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # (0,1) reward (zoomed)
    ax = axes[0, 1]
    plot_phases(ax, "mean_reward")
    boundaries(ax)
    rpeak = int(np.argmax(all_reward))
    ax.scatter([all_epochs[rpeak]], [all_reward[rpeak]], s=250, marker="*",
               color="gold", edgecolors="black", linewidths=1.2, zorder=10,
               label=f"peak ep{int(all_epochs[rpeak])}: {all_reward[rpeak]:+.3f}")
    pad = (all_reward.max() - all_reward.min()) * 0.25
    ax.set_ylim(all_reward.min() - pad, all_reward.max() + pad)
    ax.set_title("Mean reward per epoch (zoomed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # (0,2) loss
    ax = axes[0, 2]
    plot_phases(ax, "mean_loss")
    boundaries(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean total loss per epoch (→ 0)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,0) policy loss
    ax = axes[1, 0]
    plot_phases(ax, "mean_policy_loss")
    boundaries(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean policy loss per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean policy_loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,1) KL
    ax = axes[1, 1]
    plot_phases(ax, "mean_kl")
    boundaries(ax)
    ax.set_title("Mean KL(π‖π_ref) per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean KL")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (1,2) entropy
    ax = axes[1, 2]
    plot_phases(ax, "mean_entropy")
    boundaries(ax)
    ax.set_title("Mean entropy per epoch (↓ — cand sharpening)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean entropy")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"PPO on 50-node — {int(all_epochs[-1])} epochs total",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_v19_{int(all_epochs[-1])}ep_epoch_curves_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ----- Focused reward-only zoomed plot -----
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(all_epochs, all_reward, "-", color="dimgray", linewidth=2.0,
            alpha=0.5, zorder=1)
    for label, d, mk, col in phases:
        ax.plot(d["epoch"], d["mean_reward"], mk, color=col, markersize=10,
                markeredgecolor="black", markeredgewidth=0.8,
                label=label, zorder=3)
    for k in range(1, len(phases)):
        x = phases[k][1]["epoch"][0] - 0.5
        ax.axvline(x, color="gray", linestyle=":", linewidth=1.2, alpha=0.55)
    ax.scatter([all_epochs[rpeak]], [all_reward[rpeak]], s=350, marker="*",
               color="gold", edgecolors="black", linewidths=1.5, zorder=10,
               label=f"peak ep{int(all_epochs[rpeak])}: {all_reward[rpeak]:+.3f}")
    delta = all_reward[-1] - all_reward[0]
    ax.text(0.02, 0.96,
            f"ep1: {all_reward[0]:+.3f}\n"
            f"ep{int(all_epochs[-1])}: {all_reward[-1]:+.3f}\n"
            f"peak ep{int(all_epochs[rpeak])}: {all_reward[rpeak]:+.3f}\n"
            f"Δ tổng: {delta:+.3f}  ({'↑' if delta>0 else '↓'})",
            transform=ax.transAxes, fontsize=11, va="top", fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.92))
    pad = (all_reward.max() - all_reward.min()) * 0.25
    ax.set_ylim(all_reward.min() - pad, all_reward.max() + pad)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean reward per epoch", fontsize=12)
    ax.set_title(f"V19 PPO on 50-node — reward per epoch "
                 f"({int(all_epochs[-1])} epochs, zoomed)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xticks(np.arange(1, int(all_epochs[-1]) + 1))
    fig.tight_layout()
    out2 = OUT / f"50nodes_v19_{int(all_epochs[-1])}ep_reward_zoom_{DATE_TAG}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
