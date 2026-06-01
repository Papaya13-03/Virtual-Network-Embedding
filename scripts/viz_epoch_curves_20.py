"""20-epoch combined learning curves (10 initial + 10 continuation).

Reads:
  logs/ppo_v19_50nodes_10ep_epoch_summary.csv         (epochs 1-10)
  logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv    (continuation, renumber 11-20)

Saves results/figures/50nodes_v19_20ep_epoch_curves_<ts>.png.
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


def load_csv(path, epoch_offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    keys = rows[0].keys()
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data["epoch"] = data["epoch"] + epoch_offset
    return data


def main():
    first = load_csv(ROOT / "logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0)
    cont = load_csv(ROOT / "logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10)

    epochs = np.concatenate([first["epoch"], cont["epoch"]])
    succ_rate = np.concatenate([first["succ_rate"], cont["succ_rate"]]) * 100
    mean_reward = np.concatenate([first["mean_reward"], cont["mean_reward"]])
    mean_loss = np.concatenate([first["mean_loss"], cont["mean_loss"]])
    mean_policy = np.concatenate([first["mean_policy_loss"], cont["mean_policy_loss"]])
    mean_kl = np.concatenate([first["mean_kl"], cont["mean_kl"]])
    mean_ent = np.concatenate([first["mean_entropy"], cont["mean_entropy"]])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    def boundary_line(ax):
        ax.axvline(10.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
                   label="continuation")

    # (0,0) succ_rate ↑
    ax = axes[0, 0]
    ax.plot(first["epoch"], first["succ_rate"] * 100, "o-",
            color="tab:green", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["succ_rate"] * 100, "s-",
            color="tab:olive", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    peak = int(np.argmax(succ_rate))
    ax.scatter([epochs[peak]], [succ_rate[peak]], s=200, marker="*",
               color="red", zorder=10, label=f"peak ep {int(epochs[peak])}: {succ_rate[peak]:.1f}%")
    ax.set_title("Online success rate per epoch (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.04,
            f"ep1: {succ_rate[0]:.1f}%   ep20: {succ_rate[-1]:.1f}%\n"
            f"peak: {succ_rate[peak]:.1f}% (ep {int(epochs[peak])})\n"
            f"Δ: +{succ_rate[-1]-succ_rate[0]:.1f}pp",
            transform=ax.transAxes, fontsize=10, va="bottom", fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.9))

    # (0,1) reward ↑
    ax = axes[0, 1]
    ax.plot(first["epoch"], first["mean_reward"], "o-",
            color="tab:red", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["mean_reward"], "s-",
            color="tab:pink", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean reward per epoch (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.04,
            f"ep1: {mean_reward[0]:+.3f}\n"
            f"ep20: {mean_reward[-1]:+.3f}\n"
            f"Δ: {mean_reward[-1]-mean_reward[0]:+.3f}",
            transform=ax.transAxes, fontsize=10, va="bottom", fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="#ffebee", alpha=0.9))

    # (0,2) mean_loss → 0
    ax = axes[0, 2]
    ax.plot(first["epoch"], first["mean_loss"], "o-",
            color="tab:blue", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["mean_loss"], "s-",
            color="tab:cyan", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean total loss per epoch (→ 0 = converged)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,0) policy_loss
    ax = axes[1, 0]
    ax.plot(first["epoch"], first["mean_policy_loss"], "o-",
            color="tab:orange", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["mean_policy_loss"], "s-",
            color="goldenrod", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Mean policy loss per epoch", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean policy_loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,1) KL ↑
    ax = axes[1, 1]
    ax.plot(first["epoch"], first["mean_kl"], "o-",
            color="tab:purple", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["mean_kl"], "s-",
            color="indigo", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    ax.set_title("Mean KL(π‖π_ref) per epoch",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean KL")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            f"First 10 ep: ref = R2 IL → KL grows to ~0.6.\n"
            f"Cont 10 ep: ref = ep10 ckpt → KL re-starts from ~0.1.",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.9))

    # (1,2) entropy ↓
    ax = axes[1, 2]
    ax.plot(first["epoch"], first["mean_entropy"], "o-",
            color="tab:cyan", linewidth=2.5, markersize=7, label="first 10 ep")
    ax.plot(cont["epoch"], cont["mean_entropy"], "s-",
            color="teal", linewidth=2.5, markersize=7, label="continuation")
    boundary_line(ax)
    ax.set_title("Mean entropy per epoch (↓ — cand sharpening)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean entropy")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle("PPO on 50-node — 20 epochs (10 + 10 continuation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_v19_20ep_epoch_curves_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
