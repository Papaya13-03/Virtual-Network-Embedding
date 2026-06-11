"""Convergence chart for 100-node PPO: Normal vs Cost-focused recipes.

Reads the merged global-epoch summaries written by ppo_finetune.py
(continuation runs append to the same CSV with global epoch numbers):
  experiments/carl_vne_100nodes/normal/training_epoch_summary.csv
  experiments/carl_vne_100nodes/costfocused/training_epoch_summary.csv
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

NORMAL_CSV = ROOT / "experiments/carl_vne_100nodes/normal/training_epoch_summary.csv"
CF_CSV = ROOT / "experiments/carl_vne_100nodes/costfocused/training_epoch_summary.csv"

# Baselines on 100-node test set.
MP_VNE_ACC = 32.83        # MP-VNE-Legacy (pre-rename mp_vne)
MP_VNE_V4_ACC = 23.30     # MP-VNE (paper-faithful PSO; former mp_vne_v4)


def rolling(arr, w=5):
    return np.array([arr[max(0, i - w + 1):i + 1].mean() for i in range(len(arr))])


def rolling_std(arr, w=5):
    return np.array([arr[max(0, i - w + 1):i + 1].std() for i in range(len(arr))])


def load_recipe(path):
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # Skip stray repeated header rows (defensive).
            if r.get("epoch") == "epoch":
                continue
            rows.append({k: float(v) for k, v in r.items()})
    if not rows:
        return None
    return {
        "epoch": np.array([r["epoch"] for r in rows]),
        "succ": np.array([r["succ_rate"] for r in rows]) * 100,
        "reward": np.array([r["mean_reward"] for r in rows]),
    }


def main():
    normal = load_recipe(NORMAL_CSV)
    cf = load_recipe(CF_CSV)
    if normal is None or cf is None:
        print("Missing data.")
        return

    n_normal = len(normal["epoch"])
    n_cf = len(cf["epoch"])
    print(f"Normal: {n_normal} epochs loaded")
    print(f"CF    : {n_cf} epochs loaded")

    # Find peaks.
    n_peak_i = int(np.argmax(normal["succ"]))
    cf_peak_i = int(np.argmax(cf["succ"]))
    n_peak = (int(normal["epoch"][n_peak_i]), normal["succ"][n_peak_i])
    cf_peak = (int(cf["epoch"][cf_peak_i]), cf["succ"][cf_peak_i])
    print(f"Normal peak: ep {n_peak[0]} = {n_peak[1]:.2f}%")
    print(f"CF peak    : ep {cf_peak[0]} = {cf_peak[1]:.2f}%")

    plt.rcParams.update({
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    # === Figure 1: 2-panel overlay (acceptance + reward) ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: Acceptance rate.
    ax = axes[0]
    # Raw scatter.
    ax.plot(normal["epoch"], normal["succ"], "o", color="tab:green",
            markersize=4, alpha=0.35, label="Normal raw")
    ax.plot(cf["epoch"], cf["succ"], "s", color="tab:purple",
            markersize=4, alpha=0.35, label="CF raw")
    # Rolling means.
    n_mean = rolling(normal["succ"], 5)
    cf_mean = rolling(cf["succ"], 5)
    n_std = rolling_std(normal["succ"], 5)
    cf_std = rolling_std(cf["succ"], 5)
    ax.fill_between(normal["epoch"], n_mean - n_std, n_mean + n_std,
                    color="tab:green", alpha=0.12)
    ax.fill_between(cf["epoch"], cf_mean - cf_std, cf_mean + cf_std,
                    color="tab:purple", alpha=0.12)
    ax.plot(normal["epoch"], n_mean, "-", color="darkgreen", linewidth=2.4,
            label="Normal rolling mean (w=5)")
    ax.plot(cf["epoch"], cf_mean, "-", color="purple", linewidth=2.4,
            label="CF rolling mean (w=5)")
    # Peaks.
    ax.scatter([n_peak[0]], [n_peak[1]], s=300, marker="*", color="lime",
               edgecolors="black", linewidths=1.3, zorder=10,
               label=f"Normal peak: ep{n_peak[0]} = {n_peak[1]:.2f}%")
    ax.scatter([cf_peak[0]], [cf_peak[1]], s=300, marker="*", color="gold",
               edgecolors="black", linewidths=1.3, zorder=10,
               label=f"CF peak: ep{cf_peak[0]} = {cf_peak[1]:.2f}%")
    # Baselines.
    ax.axhline(MP_VNE_ACC, color="black", linestyle=":", linewidth=1.4,
               alpha=0.7, label=f"MP-VNE-Legacy = {MP_VNE_ACC}%")
    ax.axhline(MP_VNE_V4_ACC, color="tab:red", linestyle=":", linewidth=1.4,
               alpha=0.7, label=f"MP-VNE = {MP_VNE_V4_ACC}%")
    # Y-range tight.
    all_succ = np.concatenate([normal["succ"], cf["succ"]])
    lo = min(all_succ.min(), MP_VNE_V4_ACC) - 0.5
    hi = max(all_succ.max(), MP_VNE_ACC) + 0.5
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Online acceptance rate (%)")
    ax.set_title(f"100-node PPO finetune — CARL-VNE cand head — online acceptance "
                 f"(Normal {n_normal}ep, CF {n_cf}ep)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", ncol=2)

    # Panel 2: Reward.
    ax = axes[1]
    ax.plot(normal["epoch"], normal["reward"], "o", color="tab:green",
            markersize=4, alpha=0.35)
    ax.plot(cf["epoch"], cf["reward"], "s", color="tab:purple",
            markersize=4, alpha=0.35)
    nr_mean = rolling(normal["reward"], 5)
    cfr_mean = rolling(cf["reward"], 5)
    ax.plot(normal["epoch"], nr_mean, "-", color="darkgreen", linewidth=2.4,
            label=f"Normal reward (avg={normal['reward'].mean():+.3f})")
    ax.plot(cf["epoch"], cfr_mean, "-", color="purple", linewidth=2.4,
            label=f"CF reward (avg={cf['reward'].mean():+.3f})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch (1 = real ep1 from IL pretrain checkpoint)")
    ax.set_ylabel("Mean reward per epoch")
    ax.set_title("Mean reward per epoch (Normal uses +1.0/-1.0,  CF uses +0.5/-0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        f"CARL-VNE on 100-node — Normal vs Cost-focused convergence "
        f"(up to ep {int(max(normal['epoch'].max(), cf['epoch'].max()))})",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_two_recipes_overlay_{DATE_TAG}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # === Figure 2: Convergence indicators (slope of last-N) ===
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(normal["epoch"], n_mean, "-", color="darkgreen", linewidth=2.4,
            label=f"Normal rolling mean (last={n_mean[-1]:.2f}%)")
    ax.plot(cf["epoch"], cf_mean, "-", color="purple", linewidth=2.4,
            label=f"CF rolling mean (last={cf_mean[-1]:.2f}%)")

    # Last-N linear fit per recipe.
    for name, d, color in [("Normal", normal, "darkgreen"),
                            ("CF", cf, "purple")]:
        succ = d["succ"]
        eps = d["epoch"]
        M = min(20, len(succ))
        x = np.arange(M)
        slope, intercept = np.polyfit(x, succ[-M:], 1)
        fit = slope * x + intercept
        ax.plot(eps[-M:], fit, "--", color=color, linewidth=2.0, alpha=0.7,
                label=f"{name} last-{M}ep slope = {slope:+.4f} pp/ep")

    ax.axhline(MP_VNE_ACC, color="black", linestyle=":", linewidth=1.4,
               alpha=0.7, label=f"mp_vne = {MP_VNE_ACC}%")
    ax.axhline(MP_VNE_V4_ACC, color="tab:red", linestyle=":", linewidth=1.4,
               alpha=0.7, label=f"MP-VNE = {MP_VNE_V4_ACC}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acceptance rate (%) — rolling mean (w=5)")
    ax.set_title(f"100-node convergence indicator — slope on tail "
                 f"(Normal {n_normal}ep, CF {n_cf}ep)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    fig.tight_layout()
    out2 = OUT / f"100nodes_two_recipes_convergence_{DATE_TAG}.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
