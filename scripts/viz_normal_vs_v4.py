"""Normal-reward (V19 PPO) per-epoch eval vs mp_vne_v4 baseline on 50-node test.

Reward = +1.0 + 0.3·rel_cost (success) | -1.0 (fail).
Reads results/scenario_50nodes/il_mp_vne_v19_e{ep}/metrics.json for all available
epochs, and mp_vne_v4/metrics.json as the baseline reference.

Saves results/figures/50nodes_normal_vs_v4_<ts>.png (4-panel grid).
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    v4 = json.loads((ROOT / "results/scenario_50nodes/mp_vne_v4/metrics.json").read_text())
    v4_acc = v4["acceptance_rate"] * 100
    v4_cost = v4["avg_cost"]
    v4_rev_cost = v4["revenue_cost_ratio"]
    v4_delay = v4["avg_delay"]

    rows = []
    ep = 1
    while True:
        p = ROOT / f"results/scenario_50nodes/il_mp_vne_v19_e{ep}/metrics.json"
        if not p.exists():
            break
        m = json.loads(p.read_text())
        rows.append({
            "epoch": ep,
            "acc": m["acceptance_rate"] * 100,
            "cost": m["avg_cost"],
            "rev_cost": m["revenue_cost_ratio"],
            "delay": m["avg_delay"],
        })
        ep += 1

    epochs = np.array([r["epoch"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    cost = np.array([r["cost"] for r in rows])
    rev_cost = np.array([r["rev_cost"] for r in rows])
    delay = np.array([r["delay"] for r in rows])

    best_acc_idx = int(np.argmax(acc))
    best_rc_idx = int(np.argmax(rev_cost))
    best_cost_idx = int(np.argmin(cost))
    best_delay_idx = int(np.argmin(delay))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def panel(ax, vals, ylabel, title, v4_val, best_idx, *, lower_better=False,
              fmt=".2f", color="tab:green"):
        ax.plot(epochs, vals, "o-", color=color, linewidth=2.2, markersize=7,
                label=f"V19 normal-reward (per-epoch eval)")
        ax.axhline(v4_val, color="tab:red", linestyle="--", linewidth=1.8,
                   alpha=0.85, label=f"mp_vne_v4 = {v4_val:{fmt}}")
        # Mark best.
        marker_color = "gold"
        ax.scatter([epochs[best_idx]], [vals[best_idx]], s=320, marker="*",
                   color=marker_color, edgecolors="black", linewidths=1.5,
                   zorder=10,
                   label=f"best: ep {epochs[best_idx]} = {vals[best_idx]:{fmt}}")
        # Delta annotation.
        delta = vals[best_idx] - v4_val
        sign = "↑" if delta > 0 else "↓"
        good = (delta < 0) if lower_better else (delta > 0)
        good_str = "↑ better than v4" if good and not lower_better else \
                   "↓ better than v4" if good and lower_better else \
                   "worse than v4"
        ax.text(0.02, 0.96,
                f"ep1: {vals[0]:{fmt}}\n"
                f"best ep{epochs[best_idx]}: {vals[best_idx]:{fmt}}\n"
                f"Δ vs v4: {delta:+{fmt}}  ({good_str})",
                transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
                color="darkgreen" if good else "darkred",
                bbox=dict(boxstyle="round",
                          facecolor="#e8f5e9" if good else "#ffebee",
                          alpha=0.92))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch (normal-reward PPO)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    panel(axes[0, 0], acc, "Acceptance rate (%)",
          "Acceptance rate per epoch  (↑ better)", v4_acc, best_acc_idx,
          color="tab:green", fmt=".2f")
    panel(axes[0, 1], cost, "Avg cost",
          "Avg cost per epoch  (↓ better)", v4_cost, best_cost_idx,
          lower_better=True, color="tab:blue", fmt=".2f")
    panel(axes[1, 0], rev_cost, "Revenue / Cost",
          "Revenue / Cost per epoch  (↑ better)", v4_rev_cost, best_rc_idx,
          color="tab:purple", fmt=".4f")
    panel(axes[1, 1], delay, "Avg delay",
          "Avg delay per epoch  (↓ better)", v4_delay, best_delay_idx,
          lower_better=True, color="tab:orange", fmt=".2f")

    fig.suptitle(
        f"V19 normal-reward PPO on 50-node — per-epoch eval vs mp_vne_v4  ({len(rows)} epochs)\n"
        f"Reward = +1.0 + 0.3·rel_cost (success) | −1.0 (fail)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_normal_vs_v4_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary table.
    print()
    print(f"{'Metric':14} {'best ep':>8} {'V19 best':>10} {'v4':>10}  {'Δ vs v4':>10}")
    print("-" * 60)
    print(f"{'acceptance %':14} {epochs[best_acc_idx]:>8d} {acc[best_acc_idx]:>10.2f} {v4_acc:>10.2f}  {acc[best_acc_idx]-v4_acc:>+10.2f}pp")
    print(f"{'avg cost':14} {epochs[best_cost_idx]:>8d} {cost[best_cost_idx]:>10.2f} {v4_cost:>10.2f}  {cost[best_cost_idx]-v4_cost:>+10.2f}")
    print(f"{'rev_cost':14} {epochs[best_rc_idx]:>8d} {rev_cost[best_rc_idx]:>10.4f} {v4_rev_cost:>10.4f}  {rev_cost[best_rc_idx]-v4_rev_cost:>+10.4f}")
    print(f"{'avg delay':14} {epochs[best_delay_idx]:>8d} {delay[best_delay_idx]:>10.2f} {v4_delay:>10.2f}  {delay[best_delay_idx]-v4_delay:>+10.2f}")


if __name__ == "__main__":
    main()
