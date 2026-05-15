#!/usr/bin/env python3
"""Overlay training-curve comparison: optB (PSO policy-bias) vs no-bias baseline.

Reads two Actor-Critic CSVs and renders rolling-mean curves of the diagnostics
that distinguish "candidate_head learns" from "candidate_head wastes credit":
  - avg_reward          : does policy do better with bias?
  - cand_loss           : key signal — option B's whole point is to make this train
  - critic_loss         : V(s) regression should track on both
  - entropy             : avoid collapse on both
  - value_mean vs reward: critic calibration
  - adv_std             : signal magnitude (collapsed adv_std = stuck)
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def col(rows, key, default=0.0):
    return np.array([float(r.get(key, default) or default) for r in rows])


def rolling(x, w):
    if len(x) < w:
        return x.copy()
    pad = np.full(w - 1, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    rm = (c[w:] - c[:-w]) / w
    return np.concatenate([pad, rm])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optB", required=True, help="CSV from option-B run")
    p.add_argument("--baseline", required=True, help="CSV from no-bias run")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline-label", default="baseline (no bias)")
    p.add_argument("--optB-label", default="option B (policy bias)")
    args = p.parse_args()

    optB = load(args.optB)
    base = load(args.baseline)

    # Trim baseline to same length as optB for fair comparison
    n = min(len(optB), len(base))
    optB, base = optB[:n], base[:n]
    x = np.arange(1, n + 1)
    w = max(5, n // 20)

    metrics = [
        ("avg_reward", "Average Reward", "reward"),
        ("cand_loss", "Cand-head Policy Loss (lower = stronger gradient)", "cand_loss"),
        ("critic_loss", "Critic MSE Loss (R - V(s))²", "critic_loss"),
        ("entropy", "Policy Entropy (collapse < 1.0)", "entropy"),
        ("value_mean", "V(s) (baseline calibration vs reward)", "V(s)"),
        ("adv_std", "Advantage std (signal strength)", "adv_std"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Option B (PSO policy bias) vs baseline — first {n} batches", fontsize=14)

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        y_b = col(base, key)
        y_o = col(optB, key)
        ax.plot(x, y_b, color="C0", alpha=0.15)
        ax.plot(x, rolling(y_b, w), color="C0", linewidth=2.0, label=args.baseline_label)
        ax.plot(x, y_o, color="C3", alpha=0.15)
        ax.plot(x, rolling(y_o, w), color="C3", linewidth=2.0, label=args.optB_label)
        if key == "value_mean":
            # overlay reward on value_mean to see calibration
            r_b = col(base, "avg_reward")
            r_o = col(optB, "avg_reward")
            ax.plot(x, rolling(r_b, w), color="C0", linewidth=1.0, linestyle=":",
                    label=f"{args.baseline_label} reward")
            ax.plot(x, rolling(r_o, w), color="C3", linewidth=1.0, linestyle=":",
                    label=f"{args.optB_label} reward")
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("Batch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    # Numeric summary — head vs tail for both
    head = min(20, n)
    tail = min(20, n)
    print(f"\nNumeric summary (n_batches={n}, head={head}, tail={tail}):")
    print(f"  {'metric':<14} {'baseline_head':>14} {'baseline_tail':>14} {'optB_head':>14} {'optB_tail':>14}")
    for key, title, _ in metrics:
        y_b = col(base, key)
        y_o = col(optB, key)
        print(f"  {key:<14} "
              f"{y_b[:head].mean():>+14.4f} {y_b[-tail:].mean():>+14.4f} "
              f"{y_o[:head].mean():>+14.4f} {y_o[-tail:].mean():>+14.4f}")


if __name__ == "__main__":
    main()
