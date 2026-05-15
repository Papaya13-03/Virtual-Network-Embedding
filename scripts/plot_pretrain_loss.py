#!/usr/bin/env python3
"""Plot Actor-Critic pretrain diagnostics.

Layout (2x3 grid):
  (0,0) avg_reward            (0,1) critic_loss = (R - V(s))²       (0,2) entropy
  (1,0) node/link/cand loss   (1,1) value_mean vs avg_reward         (1,2) adv_std

critic_loss is the one supervised-style "real" loss in this system —
it should decrease monotonically as V(s) learns to predict returns.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x.copy()
    pad = np.full(w - 1, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    rm = (c[w:] - c[:-w]) / w
    return np.concatenate([pad, rm])


def col(rows, key, default=0.0):
    return np.array([float(r.get(key, default) or default) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title-suffix", default="")
    p.add_argument("--episodes-per-batch", type=int, default=None)
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty CSV: {args.csv}")

    avg_reward = col(rows, "avg_reward")
    node_loss = col(rows, "node_loss")
    link_loss = col(rows, "link_loss")
    cand_loss = col(rows, "cand_loss")
    critic_loss = col(rows, "critic_loss")
    value_mean = col(rows, "value_mean")
    entropy = col(rows, "entropy")
    adv_std = col(rows, "adv_std")

    n = len(rows)
    if args.episodes_per_batch:
        x = np.arange(1, n + 1) * args.episodes_per_batch
        xlabel = "Episode"
    else:
        x = np.arange(1, n + 1)
        xlabel = "Batch"

    w = max(5, n // 20)
    rm = lambda y: rolling_mean(y, w)

    suffix = args.title_suffix
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Actor-Critic pretrain diagnostics{suffix}", fontsize=14)

    def plot_panel(ax, y, color, title, ylabel, ylim=None):
        ax.plot(x, y, color=color, alpha=0.25, label="per-batch")
        ax.plot(x, rm(y), color=color, linewidth=2.0, label=f"rolling (w={w})")
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    plot_panel(axes[0, 0], avg_reward, "C0", "Average Reward", "avg_reward")
    plot_panel(axes[0, 1], critic_loss, "C1",
               "Critic Loss = MSE(V(s), R)  [the interpretable loss]",
               "critic_loss")
    plot_panel(axes[0, 2], entropy, "C5", "Policy Entropy (per-decision avg)",
               "entropy")

    # Per-head losses on the same panel for direct comparison.
    ax = axes[1, 0]
    ax.plot(x, node_loss, color="C2", alpha=0.2)
    ax.plot(x, rm(node_loss), color="C2", linewidth=2, label=f"node (rm w={w})")
    ax.plot(x, link_loss, color="C3", alpha=0.2)
    ax.plot(x, rm(link_loss), color="C3", linewidth=2, label=f"link (rm w={w})")
    ax.plot(x, cand_loss, color="C4", alpha=0.2)
    ax.plot(x, rm(cand_loss), color="C4", linewidth=2, label=f"cand (rm w={w})")
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_title("Per-head policy loss (normalized)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("policy loss")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # V(s) tracking reward
    ax = axes[1, 1]
    ax.plot(x, avg_reward, color="C0", alpha=0.25)
    ax.plot(x, rm(avg_reward), color="C0", linewidth=2, label=f"reward (rm w={w})")
    ax.plot(x, value_mean, color="C1", alpha=0.25)
    ax.plot(x, rm(value_mean), color="C1", linewidth=2, label=f"V(s) (rm w={w})")
    ax.set_title("V(s) tracks reward — gap = TD error magnitude")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value / reward")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plot_panel(axes[1, 2], adv_std, "C6", "Advantage std (signal magnitude)",
               "adv_std")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    head = min(50, n)
    tail = min(50, n)
    print(f"Stats: n_batches={n}")
    print(f"  reward       : head{head}={avg_reward[:head].mean():+.4f}  tail{tail}={avg_reward[-tail:].mean():+.4f}  delta={avg_reward[-tail:].mean() - avg_reward[:head].mean():+.4f}")
    print(f"  critic_loss  : head{head}={critic_loss[:head].mean():+.4f}  tail{tail}={critic_loss[-tail:].mean():+.4f}  (kỳ vọng giảm)")
    print(f"  value_mean   : head{head}={value_mean[:head].mean():+.4f}  tail{tail}={value_mean[-tail:].mean():+.4f}")
    print(f"  entropy      : head{head}={entropy[:head].mean():+.4f}  tail{tail}={entropy[-tail:].mean():+.4f}  (plateau = no collapse)")
    print(f"  adv_std      : head{head}={adv_std[:head].mean():+.4f}  tail{tail}={adv_std[-tail:].mean():+.4f}")
    print(f"  node_loss    : head{head}={node_loss[:head].mean():+.4f}  tail{tail}={node_loss[-tail:].mean():+.4f}")
    print(f"  link_loss    : head{head}={link_loss[:head].mean():+.4f}  tail{tail}={link_loss[-tail:].mean():+.4f}")
    print(f"  cand_loss    : head{head}={cand_loss[:head].mean():+.4f}  tail{tail}={cand_loss[-tail:].mean():+.4f}")


if __name__ == "__main__":
    main()
