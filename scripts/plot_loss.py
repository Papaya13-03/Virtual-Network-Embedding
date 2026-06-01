#!/usr/bin/env python3
"""Plot IL pretrain loss curves from imitation_*.csv files."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    batches, losses, succs = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                batches.append(int(row["batch"]))
                losses.append(float(row["avg_loss"]))
                succs.append(float(row["expert_succ"]))
            except (KeyError, ValueError):
                continue
    return np.array(batches), np.array(losses), np.array(succs)


def smooth(y, window=10):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Pairs of label=path, e.g. v2=logs/imitation_100nodes.csv")
    p.add_argument("--output", default="logs/loss_compare.png")
    p.add_argument("--smooth-window", type=int, default=10)
    args = p.parse_args()

    runs = []
    for spec in args.inputs:
        if "=" not in spec:
            print(f"skip malformed: {spec}", file=sys.stderr)
            continue
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"missing file: {path}", file=sys.stderr)
            continue
        b, l, s = load_csv(path)
        runs.append((label, b, l, s))

    fig, (ax_loss, ax_succ) = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.tab10.colors
    for i, (label, b, l, s) in enumerate(runs):
        c = colors[i % len(colors)]
        # Raw loss
        ax_loss.plot(b, l, color=c, alpha=0.25, linewidth=0.7)
        # Smoothed
        if len(l) >= args.smooth_window:
            ls = smooth(l, args.smooth_window)
            bs = b[args.smooth_window - 1:]
            ax_loss.plot(bs, ls, color=c, linewidth=2, label=f"{label} (rolling {args.smooth_window})")
        else:
            ax_loss.plot(b, l, color=c, linewidth=2, label=label)

        # Cumulative expert succ
        ax_succ.plot(b, s * 100, color=c, linewidth=2, label=label)

    # Theoretical uniform reference
    ax_loss.axhline(y=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                    label="~log(K) uniform")

    ax_loss.set_xlabel("Batch (16 VNRs each)")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("IL pretrain loss")
    ax_loss.legend(loc="best", fontsize=9)
    ax_loss.grid(alpha=0.3)

    ax_succ.set_xlabel("Batch")
    ax_succ.set_ylabel("Expert (mp_vne) success rate %")
    ax_succ.set_title("Expert success rate (cumulative)")
    ax_succ.legend(loc="best", fontsize=9)
    ax_succ.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
