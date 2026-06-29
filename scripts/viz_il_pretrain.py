#!/usr/bin/env python3
"""Regenerate the IL-pretrain convergence figure for the thesis (50-node).

Two panels only: (A) cross-entropy loss and (B) expert success rate.
The former third panel (target-in-candidate-pool match rate) was dropped on
request. Source: experiments/pretrain/logs/imitation_50nodes.csv
Output: thesis/Hinh_ve/il_pretrain_50nodes.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "pretrain" / "logs" / "imitation_50nodes.csv"
OUT = ROOT / "thesis" / "Hinh_ve" / "il_pretrain_50nodes.png"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
})


def smooth(y, window=5):
    if len(y) < window:
        return y, np.arange(len(y))
    s = np.convolve(y, np.ones(window) / window, mode="valid")
    idx = np.arange(window - 1, len(y))
    return s, idx


def main():
    batch, loss, succ = [], [], []
    with open(SRC) as f:
        for row in csv.DictReader(f):
            batch.append(int(row["batch"]))
            loss.append(float(row["avg_loss"]))
            succ.append(float(row["expert_succ"]))
    batch = np.array(batch); loss = np.array(loss); succ = np.array(succ) * 100

    fig, (ax_loss, ax_succ) = plt.subplots(1, 2, figsize=(13, 5))

    # (A) Cross-entropy loss
    ax_loss.plot(batch, loss, color="tab:red", alpha=0.30, linewidth=1.0,
                 label="mất mát theo batch")
    ls, idx = smooth(loss, 5)
    ax_loss.plot(batch[idx], ls, color="tab:red", linewidth=2.5,
                 label="trung bình trượt (w=5)")
    ax_loss.set_xlabel("Batch (16 VNR mỗi batch)")
    ax_loss.set_ylabel("Mất mát cross-entropy")
    ax_loss.set_title("(A) Mất mát cross-entropy")
    ax_loss.legend(loc="best")
    ax_loss.grid(alpha=0.3)

    # (B) Expert success rate
    ax_succ.plot(batch, succ, color="tab:green", linewidth=2.5,
                 label="tỉ lệ thành công của lời giải mẫu")
    ax_succ.set_xlabel("Batch")
    ax_succ.set_ylabel("Tỉ lệ thành công (%)")
    ax_succ.set_title("(B) Tỉ lệ thành công của lời giải mẫu (MP-VNE)")
    ax_succ.legend(loc="best")
    ax_succ.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
