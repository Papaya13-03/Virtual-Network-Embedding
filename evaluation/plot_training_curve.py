#!/usr/bin/env python3
"""Plot loss / reward / success rate / cost-per-revenue from rl_cand_vne train.jsonl."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/rl_cand_vne/train.jsonl")
    p.add_argument("--out", default="logs/rl_cand_vne/train_curve.png")
    args = p.parse_args()

    records = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        print("No records in log.")
        return

    eps = [r["episode"] for r in records]
    loss = [r["loss_total"] for r in records]
    reward = [r["reward_mean"] for r in records]
    baseline = [r["baseline"] for r in records]
    success = [r["success_rate"] for r in records]
    cpr = [r["cost_per_revenue_mean"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(eps, loss); axes[0, 0].set_title("loss_total"); axes[0, 0].set_xlabel("episode")
    axes[0, 1].plot(eps, reward, label="reward"); axes[0, 1].plot(eps, baseline, label="baseline", linestyle="--")
    axes[0, 1].set_title("reward / baseline"); axes[0, 1].set_xlabel("episode"); axes[0, 1].legend()
    axes[1, 0].plot(eps, success); axes[1, 0].set_title("success_rate"); axes[1, 0].set_xlabel("episode")
    axes[1, 1].plot(eps, cpr); axes[1, 1].set_title("cost_per_revenue_mean"); axes[1, 1].set_xlabel("episode")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
