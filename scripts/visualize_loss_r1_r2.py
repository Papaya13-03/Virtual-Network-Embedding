"""Plot IL training loss across R1 + R2 as one continuous timeline."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(y, w=8):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")

OUT = Path("docs/visualizations")
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "batch": int(r["batch"]),
                "loss": float(r["avg_loss"]),
                "expert_succ": float(r["expert_succ"]) * 100,
            })
    return rows


r1 = load("logs/imitation_v6_100nodes.csv")
r2 = load("logs/imitation_v6_r2_100nodes.csv")

# Batch=16 VNRs each; offset r2 by last r1 batch (continuous timeline).
last_r1 = r1[-1]["batch"]
ep_r1 = [r["batch"] * 16 for r in r1]
ep_r2 = [(r["batch"] + last_r1) * 16 for r in r2]

loss_r1 = [r["loss"] for r in r1]
loss_r2 = [r["loss"] for r in r2]
succ_r1 = [r["expert_succ"] for r in r1]
succ_r2 = [r["expert_succ"] for r in r2]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("V17 IL training — R1 (seed 42, 10k VNRs) → R2 (seed 200, +10k VNRs from R1 ckpt)",
             fontsize=14, fontweight="bold")

W = 8   # moving-average window (8 batches = 128 VNRs)

ax = axes[0]
# Raw curves (faded background)
ax.plot(ep_r1, loss_r1, color="#CC3344", linewidth=0.8, alpha=0.25)
ax.plot(ep_r2, loss_r2, color="#226688", linewidth=0.8, alpha=0.25)
# Smoothed (foreground)
sr1 = smooth(loss_r1, W)
sr2 = smooth(loss_r2, W)
ax.plot(ep_r1[-len(sr1):], sr1, color="#CC3344", linewidth=2.5, label=f"R1 (smoothed, w={W})")
ax.plot(ep_r2[-len(sr2):], sr2, color="#226688", linewidth=2.5, label=f"R2 (smoothed, w={W})")
ax.axvline(x=ep_r2[0], color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(ep_r2[0], max(loss_r1 + loss_r2) * 0.95, "  R1→R2 boundary\n  (lr 1e-3 → 5e-4)",
        fontsize=9, color="gray")
ax.set_title("Cross-entropy loss vs expert (raw faded, smoothed solid)", fontsize=12)
ax.set_xlabel("VNR episode (cumulative)"); ax.set_ylabel("Avg CE loss")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(ep_r1, succ_r1, color="#CC3344", linewidth=0.8, alpha=0.25)
ax.plot(ep_r2, succ_r2, color="#226688", linewidth=0.8, alpha=0.25)
ss_r1 = smooth(succ_r1, W)
ss_r2 = smooth(succ_r2, W)
ax.plot(ep_r1[-len(ss_r1):], ss_r1, color="#CC3344", linewidth=2.5, label="R1 (seed 42)")
ax.plot(ep_r2[-len(ss_r2):], ss_r2, color="#226688", linewidth=2.5, label="R2 (seed 200)")
ax.axvline(x=ep_r2[0], color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("mp_vne expert success rate during training (data signal)", fontsize=12)
ax.set_xlabel("VNR episode (cumulative)"); ax.set_ylabel("% expert success")
ax.set_ylim(0, max(succ_r1 + succ_r2) * 1.15)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = OUT / "training_loss_r1_r2.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
