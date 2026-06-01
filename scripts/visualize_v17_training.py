"""V17 training curve. V17 shares V6's checkpoint (no separate training);
plotted from logs/imitation_v6_100nodes.csv with batches converted to episodes."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path("logs/imitation_v6_100nodes.csv")
OUT = Path("docs/visualizations")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
with open(LOG) as f:
    for r in csv.DictReader(f):
        rows.append({
            "batch": int(r["batch"]),
            "avg_loss": float(r["avg_loss"]),
            "expert_succ": float(r["expert_succ"]),
            "matched_rate": float(r["matched_rate"]),
        })

batches = [r["batch"] for r in rows]
losses = [r["avg_loss"] for r in rows]
expert_succ = [r["expert_succ"] * 100 for r in rows]
matched = [r["matched_rate"] * 100 for r in rows]

# batch_size = 16 → episodes = batch × 16
episodes = [b * 16 for b in batches]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("V17 training — shares V6 checkpoint (IL on mp_vne strong_targets, 10k VNRs)",
             fontsize=14, fontweight="bold")

# Loss
ax = axes[0]
ax.plot(episodes, losses, color="#CC3344", linewidth=2)
ax.set_title("Cross-entropy loss vs expert mapping (lower = matches mp_vne better)",
             fontsize=12)
ax.set_xlabel("VNR episode (#)"); ax.set_ylabel("Avg CE loss")
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate(
    f"final ~ {losses[-1]:.2f}",
    xy=(episodes[-1], losses[-1]), xytext=(-80, 20), textcoords="offset points",
    fontsize=11, fontweight="bold", color="#CC3344",
    arrowprops=dict(arrowstyle="->", color="#CC3344", alpha=0.6),
)

# Expert success rate (data quality indicator)
ax = axes[1]
ax.plot(episodes, expert_succ, color="#226688", linewidth=2, label="mp_vne expert success %")
ax.plot(episodes, matched, color="#88aa44", linewidth=2, label="target match %")
ax.set_title("Training data signal (% VNRs where expert provides a target)",
             fontsize=12)
ax.set_xlabel("VNR episode (#)"); ax.set_ylabel("%")
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = OUT / "training_v17_curve.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
