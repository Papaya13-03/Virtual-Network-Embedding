"""Plot training curves: loss / reward / etc. for IL and RL runs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = Path("logs")
OUTPUT_DIR = Path("docs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if v not in ("", None) else float("nan")
                         for k, v in r.items() if k != "timestamp"})
    return rows


def main():
    # --- IL loss curves (V6, V11, V12, V13 standard IL) ---
    il_runs = [
        ("imitation_v6_100nodes.csv",  "V6 (mp_vne expert)",         "#2F7DC1"),
        ("imitation_v11_100nodes.csv", "V11 (nb-cond features)",     "#888888"),
        ("imitation_v12_100nodes.csv", "V12 (per-slink features)",   "#aaaa44"),
        ("imitation_v13_100nodes.csv", "V13 (self-distill labels)",  "#cc8844"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for fname, label, color in il_runs:
        path = LOG_DIR / fname
        if not path.exists():
            continue
        rows = load_csv(path)
        x = [r.get("batch", i) for i, r in enumerate(rows)]
        y = [r.get("avg_loss", r.get("loss", np.nan)) for r in rows]
        ax.plot(x, y, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Batch", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("IL Training Loss — supervised CE on mp_vne expert", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out1 = OUTPUT_DIR / "training_loss_IL.png"
    plt.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # --- V13 conditional IL: loss + model_better_rate ---
    p13 = LOG_DIR / "conditional_il_v13.csv"
    if p13.exists():
        rows = load_csv(p13)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
        x = [r["batch"] for r in rows]
        loss = [r["loss"] for r in rows]
        mb = [r.get("model_better_rate", np.nan) for r in rows]
        a1.plot(x, loss, color="#CC3344", linewidth=1.8)
        a1.set_title("V13 Conditional IL — CE loss (when model loses to expert)",
                     fontsize=12, fontweight="bold")
        a1.set_xlabel("Batch"); a1.set_ylabel("Loss")
        a1.set_yscale("log")
        a1.grid(alpha=0.3, linestyle="--")
        a2.plot(x, [v*100 for v in mb], color="#226688", linewidth=1.8, label="model > expert")
        a2.set_title("V13 — % of VNRs where model+PSO beat expert",
                     fontsize=12, fontweight="bold")
        a2.set_xlabel("Batch"); a2.set_ylabel("% model better")
        a2.set_ylim(0, 100)
        a2.grid(alpha=0.3, linestyle="--")
        for ax in (a1, a2):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out2 = OUTPUT_DIR / "training_v13_conditional.png"
        plt.savefig(out2, dpi=140, bbox_inches="tight")
        print(f"Saved: {out2}")
        plt.close()

    # --- V15 PPO: 4 panels ---
    p15 = LOG_DIR / "ppo_v15.csv"
    if p15.exists():
        rows = load_csv(p15)
        x = [r["episode"] for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("V15 RL training (REINFORCE + KL + entropy from V6 warm-start)",
                     fontsize=14, fontweight="bold")
        panels = [
            ("policy_loss", "Policy loss (-log_P × advantage)", "#CC3344"),
            ("kl",          "KL(π_new ‖ π_v6)",                "#226688"),
            ("entropy",     "Entropy (exploration)",            "#88aa44"),
            ("avg_reward",  "Reward EMA",                       "#E08020"),
        ]
        for ax, (key, title, color) in zip(axes.flat, panels):
            y = [r.get(key, np.nan) for r in rows]
            ax.plot(x, y, color=color, linewidth=1.8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Episode (VNR)"); ax.set_ylabel(title)
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out3 = OUTPUT_DIR / "training_v15_PPO.png"
        plt.savefig(out3, dpi=140, bbox_inches="tight")
        print(f"Saved: {out3}")
        plt.close()


if __name__ == "__main__":
    main()
