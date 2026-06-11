"""Visualize convergence of Normal-reward PPO training.

Goal: show when the model stops changing (= same output epoch after epoch).
Compare multiple metrics so we know which one to trust as a convergence signal.

Uses 150 epochs of normal-reward training (phase 1-7) plus the 32 per-epoch
evals we have on the test set.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

NORMAL_PHASES = [
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_epoch_summary.csv", 0),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont_epoch_summary.csv", 10),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont2_epoch_summary.csv", 20),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont3_epoch_summary.csv", 30),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont4_epoch_summary.csv", 50),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont5_epoch_summary.csv", 80),
    ("experiments/carl_vne_50nodes/normal/logs/ppo_v19_50nodes_10ep_cont6_epoch_summary.csv", 130),
]


def load_csv(path, offset=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["epoch"] = data["epoch"] + offset
    return data


def rolling(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].mean() for i in range(len(arr))])


def rolling_std(arr, w=10):
    return np.array([arr[max(0, i - w + 1):i + 1].std() for i in range(len(arr))])


def main():
    # Stitch all phases.
    parts = [load_csv(ROOT / p, off) for p, off in NORMAL_PHASES]
    epochs = np.concatenate([d["epoch"] for d in parts])
    succ = np.concatenate([d["succ_rate"] for d in parts]) * 100
    reward = np.concatenate([d["mean_reward"] for d in parts])
    kl = np.concatenate([d["mean_kl"] for d in parts])
    ent = np.concatenate([d["mean_entropy"] for d in parts])
    N = len(epochs)

    # Eval metrics (per-epoch test-set evals we have from ep1-32).
    eval_ep, eval_acc, eval_cost, eval_rc = [], [], [], []
    for ep in range(1, 50):
        p = ROOT / f"results/scenario_50nodes/il_mp_vne_v19_e{ep}/metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            eval_ep.append(ep)
            eval_acc.append(m["acceptance_rate"] * 100)
            eval_cost.append(m["avg_cost"])
            eval_rc.append(m["revenue_cost_ratio"])
    eval_ep = np.array(eval_ep)
    eval_acc = np.array(eval_acc)
    eval_cost = np.array(eval_cost)
    eval_rc = np.array(eval_rc)

    W = 10
    succ_mean = rolling(succ, W)
    succ_std = rolling_std(succ, W)

    # First-difference series (succ[t] - succ[t-1]) — random walk if converged.
    succ_diff = np.diff(succ)
    succ_diff_x = epochs[1:]

    # Cumulative |succ[t] - rolling_mean[t]| — total deviation from "expected"
    abs_dev = np.abs(succ - succ_mean)

    # Fig: 6 panels
    fig, axes = plt.subplots(3, 2, figsize=(18, 13))

    # --- (0,0) Online succ_rate + rolling mean ±1σ
    ax = axes[0, 0]
    ax.plot(epochs, succ, "o", color="tab:green", markersize=3.5, alpha=0.45,
            label="per-epoch online")
    ax.plot(epochs, succ_mean, "-", color="black", linewidth=2.2,
            label="rolling mean (w=10)")
    ax.fill_between(epochs, succ_mean - succ_std, succ_mean + succ_std,
                    color="tab:green", alpha=0.15, label="±1σ rolling")
    ax.axhline(33.13, color="tab:green", linestyle=":", linewidth=1.3,
               label="V19-best eval = 33.13%")
    ax.set_title("(A) Online succ_rate — rolling mean stabilizes ⇒ converged",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Online success rate (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.96,
            "ĐỌC: khi rolling mean (đen) phẳng → model output trung bình ổn định.\n"
            "Per-epoch noise (xanh) vẫn dao động ±0.5-1pp — đây là **noise floor**.",
            transform=ax.transAxes, fontsize=9, va="top", style="italic",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.92))

    # --- (0,1) Rolling std — best convergence indicator
    ax = axes[0, 1]
    ax.plot(epochs, succ_std, "-", color="tab:purple", linewidth=2.2)
    ax.axhline(succ_std[-50:].mean(), color="darkred", linestyle="--",
               linewidth=1.5,
               label=f"last-50ep avg σ = {succ_std[-50:].mean():.2f}pp")
    ax.set_title("(B) ★ Rolling σ — noise floor → CONVERGENCE INDICATOR",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("σ of succ_rate (pp)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "★ NÊN DÙNG metric này: khi σ chạm sàn và đi ngang → model converged.\n"
            "σ vẫn > 0 vì SAMPLING NOISE (Categorical sampling random) không phải learning.",
            transform=ax.transAxes, fontsize=9, va="top", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f3e5f5", alpha=0.92))

    # --- (1,0) First-difference (Δsucc[t] = succ[t] - succ[t-1])
    ax = axes[1, 0]
    ax.plot(succ_diff_x, succ_diff, "o-", color="tab:blue", markersize=3,
            linewidth=0.8, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    # Mark zero band ±2σ
    diff_std = succ_diff[-100:].std()
    ax.fill_between(succ_diff_x, -2*diff_std, 2*diff_std, color="black",
                    alpha=0.08, label=f"±2σ noise band (={2*diff_std:.2f}pp)")
    ax.set_title("(C) Δsucc[t] (first-difference) — random walk = converged",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δsucc[t] = succ[t] − succ[t−1] (pp)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "Mean(Δ) ≈ 0, không có drift → model OUTPUT không di chuyển.\n"
            "Phân bố trong band ±2σ ⇒ noise random.",
            transform=ax.transAxes, fontsize=9, va="top", style="italic",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.92))

    # --- (1,1) Mean KL (per-epoch) — direct policy-change measure
    ax = axes[1, 1]
    ax.plot(epochs, kl, "o-", color="tab:orange", markersize=3.5, alpha=0.6,
            linewidth=1.0)
    kl_mean = rolling(kl, W)
    ax.plot(epochs, kl_mean, "-", color="darkred", linewidth=2.0,
            label="rolling mean (w=10)")
    ax.set_title("(D) Mean KL(π‖π_ref) per epoch — policy drift",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean KL")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.96,
            "KL = khoảng cách policy hiện tại ↔ ref (init).\n"
            "Phẳng = policy không drift xa nữa. Cao hay thấp tùy ref.",
            transform=ax.transAxes, fontsize=9, va="top", style="italic",
            bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.92))

    # --- (2,0) Eval acceptance (test set) over epochs we have
    ax = axes[2, 0]
    ax.plot(eval_ep, eval_acc, "o-", color="tab:green", linewidth=2.2,
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            label="V19 per-epoch eval (test, 3000 VNR)")
    ax.axhline(33.13, color="darkgreen", linestyle=":", linewidth=1.3,
               label=f"V19-best (ep19) = 33.13%")
    ax.axhline(29.03, color="tab:red", linestyle="--", linewidth=1.3, alpha=0.7,
               label="mp_vne_v4 = 29.03%")
    best_idx = int(np.argmax(eval_acc))
    ax.scatter([eval_ep[best_idx]], [eval_acc[best_idx]], s=300, marker="*",
               color="gold", edgecolors="black", linewidths=1.3, zorder=10,
               label=f"peak: ep{eval_ep[best_idx]}={eval_acc[best_idx]:.2f}%")
    ax.set_title(f"(E) Eval acceptance per epoch ({len(eval_ep)} epochs evaluated)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Eval acceptance (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02, 0.04,
            "TRUE convergence = eval ổn định.\n"
            "Đã eval ep1-32 → peak ep19, sau đó RỚT (overfit).\n"
            "→ Online ổn định ≠ eval ổn định.",
            transform=ax.transAxes, fontsize=9, va="bottom", style="italic",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.92))

    # --- (2,1) Eval cost + rev/cost
    ax = axes[2, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(eval_ep, eval_cost, "o-", color="tab:blue", linewidth=2.0,
                  markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                  label="avg cost (left)")
    l2, = ax2.plot(eval_ep, eval_rc, "s-", color="tab:purple", linewidth=2.0,
                   markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                   label="rev/cost (right)")
    ax.axhline(216.49, color="tab:red", linestyle="--", linewidth=1.0,
               alpha=0.6, label="v4 cost=216.5")
    ax2.axhline(0.3136, color="tab:red", linestyle=":", linewidth=1.0,
                alpha=0.6, label="v4 rev/cost=0.314")
    ax.set_title("(F) Eval avg_cost + rev/cost per epoch", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg cost (↓ better)", color="tab:blue")
    ax2.set_ylabel("Rev/Cost (↑ better)", color="tab:purple")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], fontsize=9, loc="lower right")

    fig.suptitle(
        f"V19 Normal-reward PPO on 50-node — Convergence indicators ({N} online epochs, {len(eval_ep)} eval'd)\n"
        f"★ KHUYẾN NGHỊ DÙNG: panel (B) Rolling σ — đo NOISE FLOOR trực tiếp",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_normal_convergence_indicators_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
