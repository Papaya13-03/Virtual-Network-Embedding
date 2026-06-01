"""Visualize V17-arch (with various PPO/IL recipes) vs mp_vne_v4 heuristic.

Two figures:
  Fig A — Eval comparison on 200-node:
    left:  cumulative accepted VNRs over 3000-VNR eval sequence (line).
    right: final metrics bar chart (acceptance, cost, rev/cost).
  Fig B — PPO training curves: loss, reward, success-rate, KL across batches.

Saves to results/figures/.
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Timestamp every figure so successive runs don't overwrite each other.
# Format: YYYYMMDD_HHMMSS (date + hour-minute-second).
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Parse eval logs → (n_arr, cum_succ_arr)
# ---------------------------------------------------------------------------

EVAL_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+succ=(\d+)\s+\(([\d.]+)s\)")


def parse_eval_log(path):
    """Extract `(n_processed, cumulative_succ)` arrays from an eval log.

    Stops at the first `total/total` (e.g. 3000/3000) line so multiseed logs —
    which contain multiple seeds back-to-back — yield only the first seed's
    curve. If the log contains no snapshots, returns empty arrays.
    """
    n, succ = [], []
    if not Path(path).exists():
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            m = EVAL_LINE.match(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                n.append(cur)
                succ.append(int(m.group(3)))
                if cur == tot:        # end of one seed's run
                    break
    return np.array(n), np.array(succ)


def load_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Eval comparison — generic plotter (line curve + final bar)
# ---------------------------------------------------------------------------

EVAL_CASES_200 = [
    # label, eval-log path, metrics.json path, color, short-label
    ("mp_vne_v4 (heuristic)",
     "logs/eval_mp_vne_v4_200nodes.log",
     "results/scenario_200nodes/mp_vne_v4/metrics.json",
     "tab:red", "v4"),
    ("V17 IL-only (200-node)",
     "logs/eval_v17_pso_200nodes.log",
     "results/scenario_200nodes/il_mp_vne_v17_pso/metrics.json",
     "tab:gray", "V17 IL-200"),
    ("V19 (100-node IL+PPO) → 200",
     "logs/eval_v19_pso_200nodes.log",
     "results/scenario_200nodes/il_mp_vne_v19_pso/metrics.json",
     "tab:blue", "V19 transfer"),
    ("V19 fine-tuned on 200-node",
     "logs/eval_v19_200trained_200nodes.log",
     "results/scenario_200nodes/il_mp_vne_v19_pso_200trained/metrics.json",
     "tab:orange", "V19 200-FT"),
    ("R2 IL → 200-node PPO",
     "logs/eval_r2_finetuned_200nodes.log",
     "results/scenario_200nodes/il_mp_vne_r2_finetuned/metrics.json",
     "tab:green", "R2→200 PPO"),
    ("V20 (200-IL → PPO)",
     "logs/eval_v20_200nodes.log",
     "results/scenario_200nodes/il_mp_vne_v20_pso/metrics.json",
     "tab:purple", "V20"),
]

EVAL_CASES_100 = [
    # The three the user asked for: R2 (V19-pre-PPO), V19 (V19 final), v4.
    ("R2 — V19 before PPO (IL only)",
     "logs/multiseed_v17_r2.log",
     "results/scenario_100nodes_multiseed/il_mp_vne_v17_pso_r2/seed_42/metrics.json",
     "tab:gray", "R2 (IL only)"),
    ("V19 — IL + cand-PPO (deploy via PSO)",
     "logs/eval_v17_ppo_direct_via_pso_100nodes.log",
     "results/scenario_100nodes/il_mp_vne_v17_ppo_direct_via_pso/metrics.json",
     "tab:blue", "V19 (after PPO)"),
    ("mp_vne_v4 (heuristic)",
     "logs/eval_mp_vne_v4_100nodes.log",
     "results/scenario_100nodes/mp_vne_v4/metrics.json",
     "tab:red", "v4"),
]


def _plot_eval_comparison(cases, title_suffix, eval_size, out_path):
    """Generic eval-comparison plot: cumulative-accepted line + final bar."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [2, 1.4]})

    ax = axes[0]
    final_data = []
    for label, log_p, met_p, color, short in cases:
        n, s = parse_eval_log(ROOT / log_p)
        m = load_metrics(ROOT / met_p)
        if len(n) == 0 and m is None:
            continue
        if len(n) > 0:
            ax.plot(n, s, color=color, label=label, linewidth=2.0)
        if m:
            final_data.append((short, m, color))
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Cumulative accepted")
    ax.set_title(f"Cumulative acceptance over {eval_size}-node eval (3000 VNRs)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    if not final_data:
        ax.text(0.5, 0.5, "no metrics", ha="center", va="center")
    else:
        labels = [d[0] for d in final_data]
        accept = [d[1]["acceptance_rate"] * 100 for d in final_data]
        colors = [d[2] for d in final_data]
        bars = ax.barh(range(len(labels)), accept, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Acceptance rate (%)")
        ax.set_title(f"Final acceptance ({eval_size}-node)")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        for i, v in enumerate(accept):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, max(accept) * 1.18)

    fig.suptitle(title_suffix, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_eval_200nodes():
    out = OUT / f"eval_200nodes_v17_vs_v4_{DATE_TAG}.png"
    _plot_eval_comparison(EVAL_CASES_200,
                          "V17-arch variants vs mp_vne_v4 (200-node eval)",
                          eval_size=200, out_path=out)


def fig_eval_100nodes():
    out = OUT / f"eval_100nodes_v19_vs_v4_{DATE_TAG}.png"
    _plot_eval_comparison(EVAL_CASES_100,
                          "V19 (pre-PPO R2) vs V19 (post-PPO) vs mp_vne_v4 (100-node)",
                          eval_size=100, out_path=out)


# ---------------------------------------------------------------------------
# Fig B — PPO training curves
# ---------------------------------------------------------------------------

PPO_RUNS = [
    # label, csv path, color
    ("V17 PPO ordering (100-node)", "logs/ppo_v17.csv", "tab:gray"),
    ("V17 PPO direct cand (=V19 step) (100-node)", "logs/ppo_v17_direct.csv", "tab:blue"),
    ("V19 PSO-regime cand-RL (100-node)", "logs/ppo_v19_pso.csv", "tab:cyan"),
    ("V19 PPO direct (200-node FT)", "logs/ppo_v19_200nodes.csv", "tab:orange"),
    ("R2 → 200-node PPO direct", "logs/ppo_r2_200nodes.csv", "tab:green"),
    ("V20 (200-IL → PPO)", "logs/ppo_v20_200nodes.csv", "tab:purple"),
]


def load_ppo(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except ValueError:
                continue
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def fig_ppo_curves():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = [
        ("loss", "Loss"),
        ("avg_reward", "Avg reward / reward EMA"),
        ("succ_rate", "Online success rate"),
        ("kl", "KL(π‖π_ref)"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        for label, p, color in PPO_RUNS:
            d = load_ppo(ROOT / p)
            if d is None or col not in d:
                continue
            x = d["batch"] if "batch" in d else np.arange(len(d[col]))
            y = d[col]
            ax.plot(x, y, color=color, label=label, linewidth=1.5, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Batch")
        ax.grid(alpha=0.3)
    # Single legend at the bottom.
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("PPO training curves — V17-arch variants",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"ppo_training_curves_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_eval_100nodes()
    fig_eval_200nodes()
    fig_ppo_curves()
