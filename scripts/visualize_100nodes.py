"""100-node comparison: V19 before PPO (R2) vs V19 after PPO vs mp_vne_v4.

Three panels in one figure:
  1. Running acceptance rate over VNRs (succ/n) — converges to steady state,
     makes the PPO improvement visually clear.
  2. Gap to R2 baseline (V19 − R2, v4 − R2) over VNRs — direct visual of how
     PPO pulls ahead over time.
  3. Final acceptance bar chart with values labelled.

Saves to `results/figures/100nodes_v19_vs_v4_<YYYYMMDD_HHMMSS>.png`. Each run
produces a new file (date+time stamp) so successive plots are preserved.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Cases (100-node) -------------------------------------------------------

CASES = [
    # (label, eval-log path, metrics.json path, color, short-label, linestyle)
    ("R2 — V19 before PPO (IL only)",
     "logs/multiseed_v17_r2.log",
     "results/scenario_100nodes_multiseed/il_mp_vne_v17_pso_r2/seed_42/metrics.json",
     "tab:gray", "R2 (no PPO)", "--"),
    ("V19 — IL + cand-PPO (deploy via PSO)",
     "logs/eval_v17_ppo_direct_via_pso_100nodes.log",
     "results/scenario_100nodes/il_mp_vne_v17_ppo_direct_via_pso/metrics.json",
     "tab:blue", "V19 (after PPO)", "-"),
    ("mp_vne_v4 (heuristic)",
     "logs/eval_mp_vne_v4_100nodes.log",
     "results/scenario_100nodes/mp_vne_v4/metrics.json",
     "tab:red", "v4 (heuristic)", "-."),
]

EVAL_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+succ=(\d+)\s+\(([\d.]+)s\)")


def parse_eval_log(path):
    """Return (n_processed, cumulative_succ) for the first seed in the log."""
    n, succ = [], []
    p = Path(path)
    if not p.exists():
        return np.array([]), np.array([])
    with open(p) as f:
        for line in f:
            m = EVAL_LINE.match(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                n.append(cur)
                succ.append(int(m.group(3)))
                if cur == tot:
                    break
    return np.array(n), np.array(succ)


def load_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    # Collect data once per case.
    rows = []
    for label, log_p, met_p, color, short, ls in CASES:
        n, s = parse_eval_log(ROOT / log_p)
        m = load_metrics(ROOT / met_p)
        rows.append({
            "label": label, "short": short, "color": color, "ls": ls,
            "n": n, "succ": s, "metrics": m,
        })

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                             gridspec_kw={"width_ratios": [2, 2, 1.2]})

    # ----- Panel 1: running acceptance rate (succ / n) -----
    ax = axes[0]
    for r in rows:
        if len(r["n"]) == 0:
            continue
        rate = r["succ"] / r["n"] * 100
        ax.plot(r["n"], rate, color=r["color"], linestyle=r["ls"],
                label=r["label"], linewidth=2.2)
        # Endpoint annotation.
        ax.annotate(f"{rate[-1]:.1f}%",
                    xy=(r["n"][-1], rate[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=r["color"], va="center")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Running acceptance rate (%)")
    ax.set_title("Running acceptance rate over 100-node eval")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # ----- Panel 2: gap to R2 baseline -----
    ax = axes[1]
    r2 = next((r for r in rows if r["short"] == "R2 (no PPO)"), None)
    if r2 is not None and len(r2["n"]) > 0:
        # Interpolate R2's succ onto every other case's grid.
        r2_n, r2_s = r2["n"], r2["succ"]
        for r in rows:
            if r["short"] == "R2 (no PPO)":
                continue
            if len(r["n"]) == 0:
                continue
            r2_interp = np.interp(r["n"], r2_n, r2_s)
            delta = r["succ"] - r2_interp
            ax.plot(r["n"], delta, color=r["color"], linestyle=r["ls"],
                    label=f"{r['short']}  −  R2", linewidth=2.2)
        ax.axhline(0, color="tab:gray", linestyle="--", linewidth=1.2, alpha=0.6,
                   label="R2 baseline (0)")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Δ accepted VNRs (vs R2)")
    ax.set_title("Gap from R2 baseline over time\n(positive = better than R2)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # ----- Panel 3: final acceptance bar chart -----
    ax = axes[2]
    bars = [(r["short"], r["metrics"], r["color"]) for r in rows if r["metrics"]]
    if bars:
        labels = [b[0] for b in bars]
        values = [b[1]["acceptance_rate"] * 100 for b in bars]
        colors = [b[2] for b in bars]
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Acceptance rate (%)")
        ax.set_title("Final (3000 VNRs)")
        ax.grid(alpha=0.3, axis="x")
        xmax = max(values) * 1.15
        for i, v in enumerate(values):
            ax.text(v + xmax * 0.01, i, f"{v:.1f}%", va="center", fontsize=10,
                    fontweight="bold")
        ax.set_xlim(0, xmax)

    fig.suptitle("100-node: V19 before vs after PPO vs mp_vne_v4",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_v19_vs_v4_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Metrics-grid figure: acceptance, cost, delay, rev/cost (one bar per case)
# ---------------------------------------------------------------------------

METRICS_GRID = [
    # (metric_key, panel_title, format_str, higher_is_better)
    ("acceptance_rate", "Acceptance rate (%)",       "{:.1f}%",  True),
    ("avg_cost",        "Average cost",              "{:.1f}",   False),
    ("avg_delay",       "Average delay",             "{:.2f}",   False),
    ("revenue_cost_ratio", "Revenue / Cost ratio",   "{:.3f}",   True),
]


def metrics_grid():
    """2x2 bar grid: acceptance, cost, delay, rev/cost — for each case."""
    rows = []
    for label, log_p, met_p, color, short, ls in CASES:
        m = load_metrics(ROOT / met_p)
        if m is None:
            continue
        rows.append({"short": short, "color": color, "m": m})

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, fmt, higher_better) in zip(axes.flat, METRICS_GRID):
        labels = [r["short"] for r in rows]
        colors = [r["color"] for r in rows]
        if key == "acceptance_rate":
            values = [r["m"][key] * 100 for r in rows]
        else:
            values = [r["m"][key] for r in rows]

        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black",
                      linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title + ("  (↑ better)" if higher_better else "  (↓ better)"),
                     fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        # Highlight best with a star annotation.
        if higher_better:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))
        for i, v in enumerate(values):
            marker = "  ★" if i == best_idx else ""
            ax.text(i, v + (max(values) - min(values)) * 0.02,
                    fmt.format(v) + marker, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        margin = (max(values) - min(values)) * 0.18 if max(values) != min(values) else max(values) * 0.1
        ax.set_ylim(min(values) - margin * 0.5,
                    max(values) + margin)

    fig.suptitle("100-node metrics: V19 before/after PPO vs mp_vne_v4",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_metrics_grid_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
    metrics_grid()
