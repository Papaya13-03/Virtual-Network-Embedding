"""Comprehensive 50-node visualisations.

Saves 4 timestamped figures to results/figures/:
  1. 50nodes_metrics_grid_<ts>.png   — 4-metric bar grid
  2. 50nodes_cumulative_<ts>.png     — running rate + gap + bar
  3. 50nodes_lines_<ts>.png          — 4-metric running lines
  4. 50nodes_loss_reward_<ts>.png    — IL loss + PPO loss + reward EMA
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
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

REQS_PATH = ROOT / "datasets" / "scenario_50nodes" / "virtual_requests.json"
SUBSTRATE_PATH = ROOT / "datasets" / "scenario_50nodes" / "substrate.json"


def load_substrate_prices():
    """Return dict snode_id → cpu_price (matches run_eval.py recompute)."""
    s = json.loads(SUBSTRATE_PATH.read_text())
    node_prices = {}
    for dom in s.get("domains", []):
        for n in dom.get("nodes", []):
            node_prices[n["id"]] = n.get("cpu_price", 1.0)
    return node_prices


def inst_cost_from_mapping(vn, sol, node_prices):
    """Match run_eval.py:73-92 — Σ cpu_demand×cpu_price + Σ bw×hops."""
    cost = 0.0
    vnodes_by_id = {n["id"]: n for n in (vn["nodes"] if isinstance(vn["nodes"], list) else vn["nodes"].values())}
    for vnode_id, snode_id in sol.get("node_mapping", {}).items():
        v = vnodes_by_id[vnode_id]
        cost += v["cpu_demand"] * node_prices.get(snode_id, 1.0)
    # solutions.json format: link_mapping is dict of list-of-dicts with
    # "path" (list of "src->dst" strings) and "allocated_bandwidth".
    for _, paths in sol.get("link_mapping", {}).items():
        for path_info in paths:
            path_links = path_info["path"]
            bw = path_info["allocated_bandwidth"]
            cost += bw * len(path_links)
    return cost

CASES = [
    # label, eval_log, metrics, solutions, color, ls
    ("mp_vne_v4 (heuristic)",
     "logs/eval_mp_vne_v4_50nodes.log",
     "results/scenario_50nodes/mp_vne_v4/metrics.json",
     "results/scenario_50nodes/mp_vne_v4/solutions.json",
     "tab:red", "-."),
    ("V19 single-pass (5000 ep)",
     "logs/eval_v19_pso_50nodes.log",
     "results/scenario_50nodes/il_mp_vne_v19_pso/metrics.json",
     "results/scenario_50nodes/il_mp_vne_v19_pso/solutions.json",
     "tab:gray", "--"),
    ("V19 best (multi-epoch ep 19)",
     "logs/eval_v19_50nodes_e19.log",
     "results/scenario_50nodes/il_mp_vne_v19_e19/metrics.json",
     "results/scenario_50nodes/il_mp_vne_v19_e19/solutions.json",
     "tab:blue", "-"),
]

EVAL_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+succ=(\d+)\s+\(([\d.]+)s\)")


def parse_eval_log(path):
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
    return json.loads(p.read_text()) if p.exists() else None


def load_csv(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return None
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


def smooth(y, win=7):
    if len(y) < win:
        return y, np.arange(len(y))
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="valid"), np.arange(win - 1, len(y))


def vnr_revenue(vn):
    nodes = vn["nodes"]
    links = vn["links"]
    cpu = sum(n["cpu_demand"] for n in (nodes if isinstance(nodes, list) else nodes.values()))
    bw = sum(l["bandwidth_demand"] for l in (links if isinstance(links, list) else links.values()))
    return cpu + bw


def running_metrics(sol_path, vnrs_by_id, vnrs_meta, node_prices):
    """Cumulative metrics matching run_eval.py exactly:
       cost = recomputed from node_mapping + link_mapping (not solutions.json
       embedding_cost which uses solver-specific formulas).
       rev and cost are WEIGHTED BY VN LIFETIME (× duration), matching the
       official revenue_cost_ratio definition.
    """
    sols = json.loads(Path(sol_path).read_text())
    sol_by_id = {s["vnr_id"]: s for s in sols}
    items = sorted(((v, sol_by_id.get(v["id"])) for v in vnrs_meta),
                   key=lambda x: x[0]["arrival_time"])
    n = len(items)
    succ = np.zeros(n, dtype=int)
    cost = np.zeros(n)            # duration-weighted (for rev/cost ratio)
    rev = np.zeros(n)             # duration-weighted (for rev/cost ratio)
    inst_cost_arr = np.zeros(n)   # unweighted per-VN (for avg_cost)
    for i, (v_meta, sol) in enumerate(items):
        if sol is None or not sol["is_successful"]:
            continue
        vn = v_meta["virtual_network"]
        duration = v_meta.get("lifetime", 1.0)
        inst_cost = inst_cost_from_mapping(vn, sol, node_prices)
        if not np.isfinite(inst_cost):
            continue
        succ[i] = 1
        cost[i] = inst_cost * duration                         # weighted
        rev[i] = vnr_revenue(vn) * duration                    # weighted
        inst_cost_arr[i] = inst_cost                           # unweighted
    cum_succ = np.cumsum(succ)
    cum_cost = np.cumsum(cost)              # duration-weighted
    cum_rev = np.cumsum(rev)                # duration-weighted
    cum_inst_cost = np.cumsum(inst_cost_arr)  # unweighted
    idx = np.arange(1, n + 1)
    safe_succ = np.where(cum_succ == 0, 1, cum_succ)
    return {
        "n": idx,
        "acc_rate": cum_succ / idx * 100,
        "cum_succ": cum_succ,
        "avg_cost": np.where(cum_succ == 0, 0, cum_inst_cost / safe_succ),
        "rev_cost": np.where(cum_cost == 0, 0, cum_rev / np.where(cum_cost == 0, 1, cum_cost)),
    }


# ---------------------------------------------------------------------------
# Fig 1 — metrics grid
# ---------------------------------------------------------------------------

METRICS_GRID = [
    ("acceptance_rate", "Acceptance rate (%)", "{:.1f}%", True),
    ("avg_cost",        "Average cost",        "{:.1f}",  False),
    ("avg_delay",       "Average delay",       "{:.2f}",  False),
    ("revenue_cost_ratio", "Revenue / Cost",   "{:.3f}",  True),
]


def fig_metrics_grid():
    rows = []
    for label, _, met_p, _, color, _ in CASES:
        m = load_metrics(ROOT / met_p)
        if m is not None:
            rows.append({"label": label, "color": color, "m": m})

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, fmt, higher) in zip(axes.flat, METRICS_GRID):
        labels = [r["label"] for r in rows]
        colors = [r["color"] for r in rows]
        vals = ([r["m"][key] * 100 for r in rows] if key == "acceptance_rate"
                else [r["m"][key] for r in rows])
        ax.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title + ("  (↑ better)" if higher else "  (↓ better)"),
                     fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        best = int(np.argmax(vals) if higher else np.argmin(vals))
        for i, v in enumerate(vals):
            marker = "  ★" if i == best else ""
            ax.text(i, v + (max(vals) - min(vals) + 1e-9) * 0.02,
                    fmt.format(v) + marker, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        pad = (max(vals) - min(vals) + 1e-9) * 0.18
        ax.set_ylim(min(vals) - pad * 0.5, max(vals) + pad)

    fig.suptitle("50-node metrics: V19 (IL+PPO) vs mp_vne_v4",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_metrics_grid_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2 — cumulative
# ---------------------------------------------------------------------------

def fig_cumulative():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                             gridspec_kw={"width_ratios": [2, 2, 1.2]})

    rows = []
    for label, log_p, met_p, _, color, ls in CASES:
        n, s = parse_eval_log(ROOT / log_p)
        m = load_metrics(ROOT / met_p)
        rows.append({"label": label, "n": n, "succ": s, "m": m, "color": color, "ls": ls})

    ax = axes[0]
    for r in rows:
        if len(r["n"]) == 0:
            continue
        rate = r["succ"] / r["n"] * 100
        ax.plot(r["n"], rate, color=r["color"], linestyle=r["ls"],
                label=r["label"], linewidth=2.2)
        ax.annotate(f"{rate[-1]:.1f}%", xy=(r["n"][-1], rate[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=r["color"], va="center")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Running acceptance rate (%)")
    ax.set_title("Running acceptance rate over 50-node eval")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    baseline = next((r for r in rows if "v4" in r["label"]), rows[0])
    if len(baseline["n"]) > 0:
        for r in rows:
            if r is baseline or len(r["n"]) == 0:
                continue
            b_interp = np.interp(r["n"], baseline["n"], baseline["succ"])
            delta = r["succ"] - b_interp
            ax.plot(r["n"], delta, color=r["color"], linestyle=r["ls"],
                    label=f"{r['label']} − {baseline['label']}", linewidth=2.2)
        ax.axhline(0, color="tab:gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Δ accepted VNRs vs baseline")
    ax.set_title("Gap vs heuristic baseline\n(positive = better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    bars = [(r["label"], r["m"], r["color"]) for r in rows if r["m"]]
    if bars:
        labels = [b[0] for b in bars]
        vals = [b[1]["acceptance_rate"] * 100 for b in bars]
        colors = [b[2] for b in bars]
        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Acceptance rate (%)")
        ax.set_title("Final (3000 VNRs)")
        ax.grid(alpha=0.3, axis="x")
        xmax = max(vals) * 1.15
        for i, v in enumerate(vals):
            ax.text(v + xmax * 0.01, i, f"{v:.1f}%", va="center",
                    fontsize=10, fontweight="bold")
        ax.set_xlim(0, xmax)

    fig.suptitle("50-node: V19 vs mp_vne_v4", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_cumulative_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3 — 4-metric running lines
# ---------------------------------------------------------------------------

def fig_lines():
    vnrs_meta = json.loads(REQS_PATH.read_text())
    vnrs_by_id = {v["id"]: v["virtual_network"] for v in vnrs_meta}
    node_prices = load_substrate_prices()
    rows = []
    for label, _, met_p, sol_p, color, ls in CASES:
        rm = running_metrics(ROOT / sol_p, vnrs_by_id, vnrs_meta, node_prices)
        m = load_metrics(ROOT / met_p)
        rows.append({"label": label, "rm": rm, "m": m, "color": color, "ls": ls})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for r in rows:
        ax.plot(r["rm"]["n"], r["rm"]["acc_rate"], color=r["color"],
                linestyle=r["ls"], linewidth=2,
                label=f"{r['label']}  →  {r['m']['acceptance_rate']*100:.1f}%")
    ax.set_xlabel("VNRs processed"); ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Running acceptance rate (↑ better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    for r in rows:
        mask = r["rm"]["cum_succ"] > 0
        ax.plot(r["rm"]["n"][mask], r["rm"]["avg_cost"][mask],
                color=r["color"], linestyle=r["ls"], linewidth=2,
                label=f"{r['label']}  →  {r['m']['avg_cost']:.1f}")
    ax.set_xlabel("VNRs processed"); ax.set_ylabel("Avg embedding cost")
    ax.set_title("Running avg cost (↓ better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 0]
    for r in rows:
        mask = r["rm"]["cum_succ"] > 0
        ax.plot(r["rm"]["n"][mask], r["rm"]["rev_cost"][mask],
                color=r["color"], linestyle=r["ls"], linewidth=2,
                label=f"{r['label']}  →  {r['m']['revenue_cost_ratio']:.3f}")
    ax.set_xlabel("VNRs processed"); ax.set_ylabel("Revenue / Cost")
    ax.set_title("Running rev/cost ratio (↑ better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 1]
    labels = [r["label"] for r in rows]
    delays = [r["m"]["avg_delay"] for r in rows]
    colors = [r["color"] for r in rows]
    ax.bar(range(len(labels)), delays, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Average delay (final) — running needs substrate paths")
    ax.set_ylabel("Avg delay"); ax.grid(alpha=0.3, axis="y")
    best = int(np.argmin(delays))
    for i, v in enumerate(delays):
        ax.text(i, v + (max(delays) - min(delays) + 1e-9) * 0.02,
                f"{v:.2f}" + ("  ★" if i == best else ""), ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    pad = (max(delays) - min(delays) + 1e-9) * 0.18
    ax.set_ylim(min(delays) - pad * 0.5, max(delays) + pad)

    fig.suptitle("50-node running metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_lines_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4 — Loss + Reward
# ---------------------------------------------------------------------------

def fig_loss_reward():
    il = load_csv(ROOT / "logs" / "imitation_50nodes.csv")
    ppo = load_csv(ROOT / "logs" / "ppo_v19_50nodes.csv")
    if il is None or ppo is None:
        print("(missing IL or PPO csv — skip loss_reward)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")

    # IL loss
    ax = axes[0, 0]
    ax.plot(il["batch"], il["avg_loss"], color="tab:gray", alpha=0.4, label="raw")
    sm, xi = smooth(il["avg_loss"], 7)
    ax.plot(il["batch"][xi], sm, color="tab:gray", linewidth=2.4, label="smoothed")
    ax.set_title("V17 IL pretrain — loss\n(cross-entropy on expert snode)")
    ax.set_ylabel("Loss"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.text(0.02, 0.04,
            f"start={il['avg_loss'][0]:.3f}   end={il['avg_loss'][-1]:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # PPO loss
    ax = axes[0, 1]
    ax.plot(ppo["batch"], ppo["loss"], color="tab:blue", alpha=0.4, label="total (raw)")
    sm, xi = smooth(ppo["loss"], 7)
    ax.plot(ppo["batch"][xi], sm, color="tab:blue", linewidth=2.4, label="smoothed")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_title("V19 PPO — loss (policy + KL − entropy + value)")
    ax.set_ylabel("Loss"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # IL match_rate / expert_succ
    ax = axes[1, 0]
    ax.plot(il["batch"], il["matched_rate"] * 100, color="tab:green",
            linewidth=2.2, label="match rate (expert in pool)")
    ax.plot(il["batch"], il["expert_succ"] * 100, color="tab:gray",
            linestyle="--", linewidth=1.8, label="expert success (mp_vne online)")
    ax.set_xlabel("Batch"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("IL — match rate + expert success rate")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

    # PPO reward EMA + KL twin
    ax = axes[1, 1]
    ax.plot(ppo["batch"], ppo["avg_reward"], color="tab:red", alpha=0.4, label="raw")
    sm, xi = smooth(ppo["avg_reward"], 7)
    ax.plot(ppo["batch"][xi], sm, color="tab:red", linewidth=2.4, label="smoothed")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axhline(1, color="tab:green", linewidth=0.7, linestyle="--",
               alpha=0.5, label="all-success ≈+1")
    ax.axhline(-1, color="tab:gray", linewidth=0.7, linestyle="--",
               alpha=0.5, label="all-fail = −1")
    ax.set_xlabel("Batch"); ax.set_ylabel("Avg reward EMA")
    ax.set_title("V19 PPO — reward EMA")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    if "kl" in ppo:
        ax2 = ax.twinx()
        ax2.plot(ppo["batch"], ppo["kl"], color="tab:cyan", linewidth=1.6,
                 alpha=0.85, label="KL(π‖π_ref)")
        ax2.set_ylabel("KL", color="tab:cyan")
        ax2.tick_params(axis="y", labelcolor="tab:cyan")

    fig.suptitle("Training curves on 50-node: V17 IL + V19 PPO",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_loss_reward_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_metrics_grid()
    fig_cumulative()
    fig_lines()
    fig_loss_reward()
