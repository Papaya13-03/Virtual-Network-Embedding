"""Eval comparison on the 100-node test set (3000 VNRs):
MP-VNE | CARL-VNE Normal (e96) | CARL-VNE Cost-focused (e83) ★

Figure 1: running cumulative metrics over the VNR sequence (2x2).
Figure 2: final-metrics bar grid from metrics.json.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS = ROOT / "results/scenario_100nodes"
ALGOS = [
    ("MP-VNE", "mp_vne_v4", "tab:red", "-"),
    ("CARL-VNE (Normal, e96)", "carl_vne_normal_e96", "tab:green", "-"),
    ("CARL-VNE (Cost-focused, e83) ★", "carl_vne_cf_e83", "tab:purple", "-"),
]


def load_substrate_and_vnrs():
    sub = json.loads((ROOT / "datasets/scenario_100nodes/substrate.json").read_text())
    vnrs = json.loads((ROOT / "datasets/scenario_100nodes/virtual_requests.json").read_text())

    snode = {}
    for dom in sub["domains"]:
        for n in dom["nodes"]:
            snode[n["id"]] = n

    link = {}
    for dom in sub["domains"]:
        for l in dom["links"]:
            link[tuple(sorted([l["source"], l["target"]]))] = l
    for l in sub.get("inter_domain_links", []):
        link[tuple(sorted([l["source"], l["target"]]))] = l

    return snode, link, vnrs


def per_vnr_metrics(solutions, snode, link, vnrs):
    """Compute (success, cost, revenue, delay) per VNR (metrics.json formula)."""
    vnr_map = {v["id"]: v for v in vnrs}
    results = []
    for sol in solutions:
        if not sol["is_successful"]:
            results.append({"succ": False, "cost": 0.0, "rev": 0.0, "delay": 0.0})
            continue
        vnr = vnr_map[sol["vnr_id"]]
        vn = vnr["virtual_network"]
        vn_node_by_id = {n["id"]: n for n in vn["nodes"]}
        rev = sum(n["cpu_demand"] for n in vn["nodes"]) \
            + sum(l["bandwidth_demand"] for l in vn["links"])
        cost = 0.0
        delay_total = 0.0
        for vnid, snid in sol["node_mapping"].items():
            cost += vn_node_by_id[vnid]["cpu_demand"] * snode[snid]["cpu_price"]
        for _, paths in sol["link_mapping"].items():
            for path in paths:
                bw_alloc = path["allocated_bandwidth"]
                for edge in path["path"]:
                    u, v = edge.split("->")
                    key = tuple(sorted([u, v]))
                    if key in link:
                        cost += bw_alloc * link[key]["bandwidth_price"]
                        delay_total += link[key].get(
                            "transmission_delay", link[key].get("delay", 1.0))
        results.append({"succ": True, "cost": cost, "rev": rev, "delay": delay_total})
    return results


def running_metrics(results):
    N = len(results)
    acc = np.zeros(N)
    avg_cost = np.zeros(N)
    rev_cost = np.zeros(N)
    avg_delay = np.zeros(N)
    cum_succ, cum_cost, cum_rev, cum_delay = 0, 0.0, 0.0, 0.0
    for i, r in enumerate(results):
        if r["succ"]:
            cum_succ += 1
            cum_cost += r["cost"]
            cum_rev += r["rev"]
            cum_delay += r["delay"]
        acc[i] = cum_succ / (i + 1) * 100
        avg_cost[i] = cum_cost / max(cum_succ, 1)
        rev_cost[i] = cum_rev / max(cum_cost, 1e-9)
        avg_delay[i] = cum_delay / max(cum_succ, 1)
    return acc, avg_cost, rev_cost, avg_delay


def main():
    snode, link, vnrs = load_substrate_and_vnrs()

    data = {}
    finals = {}
    for label, name, color, ls in ALGOS:
        d = RESULTS / name
        if not (d / "solutions.json").exists():
            print(f"MISSING: {d}")
            continue
        sols = json.loads((d / "solutions.json").read_text())
        data[label] = (color, ls, *running_metrics(
            per_vnr_metrics(sols, snode, link, vnrs)))
        finals[label] = (color, json.loads((d / "metrics.json").read_text()))

    if not data:
        return

    # === Figure 1: running metrics 2x2 ===
    N = len(next(iter(data.values()))[2])
    x = np.arange(1, N + 1)
    panels = [
        (2, "Running acceptance rate  (↑ better)", "Cumulative acceptance (%)"),
        (3, "Running avg cost per success  (↓ better)", "Σ cost / successes"),
        (4, "Running revenue / cost  (↑ better)", "Σ revenue / Σ cost"),
        (5, "Running avg delay per success  (↓ better)", "Σ delay / successes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (idx, title, ylab) in zip(axes.flat, panels):
        for label, vals in data.items():
            ax.plot(x, vals[idx], vals[1], color=vals[0], linewidth=2.0, label=label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("VNR index")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("100-node test set (3000 VNRs) — CARL-VNE (CF / Normal) vs MP-VNE",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_eval_lines_{DATE_TAG}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # === Figure 2: final metrics bar grid ===
    metrics = [
        ("acceptance_rate", "Acceptance rate (%)", 100, "↑"),
        ("revenue_rate", "Revenue", 1, "↑"),
        ("avg_cost", "Avg cost", 1, "↓"),
        ("revenue_cost_ratio", "Revenue / cost", 1, "↑"),
        ("avg_delay", "Avg delay", 1, "↓"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    labels = list(finals.keys())
    SHORT = {"MP-VNE": "MP-VNE",
             "CARL-VNE (Normal, e96)": "CARL-VNE\nNormal",
             "CARL-VNE (Cost-focused, e83) ★": "CARL-VNE\nCF ★"}
    short = [SHORT.get(l, l) for l in labels]
    for ax, (key, title, scale, arrow) in zip(axes, metrics):
        vals = [finals[l][1][key] * scale for l in labels]
        colors = [finals[l][0] for l in labels]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"{title}  ({arrow})", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.6 + 1e-9
        ax.set_ylim(max(0, lo - pad), hi + pad * 0.6)
    fig.suptitle("100-node test set — final metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_eval_bars_{DATE_TAG}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
