"""Line plots over VNR arrivals with mean ± std bands across multiple seeds."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.load_dataset import read_virtual_requests


RESULT_DIR = Path("results/scenario_100nodes_multiseed")
OUTPUT_DIR = Path("docs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTRATE_PATH = "datasets/scenario_100nodes/substrate.json"
REQUESTS_PATH = "datasets/scenario_100nodes/virtual_requests.json"

ALGORITHMS = [
    ("mp_vne_v4",            "V4 (mp_vne paper PSO, top-1/dom)",     "#226688"),
    ("il_mp_vne_v17_pso_r2", "V17 R2 (NN, top-1/dom, retrained)",    "#CC3344"),
]
SEEDS = [42, 43, 44]


def load_substrate_prices(substrate_path):
    with open(substrate_path) as f:
        sub = json.load(f)
    node_prices = {}
    link_delays = {}
    for d in sub.get("domains", []):
        for n in d.get("nodes", []):
            node_prices[n["id"]] = n.get("cpu_price", 1.0)
        for l in d.get("links", []) + d.get("intra_domain_links", []):
            link_delays[(l["source"], l["target"])] = l.get("transmission_delay", 1.0)
            link_delays[(l["target"], l["source"])] = l.get("transmission_delay", 1.0)
    for l in sub.get("inter_domain_links", []):
        link_delays[(l["source"], l["target"])] = l.get("transmission_delay", 5.0)
        link_delays[(l["target"], l["source"])] = l.get("transmission_delay", 5.0)
    return node_prices, link_delays


def compute_running_series(vnrs, solutions_data, node_prices, link_delays):
    sol_by_id = {s["vnr_id"]: s for s in solutions_data}
    items = sorted(vnrs, key=lambda r: r.arrival_time)

    n_arrived = n_success = 0
    cum_rev = cum_cost_weighted = cum_inst_cost = cum_inst_delay = 0.0

    accept_curve, cost_curve, revcost_curve, delay_curve = [], [], [], []

    for vnr in items:
        n_arrived += 1
        sol = sol_by_id.get(vnr.id)
        if sol is not None and sol.get("is_successful"):
            n_success += 1
            vn = vnr.virtual_network
            rev = sum(n.cpu_demand for n in vn.nodes.values())
            rev += sum(l.bandwidth_demand for l in vn.links.values())
            inst_cost = 0.0
            for vnode_id, snode_id in sol["node_mapping"].items():
                v = vn.nodes[vnode_id]
                inst_cost += v.cpu_demand * node_prices.get(snode_id, 1.0)
            vlink_delays = []
            for _, paths in sol["link_mapping"].items():
                for pi in paths:
                    edges = pi["path"]
                    bw = pi["allocated_bandwidth"]
                    inst_cost += bw * len(edges)
                    pd = 0.0
                    for edge in edges:
                        u, v_id = edge.split("->")
                        pd += link_delays.get((u, v_id), 1.0)
                    vlink_delays.append(pd)
            inst_delay = (sum(vlink_delays) / len(vlink_delays)) if vlink_delays else 0.0
            duration = vnr.lifetime
            cum_rev += rev * duration
            cum_cost_weighted += inst_cost * duration
            cum_inst_cost += inst_cost
            cum_inst_delay += inst_delay

        accept_curve.append(n_success / n_arrived)
        cost_curve.append(cum_inst_cost / n_success if n_success else 0.0)
        revcost_curve.append(cum_rev / cum_cost_weighted if cum_cost_weighted else 0.0)
        delay_curve.append(cum_inst_delay / n_success if n_success else 0.0)

    return {
        "accept": np.array(accept_curve),
        "cost":   np.array(cost_curve),
        "revcost": np.array(revcost_curve),
        "delay":   np.array(delay_curve),
    }


def main():
    node_prices, link_delays = load_substrate_prices(SUBSTRATE_PATH)
    vnrs = read_virtual_requests(REQUESTS_PATH)
    print(f"  {len(vnrs)} VNRs")

    series_mean = {}   # algo_id -> {metric -> mean array}
    series_std = {}

    for algo_id, label, color in ALGORITHMS:
        per_seed = []
        for s in SEEDS:
            path = RESULT_DIR / algo_id / f"seed_{s}" / "solutions.json"
            with open(path) as f:
                sols = json.load(f)
            per_seed.append(compute_running_series(vnrs, sols, node_prices, link_delays))
        # Stack and average per metric
        means = {}
        stds = {}
        for key in ["accept", "cost", "revcost", "delay"]:
            stacked = np.stack([d[key] for d in per_seed], axis=0)   # (n_seeds, n_vnrs)
            means[key] = stacked.mean(axis=0)
            stds[key]  = stacked.std(axis=0)
        series_mean[algo_id] = means
        series_std[algo_id]  = stds
        print(f"  computed {algo_id} across {len(SEEDS)} seeds")

    panels = [
        ("accept",  "Cumulative Acceptance Rate",  "Acceptance",       "higher"),
        ("cost",    "Running Avg Embedding Cost",  "Avg cost",         "lower"),
        ("revcost", "Cumulative Revenue / Cost",   "Rev/Cost",         "higher"),
        ("delay",   "Running Avg Vlink Delay",     "Avg delay",        "lower"),
    ]
    x = np.arange(1, len(vnrs) + 1)

    # 2x2 combined
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"3-seed averages (seeds={SEEDS}) — scenario_100nodes ({len(vnrs)} VNRs)\n"
        "Solid line = mean across seeds, shaded band = ±1 std",
        fontsize=14, fontweight="bold", y=0.995,
    )
    for ax, (key, title, ylabel, direction) in zip(axes.flat, panels):
        for algo_id, label, color in ALGORITHMS:
            m = series_mean[algo_id][key]
            s = series_std[algo_id][key]
            ax.plot(x, m, label=label, color=color, linewidth=2)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            ax.scatter([x[-1]], [m[-1]], color=color, s=42, zorder=5,
                       edgecolor="black", linewidth=0.6)
        ax.set_title(f"{title} ({'↑ better' if direction=='higher' else '↓ better'})",
                     fontsize=12)
        ax.set_xlabel("VNR arrival index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=10, framealpha=0.85)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTPUT_DIR / "metrics_multiseed.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

    # Per-metric standalone
    for key, title, ylabel, direction in panels:
        fig2, ax = plt.subplots(figsize=(11, 5))
        for algo_id, label, color in ALGORITHMS:
            m = series_mean[algo_id][key]
            s = series_std[algo_id][key]
            ax.plot(x, m, label=label, color=color, linewidth=2)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            ax.annotate(
                f"{m[-1]:.3f}" if key in ("accept", "revcost") else f"{m[-1]:.1f}",
                xy=(x[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", color=color, va="center",
            )
        ax.set_title(f"{title} ({'↑ better' if direction=='higher' else '↓ better'}) — 3-seed mean ± std",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("VNR arrival index", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", fontsize=10, framealpha=0.85)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        op = OUTPUT_DIR / f"line_{key}_multiseed.png"
        plt.savefig(op, dpi=140, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {op}")


if __name__ == "__main__":
    main()
