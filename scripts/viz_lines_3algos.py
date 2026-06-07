"""Line chart: running metrics across 3000 VNRs on 50-node test set.

3 algos: mp_vne_v4 (heuristic) | Normal ep19 (V19-best) | CF cf ep195 ★

Reads solutions.json + recomputes per-VNR cost from node_mapping + link_mapping
(matching the metrics.json formula). Builds cumulative metrics:
  - Running acceptance rate (cumulative succ count / cumulative VNR count)
  - Running mean cost (cumulative cost / cumulative success count)
  - Running rev/cost ratio (cumulative revenue / cumulative cost)
  - Running avg delay (cumulative hops × delay / cumulative success count)
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

ALGOS = [
    ("mp_vne_v4", "mp_vne_v4_re", "tab:red", "-"),
    ("Normal ep19 (V19-best)", "il_mp_vne_v19_e19", "tab:green", "-"),
    ("CF cf ep195 ★", "il_mp_vne_v19_cf_ep195", "tab:purple", "-"),
]


def load_substrate_and_vnrs():
    sub_p = ROOT / "datasets/scenario_50nodes/substrate.json"
    vnr_p = ROOT / "datasets/scenario_50nodes/virtual_requests.json"
    sub = json.loads(sub_p.read_text())
    vnrs = json.loads(vnr_p.read_text())

    # snode metadata: id -> dict
    snode = {}
    for dom in sub["domains"]:
        for n in dom["nodes"]:
            snode[n["id"]] = n

    # link metadata: tuple(sorted([src, dst])) -> dict
    link = {}
    for dom in sub["domains"]:
        for l in dom["links"]:
            link[tuple(sorted([l["source"], l["target"]]))] = l
    for l in sub.get("inter_domain_links", []):
        link[tuple(sorted([l["source"], l["target"]]))] = l

    return snode, link, vnrs


def per_vnr_metrics(solutions, snode, link, vnrs):
    """Compute (success, cost, revenue, delay, hops) per VNR."""
    vnr_map = {v["id"]: v for v in vnrs}
    results = []

    for sol in solutions:
        vid = sol["vnr_id"]
        if not sol["is_successful"]:
            results.append({"succ": False, "cost": 0.0, "rev": 0.0,
                            "delay": 0.0, "hops": 0})
            continue

        vnr = vnr_map[vid]
        vn = vnr["virtual_network"]
        vn_node_by_id = {n["id"]: n for n in vn["nodes"]}

        # Revenue = Σ cpu_demand + Σ bw_demand
        rev = sum(n["cpu_demand"] for n in vn["nodes"]) \
            + sum(l["bandwidth_demand"] for l in vn["links"])

        # Cost: per VN — Σ cpu_demand × cpu_price + Σ bw_demand × hops × bw_price
        cost = 0.0
        delay_total = 0.0
        hops_total = 0
        for vnid, snid in sol["node_mapping"].items():
            cost += vn_node_by_id[vnid]["cpu_demand"] * snode[snid]["cpu_price"]

        for vlink_key, paths in sol["link_mapping"].items():
            for path in paths:
                bw_alloc = path["allocated_bandwidth"]
                for edge in path["path"]:
                    u, v = edge.split("->")
                    key = tuple(sorted([u, v]))
                    if key in link:
                        cost += bw_alloc * link[key]["bandwidth_price"]
                        delay_total += link[key].get(
                            "transmission_delay", link[key].get("delay", 1.0))
                        hops_total += 1

        results.append({"succ": True, "cost": cost, "rev": rev,
                        "delay": delay_total, "hops": hops_total})
    return results


def running_metrics(results):
    """Cumulative running metrics over the sequence."""
    N = len(results)
    acc = np.zeros(N)
    avg_cost = np.zeros(N)
    rev_cost = np.zeros(N)
    avg_delay = np.zeros(N)

    cum_succ = 0
    cum_cost = 0.0
    cum_rev = 0.0
    cum_delay = 0.0
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
    for label, name, color, ls in ALGOS:
        p = ROOT / f"results/scenario_50nodes/{name}/solutions.json"
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        sols = json.loads(p.read_text())
        results = per_vnr_metrics(sols, snode, link, vnrs)
        acc, ac, rc, ad = running_metrics(results)
        data[label] = (color, ls, acc, ac, rc, ad)

    if not data:
        return

    N = len(next(iter(data.values()))[2])
    x = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (0,0) Acceptance rate
    ax = axes[0, 0]
    for label, (color, ls, acc, _, _, _) in data.items():
        ax.plot(x, acc, ls, color=color, linewidth=2.0, label=label)
    ax.set_title("Running acceptance rate over 3000 VNRs  (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("VNR index"); ax.set_ylabel("Cumulative acceptance (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right")

    # (0,1) Avg cost
    ax = axes[0, 1]
    for label, (color, ls, _, ac, _, _) in data.items():
        ax.plot(x, ac, ls, color=color, linewidth=2.0, label=label)
    ax.set_title("Running avg cost per success  (↓ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("VNR index"); ax.set_ylabel("Cumulative cost / success")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right")

    # (1,0) Rev / Cost
    ax = axes[1, 0]
    for label, (color, ls, _, _, rc, _) in data.items():
        ax.plot(x, rc, ls, color=color, linewidth=2.0, label=label)
    ax.set_title("Running revenue / cost  (↑ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("VNR index"); ax.set_ylabel("Σ revenue / Σ cost")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right")

    # (1,1) Avg delay
    ax = axes[1, 1]
    for label, (color, ls, _, _, _, ad) in data.items():
        ax.plot(x, ad, ls, color=color, linewidth=2.0, label=label)
    ax.set_title("Running avg delay per success  (↓ better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("VNR index"); ax.set_ylabel("Cumulative delay / success")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        "50-node test set (3000 VNRs) — running metrics: CF cf ep195 vs Normal V19-best vs mp_vne_v4",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"50nodes_lines_3algos_{DATE_TAG}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
