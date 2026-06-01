"""100-node line plots: running acceptance, cost, revenue/cost over VNR index.

Reads per-VNR solutions.json + virtual_requests.json to compute running
metrics (cumulative through episode i, divided by appropriate denominator).
Provides 4 line panels — same metrics as the bar grid but tracked over time
so improvements are visible across the full eval, not just at the endpoint.

Saves to results/figures/100nodes_lines_<YYYYMMDD_HHMMSS>.png.
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

REQS_PATH = ROOT / "datasets" / "scenario_100nodes" / "virtual_requests.json"
SUBSTRATE_PATH = ROOT / "datasets" / "scenario_100nodes" / "substrate.json"


def load_node_prices():
    s = json.loads(SUBSTRATE_PATH.read_text())
    prices = {}
    for dom in s.get("domains", []):
        for n in dom.get("nodes", []):
            prices[n["id"]] = n.get("cpu_price", 1.0)
    return prices


def inst_cost_from_mapping(vn, sol, node_prices):
    """Match run_eval.py: Σ cpu_demand×cpu_price + Σ bw×hops."""
    cost = 0.0
    vnodes_by_id = {n["id"]: n for n in (vn["nodes"] if isinstance(vn["nodes"], list) else vn["nodes"].values())}
    for vnode_id, snode_id in sol.get("node_mapping", {}).items():
        v = vnodes_by_id[vnode_id]
        cost += v["cpu_demand"] * node_prices.get(snode_id, 1.0)
    for _, paths in sol.get("link_mapping", {}).items():
        for path_info in paths:
            cost += path_info["allocated_bandwidth"] * len(path_info["path"])
    return cost

CASES = [
    # (label, solutions.json path, metrics.json path, color, linestyle)
    ("V19 — no PPO (R2 IL only)",
     "results/scenario_100nodes_multiseed/il_mp_vne_v17_pso_r2/seed_42/solutions.json",
     "results/scenario_100nodes_multiseed/il_mp_vne_v17_pso_r2/seed_42/metrics.json",
     "tab:gray", "--"),
    ("V19 — after PPO",
     "results/scenario_100nodes/il_mp_vne_v17_ppo_direct_via_pso/solutions.json",
     "results/scenario_100nodes/il_mp_vne_v17_ppo_direct_via_pso/metrics.json",
     "tab:blue", "-"),
]


def vnr_revenue(vn_dict):
    """Revenue = Σ cpu_demand + Σ bandwidth_demand of the VN.
    Matches the run_eval.py convention used by the original mp_vne formulation.
    """
    nodes = vn_dict["nodes"]
    links = vn_dict["links"]
    cpu_sum = sum(n["cpu_demand"] for n in (nodes if isinstance(nodes, list) else nodes.values()))
    bw_sum = sum(l["bandwidth_demand"] for l in (links if isinstance(links, list) else links.values()))
    return cpu_sum + bw_sum


def running_metrics(solutions_path, vnrs_meta, node_prices):
    """Match run_eval.py exactly: cost recomputed from node_mapping +
    link_mapping (not solutions.json embedding_cost which has solver-specific
    formulas — v4 has inf bugs there). rev/cost ratio uses duration-weighted
    sums; avg_cost uses per-VN inst_cost (unweighted)."""
    sols = json.loads(Path(solutions_path).read_text())
    sol_by_id = {s["vnr_id"]: s for s in sols}
    items = sorted(((v, sol_by_id.get(v["id"])) for v in vnrs_meta),
                   key=lambda x: x[0]["arrival_time"])
    n = len(items)
    succ = np.zeros(n, dtype=int)
    cost_w = np.zeros(n)            # duration-weighted (rev/cost ratio)
    rev_w = np.zeros(n)             # duration-weighted (rev/cost ratio)
    inst_cost_arr = np.zeros(n)     # unweighted (avg_cost)
    for i, (v_meta, sol) in enumerate(items):
        if sol is None or not sol["is_successful"]:
            continue
        vn = v_meta["virtual_network"]
        duration = v_meta.get("lifetime", 1.0)
        inst_cost = inst_cost_from_mapping(vn, sol, node_prices)
        if not np.isfinite(inst_cost):
            continue
        succ[i] = 1
        cost_w[i] = inst_cost * duration
        rev_w[i] = vnr_revenue(vn) * duration
        inst_cost_arr[i] = inst_cost
    cum_succ = np.cumsum(succ)
    cum_cost_w = np.cumsum(cost_w)
    cum_rev_w = np.cumsum(rev_w)
    cum_inst_cost = np.cumsum(inst_cost_arr)
    idx = np.arange(1, n + 1)
    safe_succ = np.where(cum_succ == 0, 1, cum_succ)
    return {
        "n": idx,
        "acc_rate": cum_succ / idx * 100,
        "avg_cost": np.where(cum_succ == 0, 0, cum_inst_cost / safe_succ),
        "rev_cost": np.where(cum_cost_w == 0, 0, cum_rev_w / np.where(cum_cost_w == 0, 1, cum_cost_w)),
        "cum_succ": cum_succ,
    }


def main():
    vnrs_meta = json.loads(REQS_PATH.read_text())
    node_prices = load_node_prices()

    # Compute running curves for each case.
    rows = []
    for label, sol_p, met_p, color, ls in CASES:
        rm = running_metrics(ROOT / sol_p, vnrs_meta, node_prices)
        m = json.loads((ROOT / met_p).read_text())
        rows.append({"label": label, "rm": rm, "metrics": m, "color": color, "ls": ls})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ---- Acceptance rate ----
    ax = axes[0, 0]
    for r in rows:
        ax.plot(r["rm"]["n"], r["rm"]["acc_rate"],
                color=r["color"], linestyle=r["ls"], linewidth=2.0,
                label=f"{r['label']}  →  {r['metrics']['acceptance_rate']*100:.1f}%")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Running acceptance rate (↑ better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ---- Avg cost (of successful only) ----
    ax = axes[0, 1]
    for r in rows:
        # Skip early steps where avg_cost is 0 (no success yet).
        mask = r["rm"]["cum_succ"] > 0
        ax.plot(r["rm"]["n"][mask], r["rm"]["avg_cost"][mask],
                color=r["color"], linestyle=r["ls"], linewidth=2.0,
                label=f"{r['label']}  →  {r['metrics']['avg_cost']:.1f}")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Avg embedding cost")
    ax.set_title("Running avg cost on successful embeddings (↓ better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ---- Revenue / Cost ratio ----
    ax = axes[1, 0]
    for r in rows:
        mask = r["rm"]["cum_succ"] > 0
        ax.plot(r["rm"]["n"][mask], r["rm"]["rev_cost"][mask],
                color=r["color"], linestyle=r["ls"], linewidth=2.0,
                label=f"{r['label']}  →  {r['metrics']['revenue_cost_ratio']:.3f}")
    ax.set_xlabel("VNRs processed")
    ax.set_ylabel("Cumulative revenue / Cumulative cost")
    ax.set_title("Running revenue / cost ratio (↑ better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ---- Delay (final only — per-VNR delay needs substrate path lookups) ----
    ax = axes[1, 1]
    labels = [r["label"] for r in rows]
    delays = [r["metrics"]["avg_delay"] for r in rows]
    colors = [r["color"] for r in rows]
    bars = ax.bar(range(len(labels)), delays, color=colors, edgecolor="black",
                  linewidth=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Average delay (final) — running curve needs substrate paths")
    ax.set_ylabel("Avg delay")
    ax.grid(alpha=0.3, axis="y")
    best_idx = int(np.argmin(delays))
    for i, v in enumerate(delays):
        marker = "  ★" if i == best_idx else ""
        ax.text(i, v + (max(delays) - min(delays)) * 0.02,
                f"{v:.2f}{marker}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    pad = (max(delays) - min(delays)) * 0.18
    ax.set_ylim(min(delays) - pad * 0.5, max(delays) + pad)

    fig.suptitle("100-node running metrics: V19 before PPO vs V19 after PPO",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_lines_{DATE_TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
