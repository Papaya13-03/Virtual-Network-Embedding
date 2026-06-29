#!/usr/bin/env python3
"""Run an algorithm on an eval scenario, save solutions, compute 5 metrics.

Metrics (final value after full VNR stream):
  - acceptance_rate    = successful / total
  - revenue_rate       = cumulative_revenue / current_time   (long-term avg)
  - avg_cost           = mean embedding cost per successful VN
  - revenue_cost_ratio = cumulative_revenue / cumulative_cost
  - avg_delay          = mean per-vlink path delay per successful VN

Online learning is disabled so the policy is measured as-is.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from algorithms.registry import get_algorithm
from utils.load_dataset import read_substrate, read_virtual_requests
from utils.save_solution import write_solutions


def load_substrate_metadata(substrate_path: str):
    """Pull node_prices and link transmission_delays from raw JSON
    (cheaper than re-parsing through SubstrateNetwork classes)."""
    with open(substrate_path, "r") as f:
        sub = json.load(f)
    node_prices = {}
    link_delays = {}
    for d in sub["domains"]:
        for n in d["nodes"]:
            node_prices[n["id"]] = n.get("cpu_price", 1.0)
        for l in d.get("links", []) + d.get("intra_domain_links", []):
            link_delays[(l["source"], l["target"])] = l.get("transmission_delay", 1.0)
            link_delays[(l["target"], l["source"])] = l.get("transmission_delay", 1.0)
    for l in sub.get("inter_domain_links", []):
        link_delays[(l["source"], l["target"])] = l.get("transmission_delay", 5.0)
        link_delays[(l["target"], l["source"])] = l.get("transmission_delay", 5.0)
    return node_prices, link_delays


def compute_metrics(vnrs, solutions, node_prices, link_delays):
    """Compute 5 final-value metrics from VNRs + solutions."""
    sol_by_id = {s.vnr_id: s for s in solutions}
    # Process in arrival order
    items = sorted(
        [(r, sol_by_id.get(r.id)) for r in vnrs],
        key=lambda x: x[0].arrival_time,
    )

    successful_count = 0
    cumulative_rev = 0.0
    cumulative_cost = 0.0
    total_inst_cost = 0.0
    total_inst_delay = 0.0
    last_time = 0.0

    for vnr, sol in items:
        last_time = vnr.arrival_time
        if sol is None or not sol.is_successful:
            continue
        successful_count += 1

        vn = vnr.virtual_network
        rev = sum(n.cpu_demand for n in vn.nodes.values())
        rev += sum(l.bandwidth_demand for l in vn.links.values())

        # Per-VN cost: node CPU cost + per-vlink hop cost
        inst_cost = 0.0
        for vnode_id, snode_id in sol.node_mapping.items():
            v = vn.nodes[vnode_id]
            inst_cost += v.cpu_demand * node_prices.get(snode_id, 1.0)

        # Per-vlink delay (avg across vlinks)
        vlink_delays = []
        for vlink_id, paths in sol.link_mapping.items():
            for path_info in paths:
                # path_info is (path_links: List[Tuple[str,str]], bw: float)
                path_links, bw = path_info[0], path_info[1]
                inst_cost += bw * len(path_links)
                pd = 0.0
                for u, v_id in path_links:
                    pd += link_delays.get((u, v_id), 1.0)
                vlink_delays.append(pd)
        inst_delay = sum(vlink_delays) / len(vlink_delays) if vlink_delays else 0.0

        duration = vnr.lifetime
        cumulative_rev += rev * duration
        cumulative_cost += inst_cost * duration
        total_inst_cost += inst_cost
        total_inst_delay += inst_delay

    total = len(items)
    return {
        "n_total": total,
        "n_success": successful_count,
        "acceptance_rate": successful_count / total if total else 0.0,
        "revenue_rate": cumulative_rev / last_time if last_time > 0 else 0.0,
        "avg_cost": total_inst_cost / successful_count if successful_count else 0.0,
        "revenue_cost_ratio": cumulative_rev / cumulative_cost if cumulative_cost else 0.0,
        "avg_delay": total_inst_delay / successful_count if successful_count else 0.0,
    }


def freeze_policy_if_rl(algo):
    """Disable online learning so we measure the checkpoint as-is."""
    if hasattr(algo, "trainer"):
        algo.trainer.record = lambda *a, **k: None
        algo.trainer.update = lambda *a, **k: {}
    if hasattr(algo, "policy"):
        algo.policy.eval()
    if hasattr(algo, "_pretrained"):
        algo._pretrained = True


def run(algorithm_name: str, substrate_path: str, requests_path: str,
        ckpt_path: str = None, output_dir: str = None, limit: int = None,
        seed: int = 42):
    print(f"\n=== Eval: {algorithm_name} (seed={seed}) ===")
    print(f"  substrate:  {substrate_path}")
    print(f"  requests:   {requests_path}")
    print(f"  checkpoint: {ckpt_path or '(none / heuristic)'}")

    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)

    substrate = read_substrate(substrate_path)
    vnrs = read_virtual_requests(requests_path)
    if limit:
        vnrs = vnrs[:limit]

    algo = get_algorithm(algorithm_name)
    algo._master_seed = seed   # propagate to multi-restart wrappers
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
        algo.policy.load_state_dict(state, strict=False)
    if hasattr(algo, "_init_controller"):
        algo._init_controller(substrate)
    freeze_policy_if_rl(algo)

    solutions = []
    start = time.time()
    for i, req in enumerate(vnrs):
        sol = algo.solve(substrate, req)
        solutions.append(sol)
        if (i + 1) % 200 == 0:
            n_ok = sum(1 for s in solutions if s.is_successful)
            print(f"  {i+1}/{len(vnrs)}  succ={n_ok}  ({time.time()-start:.0f}s)")
    elapsed = time.time() - start

    # Metrics
    node_prices, link_delays = load_substrate_metadata(substrate_path)
    metrics = compute_metrics(vnrs, solutions, node_prices, link_delays)
    metrics["elapsed_seconds"] = elapsed
    metrics["algorithm"] = algorithm_name
    metrics["checkpoint"] = ckpt_path

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_solutions(solutions, str(out / "solutions.json"))
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved → {out}/solutions.json, metrics.json")

    print(f"\n  Results [{algorithm_name}] ({elapsed:.0f}s):")
    print(f"    acceptance_rate    : {metrics['acceptance_rate']:.1%}  ({metrics['n_success']}/{metrics['n_total']})")
    print(f"    revenue_rate       : {metrics['revenue_rate']:.2f}")
    print(f"    avg_cost           : {metrics['avg_cost']:.2f}")
    print(f"    revenue_cost_ratio : {metrics['revenue_cost_ratio']:.3f}")
    print(f"    avg_delay          : {metrics['avg_delay']:.2f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    from algorithms.registry import ALGORITHMS
    p.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS.keys()))
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", required=True, help="Output dir for solutions+metrics")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.algorithm, args.substrate, args.requests, args.checkpoint, args.output,
        args.limit, args.seed)


if __name__ == "__main__":
    main()
