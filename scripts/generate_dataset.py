#!/usr/bin/env python3
"""Generate scenario_stress_v2: 10 sparse domains × 10 nodes with TIGHT resources.

Design goals (per user spec):
  - Multi-domain: 10 domains
  - ~100 nodes total (10 nodes/domain)
  - Sparse intra-domain graphs (NOT near-complete; degree ~3 per node)
  - Sparse inter-domain (enough to keep the union connected; ring + extras)
  - Tight CPU/BW so embedding is genuinely constrained

Intra-domain topology = MST (Kruskal over random points) + few shortest non-MST
edges → guarantees connectivity at minimum density. Same recipe as generate_conus75.

Inter-domain topology = ring across all domains + a few extra random pairs
→ guarantees the whole multi-domain graph is connected, no isolated cluster.
"""
import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path


# ---------------- substrate ----------------

def _sparse_intra_domain_edges(rng: random.Random, n: int, target_edges: int):
    """Random-geometric layout → MST (Kruskal) → add shortest non-MST edges."""
    positions = [(rng.uniform(0, 1.0), rng.uniform(0, 1.0)) for _ in range(n)]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dists.append((math.hypot(dx, dy), i, j))
    dists.sort()

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    edge_set = set()
    # Step 1: MST via Kruskal (n-1 edges).
    for _, i, j in dists:
        ri, rj = find(i), find(j)
        if ri != rj:
            edges.append((i, j))
            edge_set.add(frozenset([i, j]))
            parent[ri] = rj

    # Step 2: add shortest non-MST edges until target.
    for _, i, j in dists:
        if len(edges) >= target_edges:
            break
        if frozenset([i, j]) not in edge_set:
            edges.append((i, j))
            edge_set.add(frozenset([i, j]))
    return edges


def generate_substrate(seed: int = 42, num_domains: int = 10, nodes_per_domain: int = 10):
    rng = random.Random(seed)

    domains = []
    for d in range(num_domains):
        nodes = []
        for i in range(nodes_per_domain):
            nodes.append({
                "id": f"domain_{d}_node_{i}",
                # TIGHT: 10-30 (vs scenario_stress 50-100). ~3x stricter.
                "cpu_capacity": round(rng.uniform(10.0, 30.0), 2),
                "cpu_price": round(rng.uniform(1.0, 5.0), 2),
                "processing_delay": round(rng.uniform(0.1, 2.0), 2),
            })

        # Sparse intra: MST + ~20% extras → avg degree ~2.4-2.5 (sparse, not complete).
        # Scales with size: 10 nodes → 11 edges; 100 nodes → 124 edges.
        target_edges = max(nodes_per_domain - 1 + nodes_per_domain // 5, nodes_per_domain - 1)
        edge_pairs = _sparse_intra_domain_edges(rng, nodes_per_domain, target_edges)
        links = []
        for i, j in edge_pairs:
            links.append({
                "source": f"domain_{d}_node_{i}",
                "target": f"domain_{d}_node_{j}",
                # TIGHT: 30-100 (vs scenario_stress 500-1000). ~10x stricter.
                "bandwidth_capacity": round(rng.uniform(30.0, 100.0), 2),
                "bandwidth_price": round(rng.uniform(0.1, 1.0), 2),
                "transmission_delay": round(rng.uniform(1.0, 10.0), 2),
            })

        domains.append({
            "id": f"domain_{d}",
            "nodes": nodes,
            "links": links,
        })

    # Inter-domain: ring (connect d_i to d_{i+1 mod N}) + a few extra random
    # pairs. Total target ~ num_domains * 1.5 to keep it sparse but robust.
    inter_links = []
    target_inter = int(num_domains * 1.5)  # 15 for 10 domains

    # Ring backbone (num_domains inter-links).
    pairs = [(d, (d + 1) % num_domains) for d in range(num_domains)]
    # Extras (random non-ring pairs).
    rng2 = random.Random(seed + 1)
    while len(pairs) < target_inter:
        i = rng2.randrange(num_domains)
        j = rng2.randrange(num_domains)
        if i == j:
            continue
        if (i, j) in pairs or (j, i) in pairs:
            continue
        pairs.append((i, j))

    for di, dj in pairs:
        src = rng2.choice(domains[di]["nodes"])["id"]
        dst = rng2.choice(domains[dj]["nodes"])["id"]
        inter_links.append({
            "source": src,
            "target": dst,
            # TIGHT but higher than intra: 100-300.
            "bandwidth_capacity": round(rng2.uniform(100.0, 300.0), 2),
            "bandwidth_price": round(rng2.uniform(0.5, 2.0), 2),
            "transmission_delay": round(rng2.uniform(5.0, 20.0), 2),
        })

    return {"domains": domains, "inter_domain_links": inter_links}


# ---------------- VN requests ----------------

def generate_vn(vnr_id: str, arrival_time: float, num_domains: int, rng: random.Random):
    """Small VN (3-7 nodes) with 60% allowed_domains constraint and sparse links."""
    num_nodes = rng.randint(3, 7)
    lifetime = round(rng.expovariate(1.0 / 500.0), 2)
    edge_prob = 0.3  # sparse VN

    nodes = []
    for i in range(num_nodes):
        # 60% chance of regional constraint (matches existing multi-domain scenarios)
        allowed = []
        if rng.random() < 0.6:
            num_allowed = rng.randint(1, min(2, num_domains))
            allowed = rng.sample([f"domain_{d}" for d in range(num_domains)], num_allowed)
        nodes.append({
            "id": f"{vnr_id}_node_{i}",
            # TIGHT: 1-8 (substrate has 10-30, so ~3-4 vnodes per snode max)
            "cpu_demand": round(rng.uniform(1.0, 8.0), 2),
            "allowed_domains": allowed,
        })

    # Spanning tree first → guarantee connectivity.
    shuffled = list(range(num_nodes))
    rng.shuffle(shuffled)
    links = []
    used = set()
    for i in range(1, num_nodes):
        p = shuffled[rng.randint(0, i - 1)]
        c = shuffled[i]
        a, b = (p, c) if p < c else (c, p)
        used.add((a, b))
        links.append({
            "source": f"{vnr_id}_node_{a}",
            "target": f"{vnr_id}_node_{b}",
            # TIGHT: 5-25 (intra has 30-100, so ~3-5 vlinks per slink max)
            "bandwidth_demand": round(rng.uniform(5.0, 25.0), 2),
        })
    # Add extra random edges with edge_prob.
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) in used:
                continue
            if rng.random() < edge_prob:
                links.append({
                    "source": f"{vnr_id}_node_{i}",
                    "target": f"{vnr_id}_node_{j}",
                    "bandwidth_demand": round(rng.uniform(5.0, 25.0), 2),
                })

    return {
        "id": vnr_id,
        "arrival_time": round(arrival_time, 2),
        "lifetime": lifetime,
        "virtual_network": {
            "id": f"vn_{vnr_id}",
            "nodes": nodes,
            "links": links,
        },
    }


# ---------------- driver ----------------

def generate_dataset(out_dir: Path, num_requests: int = 3000, seed: int = 42,
                     num_domains: int = 10, nodes_per_domain: int = 10,
                     arrival_rate: float = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    substrate = generate_substrate(seed, num_domains, nodes_per_domain)
    with open(out_dir / "substrate.json", "w") as f:
        json.dump(substrate, f, indent=2)

    # Scale arrival_rate with substrate size to keep utilization regime constant.
    # Reference: 10 nodes/domain → 0.1 (= 50 concurrent VNs at lifetime 500).
    # For substrate Nx larger, arrival_rate scales by N to keep same utilization %.
    if arrival_rate is None:
        arrival_rate = 0.1 * (nodes_per_domain / 10.0)
    rng = random.Random(seed + 100)
    current_time = 0.0
    requests = []
    for i in range(num_requests):
        current_time += rng.expovariate(arrival_rate)
        requests.append(generate_vn(f"vnr_{i}", current_time, num_domains, rng))
    with open(out_dir / "virtual_requests.json", "w") as f:
        json.dump(requests, f, indent=2)

    # ---- report ----
    print(f"Generated dataset → {out_dir}")
    print()
    total_nodes = sum(len(d["nodes"]) for d in substrate["domains"])
    total_intra = sum(len(d["links"]) for d in substrate["domains"])
    total_inter = len(substrate["inter_domain_links"])
    cpu_caps = [n["cpu_capacity"] for d in substrate["domains"] for n in d["nodes"]]
    bw_caps = [l["bandwidth_capacity"] for d in substrate["domains"] for l in d["links"]]
    inter_bw = [l["bandwidth_capacity"] for l in substrate["inter_domain_links"]]
    print(f"Substrate ({num_domains} domains × {nodes_per_domain} nodes):")
    print(f"  Total nodes        : {total_nodes}")
    print(f"  Intra-domain links : {total_intra}  ({total_intra/num_domains:.1f}/domain)")
    print(f"  Inter-domain links : {total_inter}")
    print(f"  CPU capacity       : [{min(cpu_caps):.1f}, {max(cpu_caps):.1f}]  mean {statistics.mean(cpu_caps):.1f}")
    print(f"  Intra BW           : [{min(bw_caps):.1f}, {max(bw_caps):.1f}]  mean {statistics.mean(bw_caps):.1f}")
    if inter_bw:
        print(f"  Inter BW           : [{min(inter_bw):.1f}, {max(inter_bw):.1f}]  mean {statistics.mean(inter_bw):.1f}")

    vn_sizes = [len(r["virtual_network"]["nodes"]) for r in requests]
    vn_links = [len(r["virtual_network"]["links"]) for r in requests]
    cpu_dems = [n["cpu_demand"] for r in requests for n in r["virtual_network"]["nodes"]]
    bw_dems = [l["bandwidth_demand"] for r in requests for l in r["virtual_network"]["links"]]
    constrained = sum(1 for r in requests for n in r["virtual_network"]["nodes"] if n["allowed_domains"])
    total_vnodes = sum(vn_sizes)
    print()
    print(f"Virtual Requests ({num_requests}):")
    print(f"  VN size            : [{min(vn_sizes)}, {max(vn_sizes)}]  mean {statistics.mean(vn_sizes):.1f}")
    print(f"  VN links / VN      : mean {statistics.mean(vn_links):.1f}")
    print(f"  CPU demand         : [{min(cpu_dems):.1f}, {max(cpu_dems):.1f}]  mean {statistics.mean(cpu_dems):.1f}")
    print(f"  BW demand          : [{min(bw_dems):.1f}, {max(bw_dems):.1f}]  mean {statistics.mean(bw_dems):.1f}")
    print(f"  allowed_domains    : {constrained}/{total_vnodes} vnodes ({100*constrained/total_vnodes:.0f}%)")

    # Scarcity ratios.
    total_cpu = sum(cpu_caps)
    avg_vn_cpu = statistics.mean(cpu_dems) * statistics.mean(vn_sizes)
    avg_vn_bw_total = statistics.mean(bw_dems) * statistics.mean(vn_links)
    total_intra_bw = sum(bw_caps)
    print()
    print(f"Scarcity check:")
    print(f"  CPU supply / VN cpu need : {total_cpu:.0f} / {avg_vn_cpu:.1f} → ~{total_cpu/avg_vn_cpu:.0f} concurrent VNs feasible")
    print(f"  Intra BW / VN bw need    : {total_intra_bw:.0f} / {avg_vn_bw_total:.1f} → ~{total_intra_bw/avg_vn_bw_total:.0f} concurrent VNs feasible")
    print(f"  Expected concurrent      : arrival_rate * lifetime = {arrival_rate:.2f} * 500 = {int(arrival_rate*500)} VNs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, help="Output directory name (e.g. scenario_100nodes)")
    p.add_argument("--requests", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domains", type=int, default=10)
    p.add_argument("--nodes-per-domain", type=int, default=10)
    p.add_argument("--arrival-rate", type=float, default=None,
                   help="Override auto-scaled arrival rate. Default scales with nodes_per_domain.")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out = project_root / "datasets" / args.scenario
    generate_dataset(out, args.requests, args.seed, args.domains, args.nodes_per_domain, args.arrival_rate)


if __name__ == "__main__":
    main()
