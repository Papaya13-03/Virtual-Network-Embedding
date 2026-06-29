"""Generate a new VNR set against an EXISTING substrate.

Reuses scripts/generate_dataset.py's `generate_vn` so VNR distribution is
identical to the original training set — only the seed differs.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from scripts.generate_dataset import generate_vn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate-from", required=True,
                   help="Existing scenario dir whose substrate.json we reuse.")
    p.add_argument("--scenario", required=True,
                   help="New scenario dir to create (contains symlink/copy of substrate + new VNRs).")
    p.add_argument("--requests", type=int, default=10000)
    p.add_argument("--seed", type=int, required=True,
                   help="Seed for VNR sequence (different from original).")
    p.add_argument("--arrival-rate", type=float, default=None)
    p.add_argument("--num-domains", type=int, default=10)
    # Tunable VNR-distribution knobs (defaults = original scenario_100nodes).
    p.add_argument("--lifetime-mean", type=float, default=500.0)
    p.add_argument("--node-min", type=int, default=3)
    p.add_argument("--node-max", type=int, default=7)
    p.add_argument("--edge-prob", type=float, default=0.3)
    p.add_argument("--cpu-lo", type=float, default=1.0)
    p.add_argument("--cpu-hi", type=float, default=8.0)
    p.add_argument("--bw-lo", type=float, default=5.0)
    p.add_argument("--bw-hi", type=float, default=25.0)
    p.add_argument("--region-prob", type=float, default=0.6)
    args = p.parse_args()

    vn_kwargs = dict(
        lifetime_mean=args.lifetime_mean, node_min=args.node_min,
        node_max=args.node_max, edge_prob=args.edge_prob,
        cpu_lo=args.cpu_lo, cpu_hi=args.cpu_hi,
        bw_lo=args.bw_lo, bw_hi=args.bw_hi, region_prob=args.region_prob,
    )

    project_root = Path(__file__).resolve().parent.parent
    src = project_root / "datasets" / args.substrate_from
    dst = project_root / "datasets" / args.scenario
    dst.mkdir(parents=True, exist_ok=True)

    # Copy substrate.
    shutil.copy(src / "substrate.json", dst / "substrate.json")

    # Auto-scale arrival rate using same formula as generate_dataset.
    arrival_rate = args.arrival_rate
    if arrival_rate is None:
        arrival_rate = 0.1  # matches the original scenario_100nodes test set

    rng = random.Random(args.seed + 100)
    requests = []
    current_time = 0.0
    for i in range(args.requests):
        current_time += rng.expovariate(arrival_rate)
        requests.append(generate_vn(f"vnr_{i}", current_time, args.num_domains,
                                    rng, **vn_kwargs))

    with open(dst / "virtual_requests.json", "w") as f:
        json.dump(requests, f, indent=2)

    print(f"Generated {args.requests} VNRs (seed={args.seed}, arrival_rate={arrival_rate})")
    print(f"  Substrate reused from: {src}")
    print(f"  Saved to: {dst}/virtual_requests.json")
    vn_sizes = [len(r["virtual_network"]["nodes"]) for r in requests]
    print(f"  VN size mean: {sum(vn_sizes)/len(vn_sizes):.1f}, range [{min(vn_sizes)}, {max(vn_sizes)}]")


if __name__ == "__main__":
    main()
