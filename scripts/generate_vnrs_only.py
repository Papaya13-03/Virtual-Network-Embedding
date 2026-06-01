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
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    src = project_root / "datasets" / args.substrate_from
    dst = project_root / "datasets" / args.scenario
    dst.mkdir(parents=True, exist_ok=True)

    # Copy substrate.
    shutil.copy(src / "substrate.json", dst / "substrate.json")

    # Auto-scale arrival rate using same formula as generate_dataset.
    arrival_rate = args.arrival_rate
    if arrival_rate is None:
        # generate_dataset uses arrival_rate = 0.4 * (nodes_per_domain/10) by default
        arrival_rate = 0.4

    rng = random.Random(args.seed + 100)
    requests = []
    current_time = 0.0
    for i in range(args.requests):
        current_time += rng.expovariate(arrival_rate)
        requests.append(generate_vn(f"vnr_{i}", current_time, args.num_domains, rng))

    with open(dst / "virtual_requests.json", "w") as f:
        json.dump(requests, f, indent=2)

    print(f"Generated {args.requests} VNRs (seed={args.seed}, arrival_rate={arrival_rate})")
    print(f"  Substrate reused from: {src}")
    print(f"  Saved to: {dst}/virtual_requests.json")
    vn_sizes = [len(r["virtual_network"]["nodes"]) for r in requests]
    print(f"  VN size mean: {sum(vn_sizes)/len(vn_sizes):.1f}, range [{min(vn_sizes)}, {max(vn_sizes)}]")


if __name__ == "__main__":
    main()
