#!/usr/bin/env python3
"""Offline generation of multi-restart mp_vne targets for IL.

For each VNR in the stream (sequential, with arrival/lifetime → substrate
state evolves naturally):
  1. Snapshot substrate state.
  2. Run mp_vne N times with DIFFERENT RNG seeds (PSO hyperparams unchanged).
  3. Pick the lowest-cost mapping (best-of-N).
  4. Restore state, then re-run mp_vne with the winning seed to commit
     persistently → substrate evolves under the strong expert's decisions.
  5. Save (vnr_id → mapping, cost) to JSON.

PSO hyperparameters in EACH individual run are identical to mp_vne baseline
(num_particles=20, num_iterations=15) so any "training advantage" comes from
selecting the best of N independent runs — not from search-budget inflation.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.mp_vne.legacy import MPVNELegacy
from utils.load_dataset import read_substrate, read_virtual_requests


def snapshot_substrate(substrate):
    snap = {}
    for d in substrate.domains.values():
        for node in d.network.nodes.values():
            snap[('cpu', node.id)] = getattr(node, 'available_cpu', node.cpu_capacity)
        for link in d.network.links.values():
            snap[('bw', link.source, link.target)] = getattr(link, 'available_bw', link.bandwidth_capacity)
    for link in substrate.inter_domain_links.values():
        snap[('bw', link.source, link.target)] = getattr(link, 'available_bw', link.bandwidth_capacity)
    return snap


def restore_substrate(substrate, snap):
    for d in substrate.domains.values():
        for node in d.network.nodes.values():
            node.available_cpu = snap[('cpu', node.id)]
        for link in d.network.links.values():
            link.available_bw = snap[('bw', link.source, link.target)]
    for link in substrate.inter_domain_links.values():
        link.available_bw = snap[('bw', link.source, link.target)]


def fmt_secs(s):
    if s < 60:
        return f"{s:.0f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--limit", type=int, default=0,
                   help="Max number of VNRs to process (0 = all).")
    p.add_argument("--output", required=True,
                   help="Output JSON path for (vnr_id → mapping, cost).")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--resume", action="store_true",
                   help="If output exists, resume from where it left off.")
    args = p.parse_args()

    substrate = read_substrate(args.substrate)
    vnrs = read_virtual_requests(args.requests)
    if args.limit > 0:
        vnrs = vnrs[:args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing_list = json.load(f)
        existing = {e["vnr_id"]: e for e in existing_list}
        print(f"Resumed: {len(existing)} VNRs already processed.")

    # Persistent expert that drives substrate evolution (its _active_mappings
    # is the canonical state; restarts use ISOLATED copies so they can't
    # corrupt the shared state via _release_expired pops).
    import copy as _copy
    expert = MPVNELegacy()

    print(f"Strong target generation:")
    print(f"  substrate: {args.substrate}")
    print(f"  vnrs:      {len(vnrs)} ({args.requests})")
    print(f"  restarts:  N={args.num_restarts}")
    print(f"  output:    {out_path}")
    print("-" * 75)

    results = list(existing.values())
    start = time.time()
    n_succ = sum(1 for r in results if r.get("successful"))
    n_done = len(existing)

    for i, vnr in enumerate(vnrs):
        if vnr.id in existing:
            continue

        # Snapshot once per VNR
        snap = snapshot_substrate(substrate)

        # N restarts — each with an ISOLATED copy of _active_mappings so
        # _release_expired's pops don't drift the canonical state.
        best_cost = float("inf")
        best_seed = None
        best_mapping = None
        for k in range(args.num_restarts):
            restore_substrate(substrate, snap)
            seed = args.seed_base + i * args.num_restarts + k
            random.seed(seed)
            mp_try = MPVNELegacy()
            # Deep copy of expert's lifecycle so restart can release/commit
            # without permanently mutating canonical state.
            mp_try._active_mappings = _copy.deepcopy(expert._active_mappings)
            try:
                sol = mp_try.solve(substrate, vnr)
            except Exception:
                sol = None
            if sol is not None and sol.is_successful and sol.embedding_cost < best_cost:
                best_cost = sol.embedding_cost
                best_seed = seed
                best_mapping = dict(sol.node_mapping)

        # Restore — then commit best run for state evolution
        restore_substrate(substrate, snap)
        successful = False
        if best_seed is not None:
            random.seed(best_seed)
            try:
                final_sol = expert.solve(substrate, vnr)
                if final_sol.is_successful:
                    successful = True
                    # Verify it produced the same mapping (should, since seed is the same)
                    best_mapping = dict(final_sol.node_mapping)
                    best_cost = final_sol.embedding_cost
            except Exception:
                pass

        entry = {
            "vnr_id": vnr.id,
            "arrival_time": vnr.arrival_time,
            "lifetime": vnr.lifetime,
            "successful": successful,
            "mapping": best_mapping if successful else None,
            "cost": best_cost if successful else None,
            "seed": best_seed,
        }
        results.append(entry)
        n_done += 1
        if successful:
            n_succ += 1

        # Periodic save + print
        if (i + 1) % args.print_every == 0:
            elapsed = time.time() - start
            rate = n_done / max(elapsed, 1)
            remain = (len(vnrs) - n_done) / max(rate, 1e-6)
            print(f"  {i+1}/{len(vnrs)}  succ={n_succ}/{n_done} ({100*n_succ/max(n_done,1):.1f}%)  "
                  f"elapsed={fmt_secs(elapsed)}  eta={fmt_secs(remain)}")
            # Save checkpoint
            with open(out_path, "w") as f:
                json.dump(results, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print("-" * 75)
    print(f"Done in {fmt_secs(elapsed)}.")
    print(f"  total      : {n_done}")
    print(f"  successful : {n_succ} ({100*n_succ/max(n_done,1):.1f}%)")
    print(f"  output     : {out_path}")


if __name__ == "__main__":
    main()
