"""Build improved targets for v13 self-distillation.

For each VNR, picks the BEST mapping from:
- mp_vne expert (strong_targets_100nodes.json) — multi-restart mp_vne, best-of-3
- v10 inference output (solutions.json) — v6 model + multi-restart PSO

Output format matches strong_targets_100nodes.json so imitation_pretrain.py
can use it via --targets-file.
"""
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert-targets", default="data/strong_targets_100nodes.json")
    p.add_argument("--v10-solutions", required=True,
                   help="solutions.json from running v10 on TRAIN VNRs")
    p.add_argument("--output", default="data/improved_targets_100nodes.json")
    args = p.parse_args()

    with open(args.expert_targets) as f:
        expert = {e["vnr_id"]: e for e in json.load(f)}
    with open(args.v10_solutions) as f:
        v10 = {s["vnr_id"]: s for s in json.load(f)}

    print(f"Expert targets: {len(expert)} entries")
    print(f"V10 solutions:  {len(v10)} entries")

    n_expert_only = 0      # expert succeeds, v10 fails
    n_v10_only = 0         # expert fails, v10 succeeds (NEW labels!)
    n_v10_better = 0       # both succeed, v10 cost < expert cost
    n_expert_better = 0    # both succeed, expert cost <= v10 cost
    n_both_fail = 0
    cost_delta_sum = 0.0   # sum(expert_cost - new_cost) when improved

    out = []
    for vnr_id in expert.keys():
        e = expert[vnr_id]
        s = v10.get(vnr_id)
        e_ok = e.get("successful") and e.get("mapping")
        s_ok = s and s.get("is_successful") and s.get("node_mapping")
        e_cost = e.get("cost") if e_ok else None
        s_cost = s.get("embedding_cost") if s_ok else None

        # Decision
        winner_entry = None
        if e_ok and s_ok:
            if s_cost < e_cost:
                n_v10_better += 1
                cost_delta_sum += (e_cost - s_cost)
                winner_entry = {
                    "vnr_id": vnr_id,
                    "arrival_time": e["arrival_time"],
                    "lifetime": e["lifetime"],
                    "successful": True,
                    "mapping": s["node_mapping"],
                    "cost": s_cost,
                    "seed": "v10",
                }
            else:
                n_expert_better += 1
                winner_entry = e
        elif e_ok:
            n_expert_only += 1
            winner_entry = e
        elif s_ok:
            n_v10_only += 1
            cost_delta_sum += s_cost  # any new label = pure gain
            winner_entry = {
                "vnr_id": vnr_id,
                "arrival_time": e["arrival_time"],
                "lifetime": e["lifetime"],
                "successful": True,
                "mapping": s["node_mapping"],
                "cost": s_cost,
                "seed": "v10_only",
            }
        else:
            n_both_fail += 1
            winner_entry = e  # keep null mapping

        out.append(winner_entry)

    n_success_new = sum(1 for e in out if e.get("successful"))
    n_success_old = sum(1 for e in expert.values() if e.get("successful"))

    print(f"\n--- Merge stats ---")
    print(f"  Both fail:           {n_both_fail}")
    print(f"  Expert-only success: {n_expert_only}")
    print(f"  V10-only success:    {n_v10_only}    ← NEW labels!")
    print(f"  Both: v10 better:    {n_v10_better}  ← improved labels!")
    print(f"  Both: expert better: {n_expert_better}")
    print(f"\n  Old #success: {n_success_old} / {len(expert)} ({n_success_old/len(expert):.1%})")
    print(f"  New #success: {n_success_new} / {len(out)} ({n_success_new/len(out):.1%})")
    print(f"  Acceptance gain: +{n_success_new - n_success_old}")
    print(f"  Sum cost delta (improvement): {cost_delta_sum:.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out)} improved targets → {args.output}")


if __name__ == "__main__":
    main()
