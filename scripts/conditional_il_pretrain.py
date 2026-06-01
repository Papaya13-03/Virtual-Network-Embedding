"""Conditional IL pretrain — V13 design.

Key idea: standard IL pulls model toward expert ALWAYS, even when model is
already better. This caps model at expert ceiling.

V13 fix: run PSO on model's candidates in the training loop. If model's PSO
output is BETTER than expert (cost-wise), SKIP the CE loss for this VNR.
Otherwise, apply standard CE loss toward expert.

Net effect:
  - Model worse than expert → learn from expert (improve)
  - Model = expert → loss ≈ 0 anyway
  - Model BETTER than expert → no gradient → model retains improvement

State evolution: ALWAYS commit expert mapping (keeps state consistent with
strong_targets labels, avoiding the V13-old distribution shift).

Compared to standard IL:
  - ~3x slower per VNR (PSO inside loop)
  - Same CE loss shape — just gated by cost comparison
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from algorithms.il_mp_vne_v6.il_mp_vne_v6 import ILMPVNEV6
from utils.load_dataset import read_substrate, read_virtual_requests


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fmt(s: float) -> str:
    if s < 60:
        return f"{s:4.0f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m:2d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


def ce_loss_from_cached(cand_scores, cand_pools, vnodes, expert_mapping):
    """CE loss for vnodes whose expert snode is in pool."""
    losses = []
    n_matched = 0
    n_targets = 0
    for i, vnode in enumerate(vnodes):
        expert_snode_id = expert_mapping.get(vnode.id)
        if expert_snode_id is None:
            continue
        n_targets += 1
        pool_i = cand_pools[i]
        scores_i = cand_scores[i]
        target_idx = None
        for j, snode in enumerate(pool_i):
            if snode.id == expert_snode_id:
                target_idx = j
                break
        if target_idx is None:
            continue
        log_probs = F.log_softmax(scores_i, dim=0)
        losses.append(-log_probs[target_idx])
        n_matched += 1
    if not losses:
        return None, n_targets, 0
    return torch.stack(losses).mean(), n_targets, n_matched


def run_pso_score(algo, vn, cand_scores, cand_pools, K=5):
    """Build top-K candidates + run PSO. Returns (mapping_or_None, fitness_or_inf).

    Uses greedy top-K (not sampled) for deterministic comparison with expert.
    """
    vnodes = list(vn.nodes.values())
    candidate_nodes = []
    candidate_weights = []
    for i, v in enumerate(vnodes):
        scores_i = cand_scores[i].detach()
        pool_i = cand_pools[i]
        if not pool_i:
            return None, float("inf")
        k = min(K, len(pool_i))
        picked = torch.topk(scores_i, k).indices.tolist()
        candidate_nodes.append([pool_i[j] for j in picked])
        if picked:
            picked_scores = scores_i[torch.tensor(picked, dtype=torch.long)]
            weights = torch.softmax(picked_scores, dim=0).tolist()
        else:
            weights = []
        candidate_weights.append(weights)

    if any(not c for c in candidate_nodes):
        return None, float("inf")

    vnode_to_idx = {v.id: i for i, v in enumerate(vnodes)}
    vlink_indices = []
    for vlink in vn.links.values():
        vlink_indices.append({
            "src_idx": vnode_to_idx[vlink.source],
            "dst_idx": vnode_to_idx[vlink.target],
            "bw": vlink.bandwidth_demand,
        })

    best_particle = algo._pso(candidate_nodes, vlink_indices, vnodes, candidate_weights)
    fitness = algo._fitness(best_particle, candidate_nodes, vlink_indices, vnodes, candidate_weights)
    if fitness == float("inf"):
        return None, fitness
    mapping = {vnodes[i].id: candidate_nodes[i][idx].id for i, idx in enumerate(best_particle)}
    return mapping, fitness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--targets-file", required=True,
                   help="Pre-computed expert mappings (strong_targets_100nodes.json)")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--candidates-k", type=int, default=5)
    p.add_argument("--checkpoint", default="checkpoints/il_mp_vne_v13_100nodes.pt")
    p.add_argument("--log-file", default="logs/conditional_il_v13.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--resume", default=None,
                   help="Warm-start from existing ckpt (e.g., v6's).")
    args = p.parse_args()

    set_seed(args.seed)
    print(f"Loading substrate from {args.substrate}")
    substrate = read_substrate(args.substrate)
    print(f"Loading VNRs from {args.requests}")
    vnrs = read_virtual_requests(args.requests)
    if args.episodes > 0:
        vnrs = vnrs[:args.episodes]
    print(f"  Using {len(vnrs)} VNRs")

    with open(args.targets_file) as f:
        offline_targets = {e["vnr_id"]: e for e in json.load(f)}
    print(f"  Loaded {len(offline_targets)} expert targets")

    algo = ILMPVNEV6()
    algo._init_controller(substrate)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
        missing, _ = algo.policy.load_state_dict(state, strict=False)
        print(f"  Resumed from {args.resume}")
        if missing:
            print(f"  Missing keys: {missing}")

    optimizer = optim.AdamW(algo.policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")
    log_f.write("episode,batch,loss,expert_succ,model_better_rate,match_rate,elapsed_s\n")

    print(f"Conditional IL: {len(vnrs)} VNRs, batch={args.batch_size}, lr={args.lr}, K={args.candidates_k}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  log:        {args.log_file}")
    print("-" * 72)

    start = time.time()
    accum_loss = 0.0
    accum_count = 0
    batch_losses = []
    batch_idx = 0
    expert_succ_total = 0
    expert_total = 0
    targets_total = 0
    matched_total = 0
    model_better_count = 0      # times model PSO < expert cost
    skipped_loss_count = 0       # how many times loss was skipped

    for ep, vnr in enumerate(vnrs):
        vn = vnr.virtual_network
        target_entry = offline_targets.get(vnr.id)
        expert_total += 1

        algo._release_expired(vnr.arrival_time)
        algo.global_controller.clear_caches()

        # 1. Policy forward — cache scores WITH grad.
        node_scores, link_scores, cand_scores, cand_pools, value = algo._forward_policy(vn)
        vnodes_in_order = list(vn.nodes.values())

        loss = None

        if target_entry and target_entry.get("successful"):
            expert_succ_total += 1
            expert_mapping = target_entry["mapping"]
            expert_cost = target_entry["cost"]

            # 2. Run PSO on model's top-K candidates.
            with torch.no_grad():
                model_mapping, model_cost = run_pso_score(
                    algo, vn, cand_scores, cand_pools, K=args.candidates_k,
                )

            # 3. Compare costs.
            model_wins = (model_mapping is not None and model_cost < expert_cost)
            if model_wins:
                model_better_count += 1
                # SKIP loss — model already beats expert.
                skipped_loss_count += 1
                loss = None
            else:
                # 4. Standard CE loss toward expert.
                loss, n_targets, n_matched = ce_loss_from_cached(
                    cand_scores, cand_pools, vnodes_in_order, expert_mapping,
                )
                targets_total += n_targets
                matched_total += n_matched

            # 5. ALWAYS commit expert mapping (state consistency).
            try:
                algo._commit_mapping_ordered(expert_mapping, vn, list(vn.links.items()))
            except ValueError:
                pass

        # 6. Accumulate + backprop every batch_size.
        if loss is not None:
            accum_loss = accum_loss + loss
            accum_count += 1

        if (ep + 1) % args.batch_size == 0:
            if accum_count > 0:
                avg = accum_loss / accum_count
                avg.backward()
                torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_norm=5.0)
                optimizer.step()
                batch_losses.append(avg.detach().item())
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            batch_idx += 1

            if batch_idx % args.print_every == 0:
                elapsed = time.time() - start
                eta = elapsed * (len(vnrs) - ep - 1) / max(ep + 1, 1)
                recent = batch_losses[-args.print_every:]
                avg_recent = sum(recent) / max(len(recent), 1)
                e_rate = expert_succ_total / max(expert_total, 1)
                mb_rate = model_better_count / max(expert_succ_total, 1)
                print(
                    f"  ep {ep+1:5d}/{len(vnrs)} batch {batch_idx:4d} "
                    f"loss={avg_recent:7.4f}  "
                    f"expert={e_rate:5.1%}  "
                    f"model_better={mb_rate:5.1%}  "
                    f"skipped={skipped_loss_count}  "
                    f"elapsed={fmt(elapsed)} eta={fmt(eta)}"
                )
                log_f.write(
                    f"{ep+1},{batch_idx},{avg_recent:.4f},"
                    f"{e_rate:.4f},{mb_rate:.4f},"
                    f"{matched_total / max(targets_total, 1):.4f},{elapsed:.1f}\n"
                )
                log_f.flush()

    # Final backprop on any leftover.
    if accum_count > 0:
        avg = accum_loss / accum_count
        avg.backward()
        optimizer.step()

    elapsed = time.time() - start
    print("-" * 72)
    print(f"  total time:                  {fmt(elapsed)}")
    print(f"  expert success rate:         {expert_succ_total}/{expert_total} = {expert_succ_total/expert_total:.1%}")
    print(f"  model beats expert rate:     {model_better_count}/{expert_succ_total} = {model_better_count/max(expert_succ_total,1):.1%}")
    print(f"  skipped loss (model better): {skipped_loss_count}")
    print(f"  target match rate:           {matched_total}/{max(targets_total,1)} = {matched_total/max(targets_total,1):.1%}")
    if batch_losses:
        recent = batch_losses[-args.print_every:]
        print(f"  final batch loss avg:        {sum(recent)/len(recent):.4f}")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": algo.policy.state_dict()}, args.checkpoint)
    print(f"  checkpoint saved to {args.checkpoint}")
    print(f"  log saved to        {args.log_file}")
    log_f.close()


if __name__ == "__main__":
    main()
