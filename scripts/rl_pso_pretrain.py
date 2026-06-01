#!/usr/bin/env python3
"""Search-aware RL pretrain — train the policy to bias PSO toward lower-cost
mappings, end-to-end via REINFORCE on the post-PSO embedding cost.

This is the Option A pivot after IL hit its expert-imitation ceiling.

Pipeline per VNR (sequential mode, matches eval):
  1. Release expired mappings.
  2. Snapshot substrate state.
  3. Sample N trajectories. Each:
       a. Policy forward + Plackett-Luce sample → top-K candidates per vnode
          (log_probs flow through node_head, link_head, cand_head).
       b. PSO search over the sampled candidates → best particle.
       c. Commit → compute cost → release (back to snapshot via restore).
  4. Group-relative reward: revenue/cost ratio normalized to the group mean.
     Failures → 0 (neutral — keeps gradient signal dominated by which
     successful samples beat the group avg, not by the failure penalty).
  5. REINFORCE: loss = −Σ_t reward_t · Σ log_probs_t.
  6. Commit the BEST trajectory's mapping persistently → substrate state
     evolves naturally for the next VNR (matches eval).

Why this can beat mp_vne on cost:
  mp_vne uses PreCost per-vnode greedy ranking — no coordination between
  vnodes. NN with GCN substrate encoding can learn cross-vnode topology
  awareness (cluster snodes together → lower hop count → lower cost),
  trained end-to-end with the post-PSO cost as reward.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim

from algorithms.il_mp_vne.il_mp_vne import ILMPVNE
from algorithms.il_mp_vne_v3.il_mp_vne_v3 import ILMPVNEV3
from algorithms.il_mp_vne_v5.il_mp_vne_v5 import ILMPVNEV5
from algorithms.il_mp_vne_v6.il_mp_vne_v6 import ILMPVNEV6
from algorithms.il_mp_vne_v7.il_mp_vne_v7 import ILMPVNEV7
from utils.load_dataset import read_substrate, read_virtual_requests

ALGO_CLASSES = {
    "il_mp_vne": ILMPVNE,
    "il_mp_vne_v3": ILMPVNEV3,
    "il_mp_vne_v5": ILMPVNEV5,
    "il_mp_vne_v6": ILMPVNEV6,
    "il_mp_vne_v7": ILMPVNEV7,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:4.0f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m:2d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


def snapshot_substrate(substrate):
    """Capture per-node available_cpu + per-link available_bw."""
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


def sum_log_probs(log_probs_dict):
    """Sum all per-decision log_probs across node/link/cand heads."""
    total = None
    for head_lps in [log_probs_dict.get("node", []),
                     log_probs_dict.get("link", []),
                     log_probs_dict.get("cand", [])]:
        for lp in head_lps:
            total = lp if total is None else (total + lp)
    return total if total is not None else torch.tensor(0.0)


def rollout_with_pso(algo, vn, substrate, snapshot):
    """One trajectory: sample top-K via policy → PSO → cost.

    Returns dict with log_probs (with grad), cost, mapping, ordered_links.
    State is rolled back to snapshot before returning, so subsequent
    rollouts see identical starting state.
    """
    # Forward pass + Plackett-Luce sample (gradient through policy)
    ordered_vnodes, ordered_links, candidates, cand_weights, log_probs, entropies, value = \
        algo.rank_all_nn(vn, sample=True)

    if any(not c for c in candidates):
        # Some vnode has empty candidate pool — no feasible mapping
        restore_substrate(substrate, snapshot)
        return {"log_probs": log_probs, "cost": None,
                "mapping": None, "ordered_links": ordered_links}

    # PSO search (non-differentiable)
    vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
    vlink_indices = []
    for vlink in vn.links.values():
        vlink_indices.append({
            "src_idx": vnode_to_idx[vlink.source],
            "dst_idx": vnode_to_idx[vlink.target],
            "bw": vlink.bandwidth_demand,
        })
    best_particle = algo._pso(candidates, vlink_indices, ordered_vnodes, cand_weights)
    mapping = {ordered_vnodes[i].id: candidates[i][idx].id
               for i, idx in enumerate(best_particle)}

    # Commit → cost → release (transparent — substrate state unchanged after)
    cost = None
    try:
        vlink_paths = algo._commit_mapping_ordered(mapping, vn, ordered_links)
        if vlink_paths:
            cost = algo._compute_cost(mapping, vn, vlink_paths)
            algo.global_controller.release_mapping(mapping, vn, vlink_paths)
    except ValueError:
        cost = None

    # Hard reset to snapshot guarantees no leakage (e.g., from partial commit
    # that raised before full rollback in commit_mapping_ordered's except).
    restore_substrate(substrate, snapshot)

    return {"log_probs": log_probs, "cost": cost,
            "mapping": mapping, "ordered_links": ordered_links}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16,
                   help="VNRs per gradient step (gradient accumulation).")
    p.add_argument("--num-samples", type=int, default=4,
                   help="N trajectories per VNR (group baseline size).")
    p.add_argument("--checkpoint", default="checkpoints/il_mp_vne_rl_pso.pt")
    p.add_argument("--log-file", default="logs/rl_pso_pretrain.csv")
    p.add_argument("--resume", default=None,
                   help="Warm-start from this checkpoint (recommended).")
    p.add_argument("--algorithm", default="il_mp_vne",
                   choices=list(ALGO_CLASSES.keys()),
                   help="Which architecture to train.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)

    print(f"Loading substrate from {args.substrate}")
    substrate = read_substrate(args.substrate)
    print(f"Loading VNRs from {args.requests}")
    vnrs = read_virtual_requests(args.requests)
    if args.episodes > 0:
        vnrs = vnrs[:args.episodes]
    print(f"  Using {len(vnrs)} VNRs")

    algo = ALGO_CLASSES[args.algorithm]()
    algo._init_controller(substrate)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
        missing, unexpected = algo.policy.load_state_dict(state, strict=False)
        print(f"  Warm-started from {args.resume}")
        if missing:
            print(f"  Missing keys: {missing}")

    # Optimizer over all policy params (includes node/link/cand heads + GCN).
    optimizer = optim.AdamW(
        algo.policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("timestamp,batch,ratio_mean,avg_cost,success_rate,reward_std,grad_loss\n")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"RL-PSO pretrain: {len(vnrs)} VNRs, batch={args.batch_size}, N={args.num_samples}, lr={args.lr}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  log: {log_path}")
    print("-" * 72)

    algo.policy.train()
    # Reset substrate to clean state at start.
    algo.global_controller.reset_allocations()
    algo.global_controller.clear_caches()

    active = {}   # vnr.id -> {mapping, vn, paths, expire}
    optimizer.zero_grad()

    start = time.time()
    accum_loss = 0.0
    accum_count = 0
    batch_idx = 0
    recent_rewards = []
    recent_ratios = []   # raw rev/cost ratios of successful trajectories
    recent_costs = []
    recent_total = 0
    recent_fails = 0
    head_rewards = []
    tail_rewards = []
    head_n = tail_n = 100

    for ep, vnr in enumerate(vnrs):
        # Release expired
        expired_ids = [rid for rid, data in active.items()
                       if data["expire"] <= vnr.arrival_time]
        for rid in expired_ids:
            data = active.pop(rid)
            algo.global_controller.release_mapping(data["mapping"], data["vn"], data["paths"])
        algo.global_controller.clear_caches()

        vn = vnr.virtual_network
        snapshot = snapshot_substrate(substrate)

        # N rollouts on identical starting state
        trajectories = []
        for _ in range(args.num_samples):
            t = rollout_with_pso(algo, vn, substrate, snapshot)
            trajectories.append(t)

        # Revenue is constant across rollouts of the same VN.
        rev = (sum(v.cpu_demand for v in vn.nodes.values())
               + sum(l.bandwidth_demand for l in vn.links.values()))

        # Per-rollout revenue/cost ratio. Failures get 0 (neutral, not −1).
        # This is the KEY change vs cost-based reward: failures no longer
        # dominate the gradient with large negative signal.
        ratios = [(rev / t["cost"]) if t["cost"] is not None else 0.0
                  for t in trajectories]
        mean_ratio = sum(ratios) / len(ratios)
        successful = [t for t in trajectories if t["cost"] is not None]

        # Build loss (REINFORCE) with group-relative rev/cost reward
        traj_loss = None
        for t, r in zip(trajectories, ratios):
            # Group-relative — successes with higher ratio get positive reward,
            # failures + lower-ratio successes get negative.
            reward = (r - mean_ratio) / max(abs(mean_ratio), 0.01)
            recent_rewards.append(reward)
            recent_total += 1
            if t["cost"] is None:
                recent_fails += 1
            else:
                recent_costs.append(t["cost"])
                recent_ratios.append(r)   # rev/cost of this successful trajectory
            if ep < head_n:
                head_rewards.append(reward)
            if ep >= len(vnrs) - tail_n:
                tail_rewards.append(reward)

            lp_sum = sum_log_probs(t["log_probs"])
            if lp_sum.requires_grad:
                term = -float(reward) * lp_sum
                traj_loss = term if traj_loss is None else (traj_loss + term)

        if traj_loss is not None:
            accum_loss = traj_loss if accum_count == 0 else (accum_loss + traj_loss)
            accum_count += 1

        # Commit BEST trajectory persistently for state evolution.
        if successful:
            best = min(successful, key=lambda t: t["cost"])
            try:
                paths = algo._commit_mapping_ordered(best["mapping"], vn, best["ordered_links"])
                if paths:
                    active[vnr.id] = {
                        "mapping": best["mapping"], "vn": vn, "paths": paths,
                        "expire": vnr.arrival_time + vnr.lifetime,
                    }
            except ValueError:
                pass

        # Backprop at batch boundary
        if (ep + 1) % args.batch_size == 0:
            if accum_count > 0:
                # Mean over the batch_size accumulated trajectories
                final_loss = accum_loss / accum_count
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_norm=5.0)
                optimizer.step()
                grad_loss_val = final_loss.detach().item()
            else:
                grad_loss_val = 0.0
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            batch_idx += 1

            if batch_idx % args.print_every == 0:
                elapsed = time.time() - start
                eta = elapsed * (len(vnrs) - ep - 1) / max(ep + 1, 1)
                # ratio = rev/cost of successful trajectories — the QUANTITY
                # we're maximizing. Group-relative `r_mean` is ~0 by design
                # and uninformative; ratio shows actual embedding efficiency.
                ratio_mean = sum(recent_ratios) / len(recent_ratios) if recent_ratios else 0.0
                # Std of within-batch reward — magnitude of learning signal.
                if len(recent_rewards) > 1:
                    rew_mean = sum(recent_rewards) / len(recent_rewards)
                    rew_std = (sum((x - rew_mean) ** 2 for x in recent_rewards) / len(recent_rewards)) ** 0.5
                else:
                    rew_std = 0.0
                c_mean = sum(recent_costs) / len(recent_costs) if recent_costs else float("nan")
                succ = (recent_total - recent_fails) / max(recent_total, 1)
                print(
                    f"  ep {ep+1:5d}/{len(vnrs)}  batch {batch_idx:4d}  "
                    f"ratio={ratio_mean:.3f} r_std={rew_std:.3f} "
                    f"cost={c_mean:6.1f} succ={succ:.0%}  "
                    f"loss={grad_loss_val:+.4f}  held={len(active):3d}  "
                    f"elapsed={fmt_secs(elapsed)} eta={fmt_secs(eta)}"
                )
                with open(log_path, "a") as f:
                    f.write(f"{time.time()},{batch_idx},{ratio_mean:.4f},{c_mean:.2f},"
                            f"{succ:.4f},{rew_std:.4f},{grad_loss_val:.4f}\n")
            recent_rewards.clear()
            recent_ratios.clear()
            recent_costs.clear()
            recent_fails = 0
            recent_total = 0

    # Drain
    if accum_count > 0:
        final_loss = accum_loss / accum_count
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

    torch.save({
        "policy_state_dict": algo.policy.state_dict(),
        "config": algo.config,
        "episodes": len(vnrs),
        "num_samples": args.num_samples,
    }, ckpt_path)

    elapsed = time.time() - start
    print("-" * 72)
    print(f"RL-PSO pretrain done in {fmt_secs(elapsed)}.")
    if head_rewards:
        h = sum(head_rewards) / len(head_rewards)
        print(f"  head-{len(head_rewards)} avg reward: {h:+.4f}")
    if tail_rewards:
        t = sum(tail_rewards) / len(tail_rewards)
        print(f"  tail-{len(tail_rewards)} avg reward: {t:+.4f}")
    print(f"  checkpoint → {ckpt_path}")
    print(f"  log → {log_path}")


if __name__ == "__main__":
    main()
