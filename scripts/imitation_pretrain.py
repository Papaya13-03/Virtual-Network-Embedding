#!/usr/bin/env python3
"""Imitation pretrain — supervised learning of the cand_head from mp_vne.

Why imitation (vs RL):
  RL pretrain on stress_v2 plateau'd at ~10% success rate (vs mp_vne ceiling
  37%). RL signal is too sparse in the substrate-evolved regime: with random
  init, most rollouts fail and provide -1 reward with no information about
  which decision was wrong. Imitation gives DENSE per-vnode supervised signal.

Setup per VNR (sequential, matches eval):
  1. mp_vne handles lifecycle (release_expired)
  2. Policy forward pass on current substrate state → cand_scores per vnode
  3. mp_vne.solve(vnr) → expert mapping (vnode_id → snode_id), commits internally
  4. Cross-entropy loss: -log P_policy(expert_snode) summed across vnodes
  5. Backprop + optimizer step

Train only cand_head. node_head / link_head stay random (they're used at
inference only for ordering; mp_vne uses original vnode order and we mimic
that during imitation, so node_head doesn't need to learn anything specific).

The substrate state evolves naturally via mp_vne's commits — same evolution
the policy will face at eval time → distribution-matched supervision.
"""
import argparse
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
import torch.nn.functional as F
import torch.optim as optim

from algorithms.carl_vne.il_mp_vne_v6 import ILMPVNEV6
from algorithms.carl_vne.il_mp_vne_v16 import ILMPVNEV16
from algorithms.carl_vne.il_mp_vne_v17 import ILMPVNEV17
from algorithms.mp_vne.mp_vne import MPVNE
from utils.load_dataset import read_substrate, read_virtual_requests

ALGO_CLASSES = {
    "il_mp_vne_v6": ILMPVNEV6,
    "il_mp_vne_v16": ILMPVNEV16,
    "il_mp_vne_v17": ILMPVNEV17,
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


def policy_forward(algo, vn):
    """Forward pass returning (cand_scores list, cand_pools list)."""
    _, _, cand_scores, cand_pools, _ = algo._forward_policy(vn)
    return cand_scores, cand_pools


def supervised_loss_from_cached(cand_scores, cand_pools, vnodes, expert_mapping):
    """Cross-entropy on cand_head: -log P(expert_snode) per vnode.

    Uses precomputed cand_scores/cand_pools (from BEFORE expert commits)
    so that the just-committed snode's cpu_slack hasn't been driven negative
    by the commit itself (which would mask it as infeasible).

    Returns (loss, n_targets, n_matched).
    """
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


def main():
    p = argparse.ArgumentParser(description="Imitation pretrain (mp_vne expert).")
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Gradient accumulation: backprop every N VNRs.")
    p.add_argument("--checkpoint", default="checkpoints/il_mp_vne_pretrain.pt")
    p.add_argument("--log-file", default="logs/imitation_pretrain.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=20,
                   help="Print every N batches.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume from. Loads policy_state_dict "
                        "into the algorithm before training continues.")
    p.add_argument("--algorithm", default="il_mp_vne_v17",
                   choices=list(ALGO_CLASSES.keys()),
                   help="Which IL policy architecture to train.")
    p.add_argument("--targets-file", default=None,
                   help="Optional JSON of pre-computed expert mappings "
                        "(from scripts/generate_strong_targets.py). When set, "
                        "the script does NOT run mp_vne online; it loads the "
                        "mapping per vnr_id and commits it persistently — much "
                        "faster training and uses the same multi-restart "
                        "stronger expert as ceiling.")
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
    expert = MPVNE()

    # Pre-computed targets (Option A — stronger expert)
    offline_targets = None
    if args.targets_file:
        import json
        with open(args.targets_file) as f:
            entries = json.load(f)
        offline_targets = {e["vnr_id"]: e for e in entries}
        print(f"  Loaded {len(offline_targets)} offline targets from {args.targets_file}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
        missing, unexpected = algo.policy.load_state_dict(state, strict=False)
        print(f"  Resumed from {args.resume}")
        if missing:
            print(f"  Missing keys: {missing}")

    # Custom optimizer — only train cand_head + the GCN encoder that feeds it.
    # We let node_head/link_head stay random since imitation doesn't constrain
    # vnode ordering (mp_vne uses original order; we iterate in same order
    # during cand prediction).
    optimizer = optim.AdamW(
        algo.policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("timestamp,batch,avg_loss,expert_succ,matched_rate\n")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Imitation pretrain: {len(vnrs)} VNRs, batch={args.batch_size}, lr={args.lr}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  log: {log_path}")
    print("-" * 72)

    algo.policy.train()
    start = time.time()

    batch_losses = []
    expert_succ = 0
    expert_total = 0
    matched_total = 0
    targets_total = 0
    batch_idx = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()

    # Track active mappings ourselves when using offline targets
    active_offline = {}   # vnr_id -> {mapping, vn, paths, expire}

    for ep, vnr in enumerate(vnrs):
        vn = vnr.virtual_network

        if offline_targets is not None:
            # ---- OFFLINE TARGETS MODE: skip mp_vne, use precomputed mapping ----
            # 1. Release expired (managed by us)
            expired = [rid for rid, d in active_offline.items()
                       if d["expire"] <= vnr.arrival_time]
            for rid in expired:
                d = active_offline.pop(rid)
                algo.global_controller.release_mapping(d["mapping"], d["vn"], d["paths"])
            algo.global_controller.clear_caches()

            # 2. Policy forward pass on current state
            cand_scores_cached, cand_pools_cached = policy_forward(algo, vn)
            vnodes_in_order = list(vn.nodes.values())

            # 3. Look up offline target
            target_entry = offline_targets.get(vnr.id)
            expert_total += 1
            success = target_entry is not None and target_entry.get("successful")
            if success:
                expert_succ += 1
                target_mapping = target_entry["mapping"]
                # 4. Loss
                loss, n_targets, n_matched = supervised_loss_from_cached(
                    cand_scores_cached, cand_pools_cached, vnodes_in_order, target_mapping,
                )
                targets_total += n_targets
                matched_total += n_matched
                # 5. Commit target's mapping persistently (state evolution)
                try:
                    paths = algo._commit_mapping_ordered(
                        target_mapping, vn,
                        list(vn.links.items()),  # default vlink order
                    )
                    if paths:
                        active_offline[vnr.id] = {
                            "mapping": target_mapping, "vn": vn, "paths": paths,
                            "expire": vnr.arrival_time + vnr.lifetime,
                        }
                except ValueError:
                    pass
            else:
                loss = None
        else:
            # ---- ONLINE EXPERT MODE (original) ----
            # 1. Release expired explicitly (using expert's lifecycle tracker)
            #    so substrate state matches what policy sees pre-commit.
            if getattr(expert, "global_controller", None) is not None:
                expert._release_expired(vnr.arrival_time)
                algo.global_controller.clear_caches()

            # 2. Policy forward pass on PRE-commit state — cache scores.
            cand_scores_cached, cand_pools_cached = policy_forward(algo, vn)
            vnodes_in_order = list(vn.nodes.values())

            # 3. Expert solves (commits). The internal _release_expired is a no-op
            #    since we already released above.
            sol = expert.solve(substrate, vnr)
            expert_total += 1
            if sol.is_successful:
                expert_succ += 1
                # 4. Loss from cached scores + expert mapping.
                loss, n_targets, n_matched = supervised_loss_from_cached(
                    cand_scores_cached, cand_pools_cached, vnodes_in_order, sol.node_mapping,
                )
                targets_total += n_targets
                matched_total += n_matched
            else:
                loss = None

        # ---- Both modes accumulate loss for gradient batching ----
        if loss is not None:
            accum_loss = accum_loss + loss
            accum_count += 1

        # Backprop every batch_size VNRs (whether expert succeeded or not).
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
                recent_avg = sum(batch_losses[-args.print_every:]) / max(len(batch_losses[-args.print_every:]), 1)
                e_rate = expert_succ / max(expert_total, 1)
                m_rate = matched_total / max(targets_total, 1)
                print(
                    f"  ep {ep+1:5d}/{len(vnrs)}  batch {batch_idx:4d}  "
                    f"loss={recent_avg:.4f}  "
                    f"expert_succ={e_rate:.1%} ({expert_succ}/{expert_total})  "
                    f"match={m_rate:.1%}  "
                    f"elapsed={fmt_secs(elapsed)} eta={fmt_secs(eta)}"
                )
                with open(log_path, "a") as f:
                    f.write(f"{time.time()},{batch_idx},{recent_avg:.6f},{e_rate:.4f},{m_rate:.4f}\n")

    # Final flush
    if accum_count > 0:
        avg = accum_loss / accum_count
        avg.backward()
        torch.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_norm=5.0)
        optimizer.step()

    torch.save({
        "policy_state_dict": algo.policy.state_dict(),
        "config": algo.config,
        "episodes": len(vnrs),
        "expert": "mp_vne",
    }, ckpt_path)

    elapsed = time.time() - start
    print("-" * 72)
    print(f"Imitation done in {fmt_secs(elapsed)}.")
    print(f"  expert success rate: {expert_succ}/{expert_total} = {expert_succ/max(expert_total,1):.1%}")
    print(f"  target match rate:   {matched_total}/{targets_total} = {matched_total/max(targets_total,1):.1%}")
    print(f"  final batch loss avg: {sum(batch_losses[-50:])/max(len(batch_losses[-50:]),1):.4f}")
    print(f"  checkpoint saved to {ckpt_path}")
    print(f"  log saved to        {log_path}")


if __name__ == "__main__":
    main()
