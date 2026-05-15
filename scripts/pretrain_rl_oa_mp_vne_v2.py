#!/usr/bin/env python3
"""Standalone pretrain driver for rl_oa_mp_vne_v2.

Differs from the inline pretrain in `RLOAMPVNE._pretrain` in two ways:
  1. Substrate **topology** is fixed (loaded from disk), but **resources**
     (available_cpu / available_bw) are randomized per episode via
     fractional drop, so the policy sees a distribution of load states.
  2. Saves a torch checkpoint and a separate CSV so pretrain quality can
     be evaluated without entanglement with online learning.

V2 is an isolated copy of the v1 algorithm — edit `algorithms/rl_oa_mp_vne_v2/`
without affecting the original `rl_oa_mp_vne` baseline.
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

from algorithms.rl_oa_mp_vne_v2.rl_oa_mp_vne import RLOAMPVNE
from algorithms.rl_oa_mp_vne_v2.vn_generator import generate_random_vn
from utils.load_dataset import read_substrate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fractional_drop(global_controller, u_max_cpu: float, u_max_bw: float) -> None:
    """Reset, then drop a random fraction of CPU/BW per node/link.

    available = capacity * (1 - u),  u ~ U[0, u_max].
    Topology stays untouched.
    """
    global_controller.reset_allocations()
    global_controller.clear_caches()
    for lc in global_controller.local_controllers:
        for node in lc.domain.network.nodes.values():
            u = random.uniform(0.0, u_max_cpu)
            node.available_cpu = node.cpu_capacity * (1.0 - u)
        for link in lc.domain.network.links.values():
            u = random.uniform(0.0, u_max_bw)
            link.available_bw = link.bandwidth_capacity * (1.0 - u)


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:4.0f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m:2d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


def main():
    p = argparse.ArgumentParser(description="Pretrain rl_oa_mp_vne_v2 with randomized substrate load.")
    p.add_argument("--substrate", required=True, help="Path to substrate JSON")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override training.batch_size from config")
    p.add_argument("--u-max-cpu", type=float, default=0.7,
                   help="Max fraction of CPU dropped per node (default 0.7)")
    p.add_argument("--u-max-bw", type=float, default=0.7,
                   help="Max fraction of BW dropped per link (default 0.7)")
    p.add_argument("--checkpoint", default="checkpoints/rl_oa_mp_vne_v2_pretrain.pt")
    p.add_argument("--log-file", default="logs/rl_oa_mp_vne_v2_pretrain.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=10,
                   help="Print every N batches (default 10)")
    args = p.parse_args()

    set_seed(args.seed)

    print(f"Loading substrate from {args.substrate}")
    substrate = read_substrate(args.substrate)

    algo = RLOAMPVNE()
    algo._init_controller(substrate)
    train_cfg = algo.config.get("training", {})
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))

    # Redirect trainer log to a separate file (don't mix with online-learning log).
    # Overwrite each run so successive pretrains don't pile data on top of each other.
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    algo.trainer.loss_log_file = str(log_path)
    with open(log_path, "w") as f:
        f.write("timestamp,avg_reward,total_loss,node_loss,link_loss,cand_loss,"
                "critic_loss,value_mean,entropy,adv_std,entropy_coef,lr\n")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pretrain rl_oa_mp_vne_v2: {args.episodes} episodes, batch_size={batch_size}")
    print(f"  load drop range: CPU U[0,{args.u_max_cpu}]  BW U[0,{args.u_max_bw}]")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  log: {log_path}")
    print("-" * 72)

    algo.policy.train()
    start = time.time()

    head_rewards, tail_rewards = [], []
    head_n = tail_n = 100
    recent_rewards = []   # last-batch raw rewards for logging

    for ep in range(args.episodes):
        # 1. Sample a random load state on the fixed topology.
        fractional_drop(algo.global_controller, args.u_max_cpu, args.u_max_bw)

        # 2. Sample a random virtual network.
        vn = generate_random_vn(
            min_nodes=train_cfg.get("vn_min_nodes", 2),
            max_nodes=train_cfg.get("vn_max_nodes", 8),
            min_cpu=train_cfg.get("vn_min_cpu", 1.0),
            max_cpu=train_cfg.get("vn_max_cpu", 30.0),
            min_bw=train_cfg.get("vn_min_bw", 5.0),
            max_bw=train_cfg.get("vn_max_bw", 80.0),
            link_prob=train_cfg.get("vn_link_prob", 0.5),
        )

        # 3. V2: direct autoregressive sampling with collision masking (no PSO).
        ordered_vnodes, ordered_links, mapping, log_probs, entropies, value = \
            algo.rank_direct(vn, sample=True)
        reward = algo._try_embedding_direct(vn, ordered_links, mapping)
        algo.trainer.record(log_probs, value, entropies, reward)
        recent_rewards.append(reward)

        # Track raw sample reward for head/tail stats (interpretable scale).
        if ep < head_n:
            head_rewards.append(reward)
        if ep >= args.episodes - tail_n:
            tail_rewards.append(reward)

        # 4. Periodic policy update.
        if (ep + 1) % batch_size == 0:
            loss_dict = algo.trainer.update()
            batch_idx = (ep + 1) // batch_size
            if batch_idx % args.print_every == 0:
                elapsed = time.time() - start
                eta = elapsed * (args.episodes - ep - 1) / max(ep + 1, 1)
                r_mean = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"  ep {ep+1:5d}/{args.episodes}  "
                    f"r={r_mean:+.3f} V={loss_dict['value_mean']:+.3f} "
                    f"crit={loss_dict['critic_loss']:.3f} "
                    f"ent={loss_dict['entropy']:.2f} "
                    f"adv_std={loss_dict['adv_std']:.3f}  "
                    f"loss={loss_dict['total_loss']:+.3f}  "
                    f"(n:{loss_dict['node_loss']:+.3f} l:{loss_dict['link_loss']:+.3f} "
                    f"c:{loss_dict['cand_loss']:+.3f})  "
                    f"elapsed={fmt_secs(elapsed)} eta={fmt_secs(eta)}"
                )
            recent_rewards.clear()

    # Drain remaining buffer.
    if algo.trainer.buffer:
        algo.trainer.update()

    torch.save({
        "policy_state_dict": algo.policy.state_dict(),
        "config": algo.config,
        "episodes": args.episodes,
        "u_max_cpu": args.u_max_cpu,
        "u_max_bw": args.u_max_bw,
        "seed": args.seed,
    }, ckpt_path)

    elapsed = time.time() - start
    print("-" * 72)
    print(f"Pretrain done in {fmt_secs(elapsed)}.")
    if head_rewards and tail_rewards:
        h = sum(head_rewards) / len(head_rewards)
        t = sum(tail_rewards) / len(tail_rewards)
        print(f"  head-{len(head_rewards)} avg reward: {h:+.4f}")
        print(f"  tail-{len(tail_rewards)} avg reward: {t:+.4f}")
        print(f"  delta:                  {t - h:+.4f}")
    print(f"  checkpoint saved to {ckpt_path}")
    print(f"  loss log saved to    {log_path}")


if __name__ == "__main__":
    main()
