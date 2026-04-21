#!/usr/bin/env python3
"""Offline training driver for rl_cand_vne."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure the project root is on the path when the script is run directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE, substrate_structure_hash
from algorithms.rl_cand_vne.state_sampler import sample_substrate_state
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains
from algorithms.oa_mp_vne.global_controller import GlobalController
from problem.request import VirtualNetworkRequest
from utils.load_dataset import read_substrate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--config", default="configs/rl_cand_vne.yaml")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--checkpoint", default="checkpoints/rl_cand_vne.pt")
    p.add_argument("--log-dir", default="logs/rl_cand_vne/")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = Path(args.log_dir) / "train.jsonl"

    sn = read_substrate(args.substrate)  # returns MultiDomainNetwork

    algo = RLCandVNE()
    algo.config = cfg
    algo.global_controller = GlobalController(sn)
    algo._baseline_helper.global_controller = algo.global_controller
    algo._pretrained = True  # skip inline pretrain; we ARE the offline trainer

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    ckpt_every = int(train_cfg.get("checkpoint_every", 500))
    vn_kwargs = {
        "min_nodes": train_cfg["vn_min_nodes"], "max_nodes": train_cfg["vn_max_nodes"],
        "min_cpu": train_cfg["vn_min_cpu"], "max_cpu": train_cfg["vn_max_cpu"],
        "min_bw": train_cfg["vn_min_bw"], "max_bw": train_cfg["vn_max_bw"],
        "link_prob": train_cfg["vn_link_prob"],
    }
    ad = train_cfg["allowed_domains"]
    domain_ids = [lc.domain.id for lc in algo.global_controller.local_controllers]
    sub_hash = substrate_structure_hash(sn)

    algo.policy.train()

    first_100_reward, last_100_reward = [], []
    first_100_cpr, last_100_cpr = [], []
    first_100_sr, last_100_sr = [], []

    with open(log_path, "w", buffering=1) as log_f:  # line-buffered
        batch_rewards = []

        for ep in range(args.episodes):
            sample_substrate_state(
                algo.global_controller, sn,
                warmup_fraction=train_cfg["warmup_fraction"],
                u_max_cpu=train_cfg["u_max_cpu"], u_max_bw=train_cfg["u_max_bw"],
                M_max=train_cfg["warmup_M_max"], vn_kwargs=vn_kwargs,
            )
            vn = generate_random_vn_with_domains(
                min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
                min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
                min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
                link_prob=vn_kwargs["link_prob"],
                domain_ids=domain_ids,
                p_all=ad["p_all"], p_single=ad["p_single"], p_subset=ad["p_subset"],
                subset_min=ad["subset_min"], subset_max=ad["subset_max"],
            )
            req = VirtualNetworkRequest(
                id=f"pt_{ep}", virtual_network=vn,
                arrival_time=0.0, lifetime=float("inf"),
            )
            reward, committed, dom_lps, sn_lps, success = algo._training_episode(req, persist=False)
            algo.trainer.record(
                domain_log_probs=dom_lps, snode_log_probs_per_vnode=sn_lps,
                reward=reward, committed_snode_indices=committed, success=success,
            )
            batch_rewards.append(reward)
            algo.global_controller.reset_allocations()
            algo.global_controller.clear_caches()

            # Track head/tail reward + cost/revenue + success for convergence report.
            cpr = -reward if success else float("nan")
            if ep < 100:
                first_100_reward.append(reward)
                first_100_cpr.append(cpr)
                first_100_sr.append(float(success))
            if ep >= args.episodes - 100:
                last_100_reward.append(reward)
                last_100_cpr.append(cpr)
                last_100_sr.append(float(success))

            if (ep + 1) % batch_size == 0:
                m = algo.trainer.update()
                log_line = {
                    "episode": ep + 1,
                    "loss_total": m["loss_total"],
                    "loss_rl": m["loss_rl"],
                    "loss_sup": m["loss_sup"],
                    "reward_mean": m["avg_reward"],
                    "reward_min": min(batch_rewards),
                    "reward_max": max(batch_rewards),
                    "success_rate": m["success_rate"],
                    "cost_per_revenue_mean": -m["avg_reward"],
                    "baseline": m["baseline"],
                    "lr": train_cfg["learning_rate"],
                    "timestamp": time.time(),
                }
                log_f.write(json.dumps(log_line) + "\n")
                batch_rewards.clear()

            if (ep + 1) % ckpt_every == 0:
                algo.save_checkpoint(args.checkpoint, substrate_hash=sub_hash)
                algo._episodes_trained = ep + 1

        if algo.trainer.buffer:
            algo.trainer.update()
        algo.save_checkpoint(args.checkpoint, substrate_hash=sub_hash)
        algo._episodes_trained = args.episodes

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return sum(xs) / len(xs) if xs else float("nan")

    r_first = _mean(first_100_reward)
    r_last = _mean(last_100_reward)
    cpr_first = _mean(first_100_cpr)
    cpr_last = _mean(last_100_cpr)
    sr_first = _mean(first_100_sr)
    sr_last = _mean(last_100_sr)

    print("=" * 60)
    print("TRAINING SUMMARY")
    print(f"  reward       first-100={r_first:.4f}  last-100={r_last:.4f}")
    print(f"  cost/rev     first-100={cpr_first:.4f}  last-100={cpr_last:.4f}")
    print(f"  success_rate first-100={sr_first:.4f}  last-100={sr_last:.4f}")
    # Rewards are negative (−cost/revenue). Convergence = last-100 mean strictly greater
    # than first-100 mean by more than one stddev of first-100 (improvement signal that
    # also tolerates noise).
    import statistics as _stats
    if first_100_reward:
        first_std = _stats.pstdev(first_100_reward) or 1e-6
        converged = (r_last - r_first) > first_std
    else:
        converged = False
    print(f"  {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
