"""V15 — RL fine-tune from V6 with REINFORCE + KL penalty + entropy bonus.

Goal: break the IL ceiling (model can only mimic mp_vne). Use direct cost
feedback as gradient signal.

Design:
  - Trainable policy π (init from V6 ckpt)
  - Frozen reference policy π_ref (V6 ckpt, never updated)
  - For each VNR:
      1. Sample candidates from π (Plackett-Luce → log_probs in grad path)
      2. Run PSO on sampled candidates → mapping + actual_cost
      3. Reward:
           if success: r = (baseline_cost - actual_cost) / baseline_cost
           if fail:    r = -1.0
         where baseline_cost = running EMA of recent successful costs
      4. Advantage = r - V_baseline (running average)
      5. Loss = -Σ log_P × advantage
              + β_KL × KL(π || π_ref)        ← keep close to V6
              - β_H × entropy                 ← maintain exploration
      6. Commit NN+PSO mapping persistently (true on-policy state evolution)

KL penalty prevents the policy collapsing under noisy rewards (this is what
killed prior pure-RL attempts on this codebase). Reference = V6 = strong
warm-start, so π stays in a sane neighborhood.
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

from algorithms.carl_vne.il_mp_vne_v6 import ILMPVNEV6
from algorithms.carl_vne.il_mp_vne_v17 import ILMPVNEV17
from algorithms.carl_vne.carl_vne import CARLVNE
from utils.load_dataset import read_substrate, read_virtual_requests

ALGO_CLASSES = {
    "il_mp_vne_v6": ILMPVNEV6,
    "il_mp_vne_v17": ILMPVNEV17,
    "carl_vne": CARLVNE,
    "il_mp_vne_v19": CARLVNE,  # backwards-compatible alias
}


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


def compute_kl(cand_scores_new, cand_scores_ref):
    """KL(π_new || π_ref) averaged over vnodes, summed over snodes."""
    kls = []
    for s_new, s_ref in zip(cand_scores_new, cand_scores_ref):
        log_p = F.log_softmax(s_new, dim=0)
        p = log_p.exp()
        log_q = F.log_softmax(s_ref.detach(), dim=0)
        kl = (p * (log_p - log_q)).sum()
        kls.append(kl)
    return torch.stack(kls).mean() if kls else torch.tensor(0.0)


# ===== True PPO helpers =====
#
# Substrate state snapshot/restore so that PPO REPLAY (re-forwarding the policy
# on the same VN with stored action choices) runs on the SAME substrate state
# the actions were sampled under. Without this, ratio = π_new / π_old is
# computed across DIFFERENT state distributions and the IS correction is
# invalid.
def snapshot_substrate(algo):
    """Cheap deep snapshot of mutable substrate resources + active mappings.
    Cost ~150 floats per call (50-node × ~3 numbers + ~100 links × 1)."""
    from collections import OrderedDict
    snap = {"nodes_cpu": {}, "links_bw": {},
            "active_mappings": OrderedDict(algo._active_mappings)}
    for lc in algo.global_controller.local_controllers:
        for snode in lc.domain.network.nodes.values():
            snap["nodes_cpu"][snode.id] = getattr(snode, "available_cpu",
                                                  snode.cpu_capacity)
        for key, link in lc.domain.network.links.items():
            snap["links_bw"][key] = getattr(link, "available_bw",
                                            link.bandwidth_capacity)
    for key, link in algo.global_controller.snetwork.inter_domain_links.items():
        snap["links_bw"][key] = getattr(link, "available_bw",
                                        link.bandwidth_capacity)
    return snap


def restore_substrate(algo, snap):
    """Re-apply a snapshot — main algo and ref_algo share substrate, so this
    restores the world for both."""
    from collections import OrderedDict
    for lc in algo.global_controller.local_controllers:
        for snode in lc.domain.network.nodes.values():
            if snode.id in snap["nodes_cpu"]:
                snode.available_cpu = snap["nodes_cpu"][snode.id]
        for key, link in lc.domain.network.links.items():
            if key in snap["links_bw"]:
                link.available_bw = snap["links_bw"][key]
    for key, link in algo.global_controller.snetwork.inter_domain_links.items():
        if key in snap["links_bw"]:
            link.available_bw = snap["links_bw"][key]
    algo._active_mappings = OrderedDict(snap["active_mappings"])


def replay_log_p_direct(algo, vn, mapping):
    """Forward policy on `vn` under CURRENT substrate state and compute log_p
    of the action sequence stored in `mapping` (vnode_id → snode_id).

    Mirrors `rank_direct(sample_cand=True)` logic but instead of sampling,
    computes Categorical.log_prob(stored_action) at each step. Mask collisions
    accumulate exactly the same way as the original rollout because we replay
    the same vnode order.

    Returns (log_p_sum, entropy_sum, value) or None if any vnode's stored
    snode is no longer in its candidate pool (substrate evolution changed
    feasibility) — caller should skip the sample.
    """
    node_scores, link_scores, cand_scores, cand_pools, value = algo._forward_policy(vn)
    vnodes = list(vn.nodes.values())
    # Frozen ordering (target=cand): _greedy_sort is deterministic given node_scores.
    ordered_vnodes = algo._greedy_sort(node_scores, vnodes)
    orig_idx = {v.id: i for i, v in enumerate(vnodes)}
    used_snode_ids = set()
    lps, ents = [], []

    for v in ordered_vnodes:
        target_snode_id = mapping.get(v.id) if mapping else None
        if target_snode_id is None:
            return None
        i = orig_idx[v.id]
        scores_i = cand_scores[i].clone()
        pool_i = cand_pools[i]
        if used_snode_ids:
            mask_vals = torch.tensor(
                [float("-inf") if pool_i[j].id in used_snode_ids else 0.0
                 for j in range(len(pool_i))],
                dtype=scores_i.dtype,
            )
            scores_i = scores_i + mask_vals
        try:
            pos = next(j for j in range(len(pool_i))
                       if pool_i[j].id == target_snode_id)
        except StopIteration:
            return None
        if not torch.isfinite(scores_i[pos]):
            return None
        dist = torch.distributions.Categorical(logits=scores_i)
        lps.append(dist.log_prob(torch.tensor(pos)))
        ents.append(dist.entropy())
        used_snode_ids.add(target_snode_id)

    if not lps:
        return None
    return torch.stack(lps).sum(), torch.stack(ents).sum(), value


def replay_cand_scores(algo, vn):
    """Forward policy on `vn` and return ONLY the raw cand_scores list (for
    KL computation). Cheap — same forward pass replay_log_p_direct does, but
    we keep it separate so callers can choose what they need."""
    _, _, cand_scores, _, _ = algo._forward_policy(vn)
    return cand_scores


def ppo_batch_update(algo, ref_algo, ppo_buf, optimizer, trainable_params,
                     args, use_kl):
    """One PPO batch update consisting of K gradient epochs over `ppo_buf`.

    For each PPO epoch and each VNR in the buffer:
      1. Restore substrate to the snapshot taken at rollout time.
      2. Re-forward policy on vn → new_log_p, new_value, new_entropy (with grad).
      3. ratio = exp(new_log_p − old_log_p.detach()).
      4. Clipped surrogate: L_pol = -min(ratio·A, clip(ratio,1±ε)·A).
      5. Add value MSE + entropy bonus + (optional) KL anchor.
      6. backward + step.

    Returns dict with last-epoch scalars for logging.
    """
    if not ppo_buf:
        return None

    # Save current substrate so we can return to S_N after the updates.
    snap_post = snapshot_substrate(algo)

    last = None
    for k in range(args.ppo_epochs):
        new_lps, old_lps, rewards_list = [], [], []
        new_values, new_ents, kls = [], [], []

        for item in ppo_buf:
            if not item["mapping"]:
                continue  # nothing to replay (mapping is None / empty)

            restore_substrate(algo, item["snap_pre"])
            algo.global_controller.clear_caches()
            if ref_algo is not None:
                ref_algo.global_controller.clear_caches()

            res = replay_log_p_direct(algo, item["vn"], item["mapping"])
            if res is None:
                continue
            new_lp, new_ent, new_v = res

            new_lps.append(new_lp)
            old_lps.append(item["old_log_p_sum"])
            rewards_list.append(item["reward"])
            new_values.append(new_v)
            new_ents.append(new_ent)

            if use_kl:
                with torch.no_grad():
                    ref_cs = replay_cand_scores(ref_algo, item["vn"])
                new_cs = replay_cand_scores(algo, item["vn"])
                kls.append(compute_kl(new_cs, ref_cs))

        if not new_lps:
            continue

        new_lps_t = torch.stack(new_lps)
        old_lps_t = torch.stack(old_lps)
        rewards_t = torch.tensor(rewards_list, dtype=torch.float32)
        values_t = torch.stack(new_values).reshape(-1)
        ents_t = torch.stack(new_ents)

        if args.use_critic:
            advantages = rewards_t - values_t.detach()
            value_loss = F.mse_loss(values_t, rewards_t)
        else:
            advantages = rewards_t - rewards_t.mean()
            value_loss = torch.tensor(0.0)

        if args.normalize_adv and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratios = torch.exp(new_lps_t - old_lps_t)
        s1 = ratios * advantages
        s2 = torch.clamp(ratios, 1.0 - args.ppo_clip,
                         1.0 + args.ppo_clip) * advantages
        policy_loss = -torch.min(s1, s2).mean()
        entropy_term = ents_t.mean()
        kl_term = (torch.stack(kls).mean() if kls else torch.tensor(0.0))

        loss = (policy_loss
                + args.value_coef * value_loss
                - args.beta_entropy * entropy_term
                + args.beta_kl * kl_term)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()

        last = {
            "loss": loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "kl": kl_term.detach().item(),
            "entropy": entropy_term.detach().item(),
            "value_loss": (value_loss.detach().item()
                           if args.use_critic else 0.0),
            "ratio_mean": ratios.detach().mean().item(),
            "ratio_clip_frac": ((ratios.detach() < 1 - args.ppo_clip)
                                | (ratios.detach() > 1 + args.ppo_clip)
                                ).float().mean().item(),
            "reward_mean": rewards_t.mean().item(),
            "advantage_mean": advantages.detach().mean().item(),
        }

    # Restore to S_N so the next batch picks up where we left off.
    restore_substrate(algo, snap_post)
    algo.global_controller.clear_caches()
    if ref_algo is not None:
        ref_algo.global_controller.clear_caches()
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--targets-file", default=None,
                   help="Optional: strong_targets file. Used only to log "
                        "comparison stats vs expert; NOT used as label.")
    p.add_argument("--ref-checkpoint", required=True,
                   help="V6 checkpoint. Used for both init AND frozen ref.")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Lower than IL (1e-3) — RL needs smaller steps.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--beta-kl", type=float, default=0.1,
                   help="KL weight: keep π close to π_ref (V6).")
    p.add_argument("--beta-entropy", type=float, default=0.01,
                   help="Entropy bonus: prevent premature commitment.")
    p.add_argument("--algorithm", default="il_mp_vne_v17",
                   choices=list(ALGO_CLASSES.keys()),
                   help="Policy architecture / inference regime. V17 = top-1 per "
                        "domain (matches V17 eval).")
    p.add_argument("--target", default="cand", choices=["cand", "ordering"],
                   help="Which head to train. cand: train cand_head + encoder "
                        "(pick good candidates), freeze ordering, KL-anchor cand. "
                        "ordering: train node/link heads, freeze encoder+cand.")
    p.add_argument("--rollout", default="direct", choices=["direct", "pso"],
                   help="direct: autoregressive snode selection (rank_direct) — "
                        "trains cand_head to PICK good candidates with real "
                        "exploration (sample 1 snode/vnode). Freezes ordering "
                        "heads, anchors cand to R2 via KL. Deploy with "
                        "il_mp_vne_v17_direct. pso: ordering-RL via rank_all_nn "
                        "+ PSO (freezes encoder+cand).")
    p.add_argument("--success-bonus", type=float, default=1.0,
                   help="Fixed positive reward (α) for any successful embedding.")
    p.add_argument("--cost-lambda", type=float, default=0.3,
                   help="(reward-mode=cost_rel) Weight on relative cost improvement.")
    p.add_argument("--reward-mode", default="cost_rel",
                   choices=["cost_rel", "rev_cost"],
                   help="cost_rel (V19/V21 default): reward_success = α + λ·"
                        "(cost_EMA−cost)/cost_EMA. Size-DEPENDENT — can give "
                        "reward_success < reward_fail for big VNRs. "
                        "rev_cost (V22): reward_success = α + β·clip("
                        "(ratio−ratio_EMA)/ratio_EMA, −clip, +clip), where "
                        "ratio = vn_revenue / cost. Size-INVARIANT, bounded — "
                        "optimises acceptance AND rev/cost together.")
    p.add_argument("--efficiency-weight", type=float, default=0.5,
                   help="(reward-mode=rev_cost) β — weight on rev/cost rel "
                        "efficiency. α=1.0 + β·rel ∈ [α−β, α+β] for success.")
    p.add_argument("--ratio-ema-decay", type=float, default=0.95,
                   help="(reward-mode=rev_cost) EMA decay for rev/cost ratio "
                        "baseline (effective window ≈ 1/(1−decay) ≈ 20 eps).")
    p.add_argument("--efficiency-clip", type=float, default=1.0,
                   help="(reward-mode=rev_cost) clip range for relative "
                        "efficiency. reward_success ∈ [α−β·clip, α+β·clip].")
    p.add_argument("--filter-impossible", action="store_true", default=False,
                   help="Skip VNRs where total CPU demand > total available "
                        "CPU slack (provably infeasible). Reduces fail-reward "
                        "noise on congested substrates (e.g. late 200-node).")
    p.add_argument("--progress-fail-reward", action="store_true", default=False,
                   help="On partial-trajectory fail (mapping incomplete), scale "
                        "fail_reward by (1 − progress). progress = num vnodes "
                        "mapped / total. Gives partial credit so policy isn't "
                        "uniformly penalised for failures it couldn't prevent.")
    p.add_argument("--epochs", type=int, default=1,
                   help="Number of training epochs. Each epoch RESETS the "
                        "substrate (reload from JSON, clear active mappings) "
                        "so per-epoch metrics are comparable. Default 1 = "
                        "single-pass legacy behaviour.")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Global index of the first epoch of this run (for "
                        "continuation runs: 20 epochs trained before => 21). "
                        "Default: auto-derived from the last row of the "
                        "existing epoch summary CSV (last epoch + 1), else 1. "
                        "CSV logs are appended, never overwritten.")
    p.add_argument("--train-ordering", action="store_true", default=True,
                   help="Include node/link ordering log-probs in the policy "
                        "gradient (they are random after IL pretrain — the main "
                        "acceptance lever under K=1).")
    p.add_argument("--freeze-shared", action="store_true", default=True,
                   help="Freeze substrate_enc + vn_gcn + vnode_attn + candidate_head "
                        "(the IL-trained shared encoder & cand head). Only the "
                        "node/link/value heads train. Prevents ordering gradients "
                        "from corrupting the shared features cand_head relies on "
                        "(root cause of the KL blow-up & success drop).")
    p.add_argument("--no-freeze-shared", dest="freeze_shared", action="store_false")
    p.add_argument("--use-critic", action="store_true", default=True,
                   help="Use the value_head V(s) as a learned, state-dependent "
                        "baseline (advantage = reward − V(s)) instead of a global "
                        "reward EMA. Adds MSE value loss. Lower variance.")
    p.add_argument("--no-use-critic", dest="use_critic", action="store_false")
    p.add_argument("--normalize-adv", action="store_true", default=True,
                   help="Standardize advantages within each batch (mean 0, std 1). "
                        "Variance reduction.")
    p.add_argument("--value-coef", type=float, default=0.5,
                   help="Weight on the critic MSE value loss.")
    p.add_argument("--baseline-ema", type=float, default=0.99,
                   help="EMA decay for running reward baseline.")
    p.add_argument("--success-cost-ema", type=float, default=0.95,
                   help="EMA decay for running successful-cost baseline.")
    p.add_argument("--fail-reward", type=float, default=-1.0)
    # ---- True PPO (clipped surrogate) ----
    p.add_argument("--ppo-mode", choices=["reinforce", "ppo"], default="reinforce",
                   help="reinforce (default): current REINFORCE+critic+KL+entropy. "
                        "ppo: TRUE PPO. Per-batch: snapshot substrate at each VNR "
                        "rollout; after batch collected, restore each snapshot and "
                        "REPLAY the policy on the same VN with stored action choices "
                        "→ new_log_p (with grad). ratio = exp(new_log_p − old_log_p) "
                        "with old_log_p detached at rollout. Apply CLIPPED SURROGATE: "
                        "L = -min(ratio·A, clip(ratio,1±ε)·A). Run K gradient epochs "
                        "over the same batch (off-policy correction via importance "
                        "ratio). Only valid for --rollout direct + --target cand.")
    p.add_argument("--ppo-clip", type=float, default=0.2,
                   help="(ppo-mode=ppo) Clip epsilon ε for ratio. PPO paper uses 0.2.")
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="(ppo-mode=ppo) Number of gradient epochs per batch. K=1 is "
                        "equivalent to REINFORCE since ratio=1 always. K≥2 gives real "
                        "PPO behaviour. Each extra epoch ~ 1× rollout cost.")
    p.add_argument("--checkpoint", default="checkpoints/il_mp_vne_v17_ppo_100nodes.pt")
    p.add_argument("--log-file", default="logs/ppo_v17.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    set_seed(args.seed)
    print(f"Loading substrate from {args.substrate}")
    substrate = read_substrate(args.substrate)
    print(f"Loading VNRs from {args.requests}")
    vnrs = read_virtual_requests(args.requests)
    if args.episodes > 0:
        vnrs = vnrs[:args.episodes]
    print(f"  Using {len(vnrs)} VNRs")

    expert_costs = {}
    if args.targets_file:
        with open(args.targets_file) as f:
            for e in json.load(f):
                if e.get("successful"):
                    expert_costs[e["vnr_id"]] = e["cost"]
        print(f"  Loaded {len(expert_costs)} expert costs for logging")

    algo_cls = ALGO_CLASSES[args.algorithm]

    # Trainable policy.
    algo = algo_cls()
    algo._init_controller(substrate)
    ckpt = torch.load(args.ref_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["policy_state_dict"] if isinstance(ckpt, dict) else ckpt
    algo.policy.load_state_dict(state, strict=False)
    print(f"  Trainable policy ({args.algorithm}) loaded from {args.ref_checkpoint}")

    # Freeze modules that are NOT the training target (independent of rollout).
    if args.target == "cand":
        # Train cand_head + shared encoder + value; freeze the ordering heads
        # (random after IL → would inject noise). KL anchors cand to the ref.
        freeze_mods = ("node_head", "link_head")
        target_desc = "cand_head + encoder + value (ordering frozen)"
        use_kl = args.beta_kl > 0
    else:  # ordering
        freeze_mods = ("substrate_enc", "vn_gcn", "vnode_attn", "candidate_head")
        target_desc = "node/link/value (encoder+cand frozen)"
        use_kl = False

    for mod_name in freeze_mods:
        mod = getattr(algo.policy, mod_name, None)
        if mod is None:
            # V21 etc. may not have all the named heads; skip silently.
            continue
        for param in mod.parameters():
            param.requires_grad_(False)
    n_train = sum(p.numel() for p in algo.policy.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in algo.policy.parameters() if not p.requires_grad)
    print(f"  rollout={args.rollout} → train: {target_desc} "
          f"({n_frozen} frozen, {n_train} trainable)")

    trainable_params = [p for p in algo.policy.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Frozen reference policy — used for the KL anchor on cand output.
    ref_algo = None
    if use_kl:
        ref_algo = algo_cls()
        ref_algo._init_controller(substrate)
        ref_algo.policy.load_state_dict(state, strict=False)
        for param in ref_algo.policy.parameters():
            param.requires_grad_(False)
        ref_algo.policy.eval()
        print(f"  Frozen reference policy loaded (KL enabled, β_KL={args.beta_kl})")
    else:
        print(f"  KL disabled (freeze_shared={args.freeze_shared}) — no reference forward")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_has_data = log_path.exists() and log_path.stat().st_size > 0
    log_f = open(log_path, "a")
    if not log_has_data:
        log_f.write("epoch,episode,batch,loss,policy_loss,value_loss,kl,entropy,"
                    "avg_reward,advantage_mean,ratio_mean,ratio_clip_frac,"
                    "succ_rate,elapsed_s\n")

    # Per-epoch summary CSV (one row per epoch, suitable for learning curves).
    epoch_log_path = log_path.with_name(log_path.stem + "_epoch_summary.csv")
    epoch_has_data = epoch_log_path.exists() and epoch_log_path.stat().st_size > 0
    epoch_f = open(epoch_log_path, "a")
    if not epoch_has_data:
        epoch_f.write("epoch,n_episodes,n_success,succ_rate,mean_loss,mean_policy_loss,"
                      "mean_value_loss,mean_kl,mean_entropy,mean_reward,"
                      "mean_advantage,mean_ratio,mean_clip_frac,elapsed_s\n")

    # Global epoch numbering: a continuation run appends to the same CSVs and
    # continues the epoch count instead of restarting at 1.
    start_epoch = args.start_epoch
    if start_epoch is None:
        start_epoch = 1
        if epoch_has_data:
            with open(epoch_log_path) as f:
                data_rows = [r for r in f.read().splitlines()
                             if r and not r.startswith("epoch,")]
            if data_rows:
                start_epoch = int(float(data_rows[-1].split(",")[0])) + 1
    if start_epoch > 1:
        print(f"  Continuation: global epochs {start_epoch}.."
              f"{start_epoch + args.epochs - 1} (appending to {epoch_log_path})")

    crit = "critic V(s)" if args.use_critic else "reward EMA"
    print(f"PPO fine-tune: {len(vnrs)} VNRs, batch={args.batch_size}, lr={args.lr}")
    if args.reward_mode == "rev_cost":
        reward_desc = (f"rev_cost α={args.success_bonus} β={args.efficiency_weight} "
                       f"clip=±{args.efficiency_clip} ema_decay={args.ratio_ema_decay}")
    else:
        reward_desc = (f"cost_rel α={args.success_bonus} λ={args.cost_lambda}")
    print(f"  baseline={crit}, normalize_adv={args.normalize_adv}, "
          f"β_H={args.beta_entropy}, value_coef={args.value_coef}, "
          f"reward={reward_desc}, fail_r={args.fail_reward}")
    print(f"  checkpoint: {args.checkpoint}")
    print("-" * 72)

    start = time.time()
    batch_idx = 0
    n_success = 0
    n_total = 0
    n_better_than_expert = 0
    n_expert_compared = 0
    reward_ema = 0.0
    success_cost_ema = None
    ratio_ema = None
    losses_recent = []
    buf = []
    ppo_buf = []  # used only when args.ppo_mode == "ppo"
    if args.ppo_mode == "ppo":
        if args.rollout != "direct" or args.target != "cand":
            raise ValueError(
                "--ppo-mode ppo only supports --rollout direct + --target cand "
                "(replay machinery is built around rank_direct's cand sampling)."
            )
        print(f"  PPO mode: clipped surrogate ε={args.ppo_clip}, "
              f"K={args.ppo_epochs} grad epochs/batch, "
              f"substrate snapshot+restore per VNR.")
    # Per-epoch accumulators (reset each epoch).
    epoch_idx = 0
    ep_n_total = 0
    ep_n_success = 0
    ep_loss_sum = 0.0
    ep_loss_count = 0
    ep_policy_sum = 0.0
    ep_kl_sum = 0.0
    ep_ent_sum = 0.0
    ep_vloss_sum = 0.0
    ep_advantage_sum = 0.0
    ep_ratio_sum = 0.0
    ep_clip_sum = 0.0
    ep_reward_sum = 0.0
    ep_reward_count = 0
    ep_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_idx = start_epoch + epoch
        if args.epochs > 1 and epoch > 0:
            # Reset substrate to fresh state at the start of each epoch (except
            # the first which already loaded fresh substrate).
            print(f"\n=== Epoch {epoch_idx}/{start_epoch + args.epochs - 1} — resetting substrate ===")
            substrate = read_substrate(args.substrate)
            algo._init_controller(substrate)
            algo._active_mappings.clear()
            if ref_algo is not None:
                ref_algo._init_controller(substrate)
                ref_algo._active_mappings.clear()
            # Reset reward baselines per epoch so per-epoch metrics are
            # comparable (start from same state).
            success_cost_ema = None
            ratio_ema = None
            buf = []
            ppo_buf = []
        # Reset per-epoch accumulators.
        ep_n_total = 0
        ep_n_success = 0
        ep_loss_sum = 0.0
        ep_loss_count = 0
        ep_policy_sum = 0.0
        ep_kl_sum = 0.0
        ep_ent_sum = 0.0
        ep_vloss_sum = 0.0
        ep_advantage_sum = 0.0
        ep_ratio_sum = 0.0
        ep_clip_sum = 0.0
        ep_reward_sum = 0.0
        ep_reward_count = 0
        ep_start_time = time.time()

        for ep, vnr in enumerate(vnrs):
            vn = vnr.virtual_network
            algo._release_expired(vnr.arrival_time)
            algo.global_controller.clear_caches()
            if ref_algo is not None:
                ref_algo._release_expired(vnr.arrival_time)
                ref_algo.global_controller.clear_caches()

            # PPO needs the EXACT substrate state at this VNR's rollout time so
            # the later replay can recompute log_p with consistent cand_pools.
            snap_pre = snapshot_substrate(algo) if args.ppo_mode == "ppo" else None

            n_total += 1
            ep_n_total += 1
    
            # Optional filter: skip provably-infeasible VNRs (would always fail
            # regardless of policy choices → only contribute noise to gradients).
            if args.filter_impossible:
                vn_cpu_demand = sum(vnode.cpu_demand for vnode in vn.nodes.values())
                total_cpu_avail = 0.0
                for dom in substrate.domains.values():
                    for snode in dom.network.nodes.values():
                        total_cpu_avail += getattr(snode, "available_cpu",
                                                   snode.cpu_capacity)
                if vn_cpu_demand > total_cpu_avail:
                    continue   # impossible — skip episode
    
            # 1. Rollout → mapping (+ differentiable log-probs).
            if args.rollout == "direct":
                # Autoregressive cand selection with SAMPLED candidates (1 snode/
                # vnode) → real exploration over snodes. sample=True matches the
                # il_mp_vne_v17_direct eval path exactly (order + cand both sampled);
                # ordering heads are frozen so only cand_head receives gradient.
                try:
                    (ordered_vnodes, ordered_links, mapping,
                     log_probs_dict, entropies_dict, value) = algo.rank_direct(
                        vn, sample=True)
                except Exception:
                    continue
                candidate_nodes = True  # sentinel: direct path always has cand_lp
            else:  # pso
                # sample_cand=True (when training cand) → per-domain candidate is
                # SAMPLED → real exploration in the deployed PSO regime.
                try:
                    (ordered_vnodes, ordered_links, candidate_nodes, cand_weights,
                     log_probs_dict, entropies_dict, value) = algo.rank_all_nn(
                        vn, sample=True, sample_cand=(args.target == "cand"))
                except Exception:
                    continue
                if not candidate_nodes or any(not c for c in candidate_nodes):
                    continue
                vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
                vlink_indices = [{
                    "src_idx": vnode_to_idx[vl.source],
                    "dst_idx": vnode_to_idx[vl.target],
                    "bw": vl.bandwidth_demand,
                } for vl in vn.links.values()]
                with torch.no_grad():
                    best_particle = algo._pso(candidate_nodes, vlink_indices, ordered_vnodes, cand_weights)
                    mapping = {ordered_vnodes[i].id: candidate_nodes[i][idx].id
                               for i, idx in enumerate(best_particle)}
    
            # 2. Commit + measure cost (persistent → on-policy state evolution).
            actual_cost = None
            succeeded = False
            if mapping is not None:
                try:
                    vlink_paths = algo._commit_mapping_ordered(mapping, vn, ordered_links)
                    if vlink_paths:
                        actual_cost = algo._compute_cost(mapping, vn, vlink_paths)
                        succeeded = True
                        n_success += 1
                        ep_n_success += 1
                        algo._active_mappings[vnr.id] = {
                            "mapping": mapping, "vnetwork": vn,
                            "vlink_paths": vlink_paths,
                            "expire_time": vnr.arrival_time + vnr.lifetime,
                        }
                except ValueError:
                    actual_cost = None
    
            # 4. Reward.
            if succeeded and actual_cost is not None:
                if args.reward_mode == "cost_rel":
                    # V19/V21 default: cost_EMA baseline (size-DEPENDENT).
                    if success_cost_ema is None:
                        success_cost_ema = actual_cost
                    else:
                        success_cost_ema = (args.success_cost_ema * success_cost_ema
                                            + (1 - args.success_cost_ema) * actual_cost)
                    rel_cost = (success_cost_ema - actual_cost) / success_cost_ema
                    reward = args.success_bonus + args.cost_lambda * rel_cost
                else:  # rev_cost  — V22: rev/cost ratio (size-INVARIANT, clipped)
                    vn_revenue = (sum(n.cpu_demand for n in vn.nodes.values())
                                  + sum(l.bandwidth_demand for l in vn.links.values()))
                    ratio = vn_revenue / actual_cost
                    if ratio_ema is None:
                        ratio_ema = ratio
                    else:
                        ratio_ema = (args.ratio_ema_decay * ratio_ema
                                     + (1 - args.ratio_ema_decay) * ratio)
                    rel = (ratio - ratio_ema) / ratio_ema
                    rel = max(-args.efficiency_clip, min(args.efficiency_clip, rel))
                    reward = args.success_bonus + args.efficiency_weight * rel
                # Compare with expert for logging only.
                exp_c = expert_costs.get(vnr.id)
                if exp_c is not None:
                    n_expert_compared += 1
                    if actual_cost < exp_c:
                        n_better_than_expert += 1
            else:
                # Fail reward — optionally scaled by trajectory progress so partial
                # successes are penalised less than instant failures.
                if args.progress_fail_reward and mapping is not None:
                    progress = len(mapping) / max(len(vn.nodes), 1)   # 0..1
                    reward = args.fail_reward * (1.0 - 0.5 * progress)
                else:
                    reward = args.fail_reward
    
            # 5. Differentiable terms (advantage computed at batch level).
            # Include node/link ordering log-probs (random after IL pretrain → the
            # main acceptance lever under K=1) plus candidate-selection log-probs.
            # Only include ordering log-probs when ordering IS the training target.
            # When target=cand the ordering heads are frozen → cand log-probs only.
            include_ordering = (args.target == "ordering") and args.train_ordering
            lps = list(log_probs_dict["cand"])
            if include_ordering:
                lps += list(log_probs_dict["node"]) + list(log_probs_dict["link"])
            if not lps:
                continue
            log_p_sum = torch.stack(lps).sum()
    
            ents = list(entropies_dict["cand"])
            if include_ordering:
                ents += list(entropies_dict["node"]) + list(entropies_dict["link"])
            entropy = torch.stack(ents).sum() if ents else torch.tensor(0.0)
    
            # KL vs frozen reference (only when shared encoder is NOT frozen).
            # In PPO mode the KL is computed during the replay loop instead, so
            # skip the extra per-VNR forward pass here.
            kl = None
            if use_kl and args.ppo_mode != "ppo":
                with torch.no_grad():
                    _, _, ref_cand_scores, _, _ = ref_algo._forward_policy(vn)
                _, _, new_cand_scores, _, _ = algo._forward_policy(vn)
                kl = compute_kl(new_cand_scores, ref_cand_scores)

            ep_reward_sum += reward
            ep_reward_count += 1
            if args.ppo_mode == "ppo":
                # Store detached old log-prob + snapshot for the PPO replay.
                # We drop the rollout's autograd graph (no grad path through
                # rollout) — gradients in PPO mode come from the REPLAY forward.
                ppo_buf.append({
                    "vn": vn,
                    "mapping": mapping,
                    "snap_pre": snap_pre,
                    "old_log_p_sum": log_p_sum.detach(),
                    "reward": reward,
                    "succeeded": succeeded,
                })
            else:
                buf.append({
                    "log_p_sum": log_p_sum,
                    "entropy": entropy,
                    "value": value,          # scalar tensor with grad (critic)
                    "kl": kl,
                    "reward": reward,
                })
    
            # ---- Batch update: compute advantages across the batch, single backward ----
            if (ep + 1) % args.batch_size == 0:
                if args.ppo_mode == "ppo":
                    last = ppo_batch_update(algo, ref_algo, ppo_buf, optimizer,
                                            trainable_params, args, use_kl)
                    if last is not None:
                        losses_recent.append(last["loss"])
                        last_policy = last["policy_loss"]
                        last_kl = last["kl"]
                        last_ent = last["entropy"]
                        last_vloss = last["value_loss"]
                        last_rmean = last["reward_mean"]
                        last_radv = last["advantage_mean"]
                        last_ratio = last["ratio_mean"]
                        last_clip = last["ratio_clip_frac"]
                        ep_loss_sum += last["loss"]
                        ep_policy_sum += last_policy
                        ep_kl_sum += last_kl
                        ep_ent_sum += last_ent
                        ep_vloss_sum += last_vloss
                        ep_advantage_sum += last_radv
                        ep_ratio_sum += last_ratio
                        ep_clip_sum += last_clip
                        ep_loss_count += 1
                elif buf:
                    rewards = torch.tensor([b["reward"] for b in buf], dtype=torch.float32)
                    log_p_sums = torch.stack([b["log_p_sum"] for b in buf])
                    entropies = torch.stack([b["entropy"] for b in buf])
    
                    if args.use_critic:
                        values = torch.stack([b["value"] for b in buf]).reshape(-1)
                        advantages = rewards - values.detach()
                        value_loss = F.mse_loss(values, rewards)
                    else:
                        advantages = rewards - reward_ema
                        value_loss = torch.tensor(0.0)
                        reward_ema = (args.baseline_ema * reward_ema
                                      + (1 - args.baseline_ema) * rewards.mean().item())
    
                    if args.normalize_adv and advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
                    policy_loss = -(log_p_sums * advantages).mean()
                    entropy_term = entropies.mean()
                    kl_term = (torch.stack([b["kl"] for b in buf]).mean()
                               if use_kl else torch.tensor(0.0))
    
                    loss = (policy_loss
                            + args.beta_kl * kl_term
                            - args.beta_entropy * entropy_term
                            + args.value_coef * value_loss)
    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                    optimizer.step()
                    losses_recent.append(loss.detach().item())
    
                    last_policy = policy_loss.detach().item()
                    last_kl = kl_term.detach().item()
                    last_ent = entropy_term.detach().item()
                    last_vloss = value_loss.detach().item() if args.use_critic else 0.0
                    last_radv = advantages.detach().mean().item()
                    # REINFORCE has no IS ratio — set to 1.0 / 0.0 placeholders so
                    # CSV columns line up with the PPO mode case.
                    last_ratio = 1.0
                    last_clip = 0.0
                    # Per-epoch accumulators (batch-level metrics).
                    ep_loss_sum += loss.detach().item()
                    ep_policy_sum += last_policy
                    ep_kl_sum += last_kl
                    ep_ent_sum += last_ent
                    ep_vloss_sum += last_vloss
                    ep_advantage_sum += last_radv
                    ep_ratio_sum += last_ratio
                    ep_clip_sum += last_clip
                    ep_loss_count += 1
                    last_rmean = rewards.mean().item()
                buf = []
                ppo_buf = []
                batch_idx += 1
    
                if batch_idx % args.print_every == 0:
                    elapsed = time.time() - start
                    eta = elapsed * (len(vnrs) - ep - 1) / max(ep + 1, 1)
                    recent = losses_recent[-args.print_every:]
                    avg_loss = sum(recent) / max(len(recent), 1)
                    succ_rate = n_success / max(n_total, 1)
                    better_rate = n_better_than_expert / max(n_expert_compared, 1)
                    # Extra PPO diagnostics in stdout (ratio, clip-fraction).
                    ppo_extra = (f"r={last_ratio:5.3f} clip={last_clip:5.1%} "
                                 if args.ppo_mode == "ppo" else "")
                    print(
                        f"  ep {ep+1:5d}/{len(vnrs)} batch {batch_idx:4d} "
                        f"loss={avg_loss:8.3f} "
                        f"pol={last_policy:7.3f} "
                        f"vL={last_vloss:6.3f} "
                        f"kl={last_kl:5.2f} "
                        f"H={last_ent:6.2f} "
                        f"adv={last_radv:+6.3f} "
                        f"{ppo_extra}"
                        f"r̄={last_rmean:6.3f} "
                        f"succ={succ_rate:5.1%} "
                        f">exp={better_rate:5.1%} "
                        f"el={fmt(elapsed)} eta={fmt(eta)}"
                    )
                    log_f.write(
                        f"{epoch_idx},{ep+1},{batch_idx},{avg_loss:.4f},"
                        f"{last_policy:.4f},{last_vloss:.4f},{last_kl:.4f},"
                        f"{last_ent:.4f},{last_rmean:.4f},{last_radv:.4f},"
                        f"{last_ratio:.4f},{last_clip:.4f},"
                        f"{succ_rate:.4f},{elapsed:.1f}\n"
                    )
                    log_f.flush()
        # --- End of one epoch over the VNR sequence: write summary row. ---
        ep_succ_rate = ep_n_success / max(ep_n_total, 1)
        mean_loss = ep_loss_sum / max(ep_loss_count, 1)
        mean_pol = ep_policy_sum / max(ep_loss_count, 1)
        mean_vl = ep_vloss_sum / max(ep_loss_count, 1)
        mean_kl = ep_kl_sum / max(ep_loss_count, 1)
        mean_ent = ep_ent_sum / max(ep_loss_count, 1)
        mean_adv = ep_advantage_sum / max(ep_loss_count, 1)
        mean_ratio = ep_ratio_sum / max(ep_loss_count, 1)
        mean_clip = ep_clip_sum / max(ep_loss_count, 1)
        mean_reward = ep_reward_sum / max(ep_reward_count, 1)
        ep_elapsed = time.time() - ep_start_time
        ppo_extra_print = (f" mean_ratio={mean_ratio:.3f} mean_clip={mean_clip:.1%}"
                           if args.ppo_mode == "ppo" else "")
        print(f"  >>> Epoch {epoch_idx}/{start_epoch + args.epochs - 1} DONE: "
              f"succ={ep_succ_rate:.1%} ({ep_n_success}/{ep_n_total}) "
              f"mean_loss={mean_loss:7.4f} "
              f"mean_reward={mean_reward:6.3f} "
              f"mean_KL={mean_kl:.3f} mean_H={mean_ent:.2f}"
              f"{ppo_extra_print} "
              f"epoch_time={fmt(ep_elapsed)}")
        epoch_f.write(
            f"{epoch_idx},{ep_n_total},{ep_n_success},{ep_succ_rate:.4f},"
            f"{mean_loss:.4f},{mean_pol:.4f},{mean_vl:.4f},{mean_kl:.4f},"
            f"{mean_ent:.4f},{mean_reward:.4f},{mean_adv:.4f},"
            f"{mean_ratio:.4f},{mean_clip:.4f},{ep_elapsed:.1f}\n"
        )
        epoch_f.flush()
        # Save per-epoch checkpoint (epoch_idx suffix). Allows recovery of the
        # best-online-success checkpoint after training (e.g. when the final
        # epoch over-trains relative to a mid-training peak).
        if args.epochs > 1 or start_epoch > 1:
            ckpt_stem = Path(args.checkpoint)
            ep_ckpt = ckpt_stem.with_name(
                ckpt_stem.stem + f"_e{epoch_idx}" + ckpt_stem.suffix)
            torch.save({"policy_state_dict": algo.policy.state_dict()}, ep_ckpt)

    elapsed = time.time() - start
    print("-" * 72)
    print(f"  total time:               {fmt(elapsed)}")
    print(f"  success rate:             {n_success}/{n_total} = {n_success/max(n_total,1):.1%}")
    if n_expert_compared > 0:
        print(f"  beat expert cost rate:    {n_better_than_expert}/{n_expert_compared} = "
              f"{n_better_than_expert/n_expert_compared:.1%}")
    if not args.use_critic:
        print(f"  final reward EMA:         {reward_ema:.4f}")
    print(f"  final success_cost_ema:   {success_cost_ema}")
    if losses_recent:
        print(f"  final loss avg:           {sum(losses_recent[-args.print_every:])/min(args.print_every, len(losses_recent)):.4f}")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": algo.policy.state_dict()}, args.checkpoint)
    print(f"  checkpoint saved to {args.checkpoint}")
    log_f.close()
    epoch_f.close()
    print(f"  epoch summary log: {epoch_log_path}")


if __name__ == "__main__":
    main()
