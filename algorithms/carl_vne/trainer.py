import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.optim as optim

from algorithms.carl_vne.policy_network import PolicyNetwork


class RankingTrainer:
    """
    Actor-Critic trainer for V2 — redesigned optimization to push reward beyond
    the plateau seen with the original (fixed-coef) version.

    Key optimization redesigns vs. the V1 trainer:
      1. **Advantage normalization within batch** — `A = (A - A.mean()) / A.std()`
         after subtracting V(s). Stabilizes gradient magnitude across batches
         even when reward variance shifts.
      2. **Entropy annealing** — β linearly decays from `entropy_start_coef`
         (high, force exploration) to `entropy_end_coef` (low, exploit) over
         `entropy_anneal_batches`. Directly attacks the local-optimum plateau.
      3. **Critic coefficient ramp-up** — c_v scales smoothly from 0 to its
         target over `critic_warmup_batches` instead of a hard step, so policy
         gradient is never multiplied by a noisy V(s) early on.
      4. **AdamW + weight decay** + **cosine LR schedule** — regularization
         and learning-rate annealing for the late-training fine-tuning phase
         where REINFORCE typically stalls.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        critic_coef: float = 0.5,
        # Entropy schedule: start high, decay low.
        entropy_start_coef: float = 0.05,
        entropy_end_coef: float = 0.005,
        entropy_anneal_batches: int = 60,
        # Critic warmup: ramp 0 → 1.0 (multiplier on critic_coef).
        critic_warmup_batches: int = 8,
        # Optimization knobs.
        weight_decay: float = 1e-4,
        lr_cosine_total_batches: int = 80,
        lr_min_ratio: float = 0.1,
        # Advantage normalization (set False to disable).
        normalize_advantage: bool = True,
        # Self-critic mode: reward already serves as advantage (greedy rollout
        # is the baseline). Skips V(s) subtraction, batch normalization, and
        # critic gradient. Set on the trainer instance by the self-critic
        # pretrain driver; default False preserves V1 actor-critic behavior.
        self_critic_mode: bool = False,
    ):
        self.policy = policy
        self.optimizer = optim.AdamW(
            policy.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.lr_cosine_total = lr_cosine_total_batches

        self.gamma = gamma
        self.batch_size = batch_size
        self.critic_coef = critic_coef
        self.entropy_start = entropy_start_coef
        self.entropy_end = entropy_end_coef
        self.entropy_anneal_batches = max(entropy_anneal_batches, 1)
        self.critic_warmup_batches = max(critic_warmup_batches, 1)
        self.normalize_advantage = normalize_advantage
        self.self_critic_mode = self_critic_mode

        self._update_count = 0
        self.buffer: List[Tuple[Dict, torch.Tensor, Dict, float]] = []

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.loss_log_file = os.path.join(log_dir, "rl_training_loss.csv")
        if not os.path.exists(self.loss_log_file) or os.path.getsize(self.loss_log_file) == 0:
            with open(self.loss_log_file, "w") as f:
                f.write("timestamp,avg_reward,total_loss,node_loss,link_loss,cand_loss,"
                        "critic_loss,value_mean,entropy,adv_std,entropy_coef,lr\n")

    # ---- Schedules ----

    def _current_entropy_coef(self) -> float:
        """Linear decay from start to end over `entropy_anneal_batches`."""
        t = min(self._update_count, self.entropy_anneal_batches) / self.entropy_anneal_batches
        return (1.0 - t) * self.entropy_start + t * self.entropy_end

    def _current_critic_scale(self) -> float:
        """Linear ramp 0 → 1 over `critic_warmup_batches`. Smooth replacement
        for the hard step in the previous implementation."""
        if self._update_count >= self.critic_warmup_batches:
            return 1.0
        return self._update_count / self.critic_warmup_batches

    def _current_lr(self) -> float:
        """Cosine annealing: lr_init → lr_min over `lr_cosine_total`."""
        if self.lr_cosine_total <= 0:
            return self.lr_init
        t = min(self._update_count, self.lr_cosine_total) / self.lr_cosine_total
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return self.lr_min + (self.lr_init - self.lr_min) * cosine

    def _apply_lr(self) -> float:
        lr = self._current_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    # ---- Buffer ----

    def record(
        self,
        log_probs: Dict[str, List[torch.Tensor]],
        value: torch.Tensor,
        entropies: Dict[str, List[torch.Tensor]],
        reward: float,
    ) -> None:
        self.buffer.append((log_probs, value, entropies, reward))

    # ---- Update ----

    def update(self) -> Dict[str, float]:
        if not self.buffer:
            return {k: 0.0 for k in (
                "avg_reward", "total_loss", "node_loss", "link_loss", "cand_loss",
                "critic_loss", "value_mean", "entropy", "adv_std",
                "entropy_coef", "lr",
            )}

        rewards_t = torch.tensor([r for _, _, _, r in self.buffer], dtype=torch.float32)
        values_t = torch.stack([v for _, v, _, _ in self.buffer])  # keep grad

        avg_reward = rewards_t.mean().item()

        if self.self_critic_mode:
            # Self-critic baseline (Rennie et al.): the greedy rollout in the
            # data-collection step already centers the reward, so reward IS the
            # advantage. Skip V(s) subtraction and batch-normalization — both
            # would re-baseline an already-centered signal and destroy its
            # meaningful zero-point (failures vs near-parity collapse to the
            # same magnitude after std-divide).
            advantages = rewards_t
            adv_std = rewards_t.std().item() if rewards_t.numel() > 1 else 0.0
        else:
            # Per-instance advantage; optionally normalize within batch for stable
            # gradient magnitude across episodes with different reward ranges.
            raw_advantages = rewards_t - values_t.detach()
            adv_std = raw_advantages.std().item() if raw_advantages.numel() > 1 else 0.0
            if self.normalize_advantage and adv_std > 1e-6:
                advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
            else:
                advantages = raw_advantages - raw_advantages.mean()

        critic_loss = ((values_t - rewards_t) ** 2).mean()

        # Policy loss with per-head normalization.
        total_node_loss = torch.tensor(0.0)
        total_link_loss = torch.tensor(0.0)
        total_cand_loss = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)
        n_node = n_link = n_cand = n_ent = 0

        # Keep adv as a tensor (detached from V-graph but staying in autograd
        # so per-step operations are batched efficiently by torch).
        for (log_probs, _, entropies, _), adv in zip(self.buffer, advantages):
            for lp in log_probs.get("node", []):
                total_node_loss = total_node_loss - lp * adv
                n_node += 1
            for lp in log_probs.get("link", []):
                total_link_loss = total_link_loss - lp * adv
                n_link += 1
            for lp in log_probs.get("cand", []):
                total_cand_loss = total_cand_loss - lp * adv
                n_cand += 1
            for h in entropies.get("node", []) + entropies.get("link", []) + entropies.get("cand", []):
                total_entropy = total_entropy + h
                n_ent += 1

        node_norm = total_node_loss / max(n_node, 1)
        link_norm = total_link_loss / max(n_link, 1)
        cand_norm = total_cand_loss / max(n_cand, 1)
        policy_loss = node_norm + link_norm + cand_norm
        entropy_mean = total_entropy / max(n_ent, 1)

        # Smooth critic warmup ramp + annealed entropy + scheduled LR.
        critic_scale = self._current_critic_scale()
        ent_coef = self._current_entropy_coef()
        if self.self_critic_mode:
            # No V(s) baseline → no reason to delay policy gradient or train
            # a critic that doesn't enter the policy loss.
            policy_scale = 1.0
            critic_scale = 0.0
        else:
            # Don't apply policy gradient until critic is partially trained
            # (first half of warmup) — avoids steering policy by noisy V(s).
            policy_scale = 1.0 if self._update_count >= self.critic_warmup_batches // 2 else 0.0

        total_loss = (
            policy_scale * policy_loss
            + critic_scale * self.critic_coef * critic_loss
            - ent_coef * entropy_mean
        )

        lr = self._apply_lr()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()
        self._update_count += 1

        loss_dict = {
            "avg_reward": avg_reward,
            "total_loss": total_loss.item(),
            "node_loss": node_norm.item(),
            "link_loss": link_norm.item(),
            "cand_loss": cand_norm.item(),
            "critic_loss": critic_loss.item(),
            "value_mean": values_t.mean().item(),
            "entropy": entropy_mean.item(),
            "adv_std": adv_std,
            "entropy_coef": ent_coef,
            "lr": lr,
        }

        with open(self.loss_log_file, "a") as f:
            f.write(
                f"{time.time()},{loss_dict['avg_reward']:.6f},{loss_dict['total_loss']:.6f},"
                f"{loss_dict['node_loss']:.6f},{loss_dict['link_loss']:.6f},"
                f"{loss_dict['cand_loss']:.6f},{loss_dict['critic_loss']:.6f},"
                f"{loss_dict['value_mean']:.6f},{loss_dict['entropy']:.6f},"
                f"{loss_dict['adv_std']:.6f},{loss_dict['entropy_coef']:.6f},"
                f"{loss_dict['lr']:.6f}\n"
            )

        self.buffer.clear()
        return loss_dict
