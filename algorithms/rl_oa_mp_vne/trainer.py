import torch
import torch.optim as optim
from typing import List, Tuple, Dict
import os
import time

from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork


class RankingTrainer:
    """
    Actor-Critic trainer for the ranking policy network.

    Key design choices:
      - **State-value baseline V(s)** (Actor-Critic): advantage = R - V(s).
        Replaces batch-mean baseline; gives per-VN-instance baseline so
        gradient is not diluted by VN-sampling variance.
      - **Critic MSE loss**: (R - V(s))² — the only loss in this system with
        a direct supervised-learning interpretation (regression on returns).
      - **Per-head loss normalization**: each head's loss divided by its
        number of log_prob terms. Without this, cand-head (K * num_vnode
        terms) dominates the gradient and node/link heads barely train.
      - **Entropy bonus**: -β * H(π) keeps the policy exploring; prevents
        the entropy-collapse failure mode seen in earlier runs.
      - **Critic warmup**: first N episodes train critic only (policy_loss
        scaled to 0); without warmup, advantages computed against an
        untrained V(s) are pure noise.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        critic_coef: float = 0.5,
        entropy_coef: float = 0.01,
        critic_warmup_batches: int = 4,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.critic_warmup_batches = critic_warmup_batches
        self._update_count = 0
        self.buffer: List[Tuple[Dict, torch.Tensor, Dict, float]] = []

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.loss_log_file = os.path.join(log_dir, "rl_training_loss.csv")
        if not os.path.exists(self.loss_log_file) or os.path.getsize(self.loss_log_file) == 0:
            with open(self.loss_log_file, "w") as f:
                f.write("timestamp,avg_reward,total_loss,node_loss,link_loss,cand_loss,"
                        "critic_loss,value_mean,entropy,adv_std\n")

    def record(
        self,
        log_probs: Dict[str, List[torch.Tensor]],
        value: torch.Tensor,
        entropies: Dict[str, List[torch.Tensor]],
        reward: float,
    ) -> None:
        """Store one episode's experience."""
        self.buffer.append((log_probs, value, entropies, reward))

    def update(self) -> Dict[str, float]:
        """Actor-Critic update."""
        if not self.buffer:
            return {k: 0.0 for k in (
                "avg_reward", "total_loss", "node_loss", "link_loss", "cand_loss",
                "critic_loss", "value_mean", "entropy", "adv_std",
            )}

        rewards_t = torch.tensor([r for _, _, _, r in self.buffer], dtype=torch.float32)
        values_t = torch.stack([v for _, v, _, _ in self.buffer])  # keep grad

        # Per-instance advantage from V(s) — no batch standardization needed
        # because V(s) is the per-VN baseline.
        advantages = rewards_t - values_t.detach()
        avg_reward = rewards_t.mean().item()
        adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0

        # Critic loss — MSE between predicted value and observed return.
        critic_loss = ((values_t - rewards_t) ** 2).mean()

        # Policy loss: per-head accumulation, normalized by num terms.
        total_node_loss = torch.tensor(0.0)
        total_link_loss = torch.tensor(0.0)
        total_cand_loss = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)
        n_node = n_link = n_cand = 0
        n_ent = 0

        for (log_probs, _, entropies, _), adv in zip(self.buffer, advantages):
            adv_val = adv.item()
            for lp in log_probs.get("node", []):
                total_node_loss = total_node_loss - lp * adv_val
                n_node += 1
            for lp in log_probs.get("link", []):
                total_link_loss = total_link_loss - lp * adv_val
                n_link += 1
            for lp in log_probs.get("cand", []):
                total_cand_loss = total_cand_loss - lp * adv_val
                n_cand += 1
            for h in entropies.get("node", []) + entropies.get("link", []) + entropies.get("cand", []):
                total_entropy = total_entropy + h
                n_ent += 1

        # Normalize per-head by number of log_prob terms so node/link don't
        # get drowned out by cand (which has K * num_vnode terms).
        node_norm = total_node_loss / max(n_node, 1)
        link_norm = total_link_loss / max(n_link, 1)
        cand_norm = total_cand_loss / max(n_cand, 1)
        policy_loss = node_norm + link_norm + cand_norm

        entropy_mean = total_entropy / max(n_ent, 1)

        # Warmup: critic-only training in early batches.
        if self._update_count < self.critic_warmup_batches:
            policy_scale = 0.0
        else:
            policy_scale = 1.0

        total_loss = (
            policy_scale * policy_loss
            + self.critic_coef * critic_loss
            - self.entropy_coef * entropy_mean
        )

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
        }

        with open(self.loss_log_file, "a") as f:
            f.write(
                f"{time.time()},{loss_dict['avg_reward']:.6f},{loss_dict['total_loss']:.6f},"
                f"{loss_dict['node_loss']:.6f},{loss_dict['link_loss']:.6f},"
                f"{loss_dict['cand_loss']:.6f},{loss_dict['critic_loss']:.6f},"
                f"{loss_dict['value_mean']:.6f},{loss_dict['entropy']:.6f},"
                f"{loss_dict['adv_std']:.6f}\n"
            )

        self.buffer.clear()
        return loss_dict
