from collections import deque
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from algorithms.rl_cand_vne.policy_network import PolicyNetwork


class Trainer:
    """
    REINFORCE (cost-minimizing via negative-cost/revenue reward) plus a supervised
    auxiliary loss on the actually-committed snode for successful episodes.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 1e-3,
        lam_sup: float = 1.0,
        baseline_window: int = 100,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.lam_sup = lam_sup
        self._baseline_buf: deque = deque(maxlen=baseline_window)
        self.buffer: List[Dict] = []

    def record(
        self,
        domain_log_probs: List[torch.Tensor],
        snode_log_probs_per_vnode: List[List[torch.Tensor]],
        reward: float,
        committed_snode_indices: Optional[List[int]],
        success: bool,
    ) -> None:
        self.buffer.append({
            "domain_log_probs": domain_log_probs,
            "snode_log_probs_per_vnode": snode_log_probs_per_vnode,
            "reward": float(reward),
            "committed_snode_indices": committed_snode_indices,
            "success": bool(success),
        })
        self._baseline_buf.append(float(reward))

    def baseline(self) -> float:
        if not self._baseline_buf:
            return 0.0
        return sum(self._baseline_buf) / len(self._baseline_buf)

    def update(self) -> Dict[str, float]:
        if not self.buffer:
            return {
                "loss_total": 0.0, "loss_rl": 0.0, "loss_sup": 0.0,
                "avg_reward": 0.0, "success_rate": 0.0, "baseline": self.baseline(),
            }

        b_val = self.baseline()
        total_loss = torch.zeros(())
        rl_loss = torch.zeros(())
        sup_loss = torch.zeros(())
        n_sup = 0
        n = len(self.buffer)
        success_count = 0

        for ep in self.buffer:
            R = ep["reward"]
            adv = R - b_val
            ep_log_prob_sum = torch.zeros(())
            for lp in ep["domain_log_probs"]:
                ep_log_prob_sum = ep_log_prob_sum + lp
            for lps in ep["snode_log_probs_per_vnode"]:
                for lp in lps:
                    ep_log_prob_sum = ep_log_prob_sum + lp
            rl_loss = rl_loss + (-adv) * ep_log_prob_sum

            if ep["success"]:
                success_count += 1
                indices = ep["committed_snode_indices"]
                # Supervised aux: maximize log-prob of the first sampled candidate per vnode
                # on successful episodes. This is a weak teacher (it nudges the top Plackett-
                # Luce draw upward) rather than a tight "exactly-committed-snode" signal,
                # because PSO selects the committed index from the K-sized candidate set —
                # not necessarily the first draw. A tighter teacher would require a second
                # forward pass to compute log pi(committed | vnode, domain) directly.
                for i, lps in enumerate(ep["snode_log_probs_per_vnode"]):
                    if not lps:
                        continue
                    sup_loss = sup_loss + (-lps[0])
                    n_sup += 1

        rl_loss = rl_loss / n
        if n_sup > 0:
            sup_loss = sup_loss / n_sup
        else:
            sup_loss = torch.zeros(())

        total_loss = rl_loss + self.lam_sup * sup_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_rl": float(rl_loss.item()),
            "loss_sup": float(sup_loss.item()),
            "avg_reward": sum(ep["reward"] for ep in self.buffer) / n,
            "success_rate": success_count / n,
            "baseline": b_val,
        }
        self.buffer.clear()
        return metrics
