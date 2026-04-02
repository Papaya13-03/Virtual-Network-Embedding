import torch
import torch.optim as optim
from typing import List, Tuple

from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork


class RankingTrainer:
    """
    REINFORCE trainer for the ranking policy network.
    Accumulates (log_probs, reward) experiences in a buffer.
    On update(), computes policy gradient with baseline subtraction.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 16,
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer: List[Tuple[List[torch.Tensor], float]] = []

    def record(self, log_probs: List[torch.Tensor], reward: float) -> None:
        """Store one episode's log-probabilities and reward."""
        self.buffer.append((log_probs, reward))

    def update(self) -> float:
        """
        Run REINFORCE update over buffered experiences.
        Returns average loss value.
        """
        if not self.buffer:
            return 0.0

        rewards = [r for _, r in self.buffer]
        baseline = sum(rewards) / len(rewards)

        total_loss = torch.tensor(0.0)
        for log_probs, reward in self.buffer:
            advantage = reward - baseline
            episode_loss = torch.tensor(0.0)
            for lp in log_probs:
                episode_loss = episode_loss - lp * advantage
            total_loss = total_loss + episode_loss

        total_loss = total_loss / len(self.buffer)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        avg_loss = total_loss.item()
        self.buffer.clear()
        return avg_loss
