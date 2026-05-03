import torch
import torch.optim as optim
from typing import List, Tuple, Dict
import os
import time

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
        self.buffer: List[Tuple[Dict[str, List[torch.Tensor]], float]] = []

        # Open log file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.loss_log_file = os.path.join(log_dir, "rl_training_loss.csv")
        # Write header if file doesn't exist or is empty
        if not os.path.exists(self.loss_log_file) or os.path.getsize(self.loss_log_file) == 0:
            with open(self.loss_log_file, "w") as f:
                f.write("timestamp,avg_reward,total_loss,node_loss,link_loss,cand_loss\n")

    def record(self, log_probs: Dict[str, List[torch.Tensor]], reward: float) -> None:
        """Store one episode's log-probabilities and reward."""
        self.buffer.append((log_probs, reward))

    def update(self) -> Dict[str, float]:
        """
        Run REINFORCE update over buffered experiences.
        Returns a dictionary of average loss values.
        """
        if not self.buffer:
            return {"avg_reward": 0.0, "total_loss": 0.0, "node_loss": 0.0, "link_loss": 0.0, "cand_loss": 0.0}

        rewards = [r for _, r in self.buffer]
        baseline = sum(rewards) / len(rewards)

        total_node_loss = torch.tensor(0.0)
        total_link_loss = torch.tensor(0.0)
        total_cand_loss = torch.tensor(0.0)

        for log_probs, reward in self.buffer:
            advantage = reward - baseline
            
            node_loss = torch.tensor(0.0)
            for lp in log_probs.get("node", []):
                node_loss = node_loss - lp * advantage
                
            link_loss = torch.tensor(0.0)
            for lp in log_probs.get("link", []):
                link_loss = link_loss - lp * advantage
                
            cand_loss = torch.tensor(0.0)
            for lp in log_probs.get("cand", []):
                cand_loss = cand_loss - lp * advantage
                
            total_node_loss = total_node_loss + node_loss
            total_link_loss = total_link_loss + link_loss
            total_cand_loss = total_cand_loss + cand_loss

        total_node_loss = total_node_loss / len(self.buffer)
        total_link_loss = total_link_loss / len(self.buffer)
        total_cand_loss = total_cand_loss / len(self.buffer)
        total_loss = total_node_loss + total_link_loss + total_cand_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_dict = {
            "avg_reward": baseline,
            "total_loss": total_loss.item(),
            "node_loss": total_node_loss.item(),
            "link_loss": total_link_loss.item(),
            "cand_loss": total_cand_loss.item()
        }
        
        # Append to log file
        with open(self.loss_log_file, "a") as f:
            f.write(f"{time.time()},{loss_dict['avg_reward']:.6f},{loss_dict['total_loss']:.6f},{loss_dict['node_loss']:.6f},{loss_dict['link_loss']:.6f},{loss_dict['cand_loss']:.6f}\n")

        self.buffer.clear()
        return loss_dict
