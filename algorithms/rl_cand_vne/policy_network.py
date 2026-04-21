from typing import List, Tuple
import torch
import torch.nn as nn


class GCNEncoder(nn.Module):
    """2-layer graph convolutional network."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=True)
        self.W2 = nn.Linear(hidden, hidden, bias=True)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = torch.relu(self.W1(A_norm @ X))
        H = torch.relu(self.W2(A_norm @ H))
        return H


def plackett_luce_topk(
    logits: torch.Tensor, k: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Sample an ordered subset of size min(k, n) via Plackett-Luce (sampling without replacement).
    Returns (indices into original logits, list of log-probs of each draw).
    """
    n = logits.shape[0]
    k_eff = min(k, n)
    remaining_logits = logits.clone()
    remaining_idx = list(range(n))
    chosen: List[int] = []
    log_probs: List[torch.Tensor] = []

    for _ in range(k_eff):
        probs = torch.softmax(remaining_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        pos = dist.sample()
        log_probs.append(dist.log_prob(pos))
        chosen.append(remaining_idx[pos.item()])
        mask = torch.ones(len(remaining_idx), dtype=torch.bool)
        mask[pos.item()] = False
        remaining_logits = remaining_logits[mask]
        remaining_idx = [ri for j, ri in enumerate(remaining_idx) if j != pos.item()]

    return chosen, log_probs
