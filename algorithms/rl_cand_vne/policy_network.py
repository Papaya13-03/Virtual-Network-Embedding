import math
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


class DomainHead(nn.Module):
    """Dot-product attention from vnode h_A to allowed-domain embeddings."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(self, h_A: torch.Tensor, g_domains: torch.Tensor) -> torch.Tensor:
        """
        h_A: (hidden,)
        g_domains: (num_allowed_domains, hidden)
        Returns: (num_allowed_domains,) logits (unnormalized).
        """
        q = self.W_q(h_A)
        k = self.W_k(g_domains)
        return (k @ q) / self.scale


class SNodeHead(nn.Module):
    """Dot-product attention from (h_A, g_d) to per-snode embeddings with feasibility mask."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W_q = nn.Linear(2 * hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def forward(
        self,
        h_A: torch.Tensor,
        g_d: torch.Tensor,
        e_snodes: torch.Tensor,
        available_cpu: torch.Tensor,
        cpu_demand: float,
    ) -> torch.Tensor:
        """
        h_A: (hidden,)
        g_d: (hidden,)
        e_snodes: (num_snodes, hidden)
        available_cpu: (num_snodes,)
        Returns: (num_snodes,) logits with infeasible snodes at -inf (unless all are infeasible).
        """
        q = self.W_q(torch.cat([h_A, g_d], dim=0))
        k = self.W_k(e_snodes)
        logits = (k @ q) / self.scale

        feasible = available_cpu >= cpu_demand
        if torch.any(feasible):
            logits = logits.masked_fill(~feasible, float("-inf"))
        return logits
