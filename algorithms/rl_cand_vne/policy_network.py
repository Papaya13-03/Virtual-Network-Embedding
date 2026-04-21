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
        # If all remaining logits are -inf (all infeasible), stop early.
        if torch.all(remaining_logits == float("-inf")):
            break
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


class PolicyNetwork(nn.Module):
    """
    Full policy. For each vnode A:
      1. VN encoder over the virtual graph → h_A.
      2. Domain encoder over each of A's allowed domains → per-snode e_s and pooled g_d.
      3. Domain head picks one allowed domain d* (sample/argmax).
      4. SNode head scores snodes in d*; top-K (Plackett-Luce/argsort) → candidate set.
    """

    def __init__(
        self,
        vnode_feat_size: int = 5,
        snode_feat_size: int = 5,
        hidden: int = 64,
        K: int = 5,
    ):
        super().__init__()
        self.hidden = hidden
        self.K = K
        self.vn_encoder = GCNEncoder(in_dim=vnode_feat_size, hidden=hidden)
        self.domain_encoder = GCNEncoder(in_dim=snode_feat_size, hidden=hidden)
        self.domain_head = DomainHead(hidden=hidden)
        self.snode_head = SNodeHead(hidden=hidden)

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vn_adj_norm: torch.Tensor,
        domain_inputs_per_vnode: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        cpu_demands: List[float],
        sample: bool = True,
    ):
        """
        vnode_feats: (n_v, f_v)
        vn_adj_norm: (n_v, n_v)
        domain_inputs_per_vnode[i] = list of (X_d, A_d_norm, available_cpu) for each allowed domain of vnode i.
        cpu_demands[i]: cpu demand for vnode i (used for feasibility mask).
        sample=True uses stochastic sampling (training). False = argmax / top-K (inference).

        Returns dict:
          chosen_domains: list[int]                         # index into vnode i's allowed domain list
          chosen_snodes: list[list[int]]                    # per vnode, list of snode indices within chosen domain
          domain_log_probs: list[Tensor]                    # scalar per vnode
          snode_log_probs_per_vnode: list[list[Tensor]]     # per vnode, list of log-probs (length = len(chosen_snodes[i]))
        """
        H_v = self.vn_encoder(vnode_feats, vn_adj_norm)  # (n_v, hidden)
        n_v = H_v.shape[0]

        chosen_domains: List[int] = []
        chosen_snodes: List[List[int]] = []
        domain_log_probs: List[torch.Tensor] = []
        snode_log_probs_per_vnode: List[List[torch.Tensor]] = []

        for i in range(n_v):
            h_A = H_v[i]
            allowed = domain_inputs_per_vnode[i]
            assert len(allowed) >= 1, f"vnode {i} has no allowed domains"

            # Encode each allowed domain once
            per_domain_E = []
            per_domain_g = []
            per_domain_avail = []
            for (X_d, A_d, avail) in allowed:
                E_d = self.domain_encoder(X_d, A_d)
                per_domain_E.append(E_d)
                per_domain_g.append(E_d.mean(dim=0))
                per_domain_avail.append(avail)

            g_stack = torch.stack(per_domain_g, dim=0)  # (n_allowed, hidden)

            dom_logits = self.domain_head(h_A, g_stack)
            if sample:
                dom_dist = torch.distributions.Categorical(logits=dom_logits)
                d_star = dom_dist.sample()
                domain_log_probs.append(dom_dist.log_prob(d_star))
                d_idx = d_star.item()
            else:
                d_idx = int(torch.argmax(dom_logits).item())
                domain_log_probs.append(torch.softmax(dom_logits, dim=0)[d_idx].log())
            chosen_domains.append(d_idx)

            E_d = per_domain_E[d_idx]
            g_d = per_domain_g[d_idx]
            avail = per_domain_avail[d_idx]
            sn_logits = self.snode_head(h_A, g_d, E_d, avail, cpu_demands[i])

            if sample:
                snode_idx, snode_lps = plackett_luce_topk(sn_logits, self.K)
            else:
                k_eff = min(self.K, sn_logits.shape[0])
                order = torch.argsort(sn_logits, descending=True)[:k_eff].tolist()
                probs = torch.softmax(sn_logits, dim=0)
                snode_idx = order
                snode_lps = [probs[j].log() for j in order]
            chosen_snodes.append(snode_idx)
            snode_log_probs_per_vnode.append(snode_lps)

        return {
            "chosen_domains": chosen_domains,
            "chosen_snodes": chosen_snodes,
            "domain_log_probs": domain_log_probs,
            "snode_log_probs_per_vnode": snode_log_probs_per_vnode,
        }
