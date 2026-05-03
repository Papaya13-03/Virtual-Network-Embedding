import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class GCNEncoder(nn.Module):
    """2-layer Graph Convolutional Network for substrate domain embedding."""

    def __init__(self, node_feat_size: int, hidden_size: int = 32):
        super().__init__()
        self.W1 = nn.Linear(node_feat_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = torch.relu(self.W1(A_norm @ X))
        H = torch.relu(self.W2(A_norm @ H))
        return H


class CandidateHead(nn.Module):
    """Cross-attention scorer over substrate nodes for a single virtual node.

    Inspired by the Attention Model (Kool et al., ICLR'19): dot-product
    attention between a learned vnode query and substrate-node keys produced
    by the shared GCN encoder, augmented with a CPU-slack feature and a
    learned residual MLP. Infeasible snodes (cpu_slack < 0) are masked with
    -inf so softmax/top-K ignores them.
    """

    def __init__(self, vnode_feat_size: int, gcn_hidden: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** 0.5
        self.query_proj = nn.Linear(vnode_feat_size, hidden_size, bias=True)
        self.key_proj = nn.Linear(gcn_hidden + 1, hidden_size, bias=True)
        self.residual = nn.Sequential(
            nn.Linear(vnode_feat_size + gcn_hidden + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        vnode_feat: torch.Tensor,    # (vnode_feat_size,)
        snode_embeds: torch.Tensor,  # (N, gcn_hidden)
        cpu_slack: torch.Tensor,     # (N,)
    ) -> torch.Tensor:               # (N,) — infeasible entries masked to -inf
        n = snode_embeds.shape[0]
        # Normalize slack for numerical stability; keep sign (negative = infeasible).
        denom = cpu_slack.abs().max().clamp(min=1.0)
        slack_norm = (cpu_slack / denom).unsqueeze(-1)  # (N, 1)

        key_input = torch.cat([snode_embeds, slack_norm], dim=-1)
        q = self.query_proj(vnode_feat)          # (hidden,)
        k = self.key_proj(key_input)             # (N, hidden)
        attn = (k @ q) / self.scale              # (N,)

        q_expand = vnode_feat.unsqueeze(0).expand(n, -1)
        mlp_input = torch.cat([q_expand, snode_embeds, slack_norm], dim=-1)
        residual = self.residual(mlp_input).squeeze(-1)

        scores = attn + residual
        mask = cpu_slack < 0
        scores = scores.masked_fill(mask, float("-inf"))
        return scores


class PolicyNetwork(nn.Module):
    """
    Policy network for ranking vnodes, vlinks, and scoring substrate candidates.

    Three heads share the same substrate GCN encoder:
    - `node_head` ranks virtual nodes for OA-PSO ordering.
    - `link_head` ranks virtual links for OA-PSO ordering.
    - `candidate_head` scores substrate nodes per vnode (attention-based).
    """

    def __init__(
        self,
        vnode_feat_size: int = 5,
        vlink_feat_size: int = 5,
        gcn_node_feat_size: int = 5,
        gcn_hidden: int = 32,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.gcn = GCNEncoder(gcn_node_feat_size, gcn_hidden)

        node_input_size = vnode_feat_size + gcn_hidden
        self.node_head = nn.Sequential(
            nn.Linear(node_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        link_input_size = vlink_feat_size + gcn_hidden
        self.link_head = nn.Sequential(
            nn.Linear(link_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.candidate_head = CandidateHead(
            vnode_feat_size=vnode_feat_size,
            gcn_hidden=gcn_hidden,
            hidden_size=hidden_size,
        )

    def _embed_domains(
        self,
        domain_node_feats: List[torch.Tensor],
        domain_adj_mats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Full per-node GCN embeddings per domain — list of (N_d, gcn_hidden)."""
        return [self.gcn(X, A) for X, A in zip(domain_node_feats, domain_adj_mats)]

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vlink_feats: torch.Tensor,
        per_vnode_domain_feats: List[Tuple[torch.Tensor, ...]],
        per_vnode_domain_adjs: List[Tuple[torch.Tensor, ...]],
        per_vnode_cpu_slacks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            vnode_feats: (num_vnodes, vnode_feat_size)
            vlink_feats: (num_vlinks, vlink_feat_size)
            per_vnode_domain_feats[i]: tuple of domain X tensors allowed for vnode i
            per_vnode_domain_adjs[i]:  tuple of domain A tensors allowed for vnode i
            per_vnode_cpu_slacks[i]:   (N_total_i,) cpu_slack per snode in the flattened
                                       pool (concat of allowed domains). If None, the
                                       candidate head is skipped and the third return
                                       value is None.
        Returns:
            node_scores: (num_vnodes,)
            link_scores: (num_vlinks,)
            candidate_scores: list[(N_total_i,)] or None
        """
        num_vnodes = vnode_feats.shape[0]
        num_vlinks = vlink_feats.shape[0]

        per_vnode_pools: List[torch.Tensor] = []
        node_contexts = []
        for i in range(num_vnodes):
            full_list = self._embed_domains(
                per_vnode_domain_feats[i], per_vnode_domain_adjs[i]
            )
            pool = torch.cat(full_list, dim=0) if full_list else torch.zeros(
                0, self.gcn.W2.out_features
            )
            per_vnode_pools.append(pool)
            node_contexts.append(pool.mean(dim=0) if pool.shape[0] > 0 else torch.zeros(self.gcn.W2.out_features))

        substrate_ctx = torch.stack(node_contexts)

        node_input = torch.cat([vnode_feats, substrate_ctx], dim=1)
        node_scores = self.node_head(node_input).squeeze(-1)

        link_ctx = substrate_ctx.mean(dim=0).unsqueeze(0).expand(num_vlinks, -1)
        link_input = torch.cat([vlink_feats, link_ctx], dim=1)
        link_scores = self.link_head(link_input).squeeze(-1)

        candidate_scores: Optional[List[torch.Tensor]] = None
        if per_vnode_cpu_slacks is not None:
            candidate_scores = []
            for i in range(num_vnodes):
                pool = per_vnode_pools[i]
                slack = per_vnode_cpu_slacks[i]
                if pool.shape[0] == 0:
                    candidate_scores.append(torch.empty(0))
                    continue
                scores = self.candidate_head(vnode_feats[i], pool, slack)
                candidate_scores.append(scores)

        return node_scores, link_scores, candidate_scores
