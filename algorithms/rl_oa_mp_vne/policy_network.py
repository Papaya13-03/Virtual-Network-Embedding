import torch
import torch.nn as nn
from typing import List, Tuple


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


class PolicyNetwork(nn.Module):
    """
    Policy network for ranking virtual nodes and links.
    Uses a shared GCN to embed substrate domains, then concatenates
    the aggregated substrate embedding with virtual features to produce
    priority scores for vnodes and vlinks.
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

    def _embed_domains(
        self,
        domain_node_feats: List[torch.Tensor],
        domain_adj_mats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run GCN on each domain, return list of domain embeddings (gcn_hidden,)."""
        embeddings = []
        for X, A in zip(domain_node_feats, domain_adj_mats):
            H = self.gcn(X, A)
            embed = H.mean(dim=0)
            embeddings.append(embed)
        return embeddings

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vlink_feats: torch.Tensor,
        per_vnode_domain_feats: List[Tuple[torch.Tensor, ...]],
        per_vnode_domain_adjs: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vnode_feats: (num_vnodes, vnode_feat_size)
            vlink_feats: (num_vlinks, vlink_feat_size)
            per_vnode_domain_feats: For each vnode, tuple of domain X tensors
            per_vnode_domain_adjs: For each vnode, tuple of domain A tensors
        Returns:
            node_scores: (num_vnodes,)
            link_scores: (num_vlinks,)
        """
        num_vnodes = vnode_feats.shape[0]
        num_vlinks = vlink_feats.shape[0]

        node_contexts = []
        for i in range(num_vnodes):
            domain_Xs = per_vnode_domain_feats[i]
            domain_As = per_vnode_domain_adjs[i]
            domain_embeds = self._embed_domains(domain_Xs, domain_As)
            ctx = torch.stack(domain_embeds).mean(dim=0)
            node_contexts.append(ctx)
        substrate_ctx = torch.stack(node_contexts)

        node_input = torch.cat([vnode_feats, substrate_ctx], dim=1)
        node_scores = self.node_head(node_input).squeeze(-1)

        link_ctx = substrate_ctx.mean(dim=0).unsqueeze(0).expand(num_vlinks, -1)
        link_input = torch.cat([vlink_feats, link_ctx], dim=1)
        link_scores = self.link_head(link_input).squeeze(-1)

        return node_scores, link_scores
