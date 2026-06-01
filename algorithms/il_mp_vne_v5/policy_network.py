"""V3 policy network — hierarchical multi-domain encoder + joint VN encoder.

Architecture (respects multi-domain locality):
  Level 1 (intra, per-domain):
    For each domain independently:
      GCN over the domain's substrate graph → snode embeddings.
      Max-pool over snode embeddings → domain summary.
  Level 2 (inter, over domains):
    GCN over (D, D) inter-domain adjacency on domain summaries →
    domain embeddings that are cross-domain-aware (but only via
    aggregated domain-level signals, never via individual snodes from
    other domains).

  VN side:
    GCN over VN adjacency → vnode embeddings (neighbor-aware).
    Cross-vnode self-attention → vnode embeddings (globally aware).

  Heads:
    node_head: per-vnode score = MLP([vn_embed, substrate_ctx]).
    link_head: per-vlink score = MLP([vlink_feat, src_vn_embed, dst_vn_embed]).
              (Only sees its two endpoints, not the whole VN.)
    cand_head: per-(vnode, snode) cross-attention. snode context =
              [intra_embed, inter_domain_embed] — node-local + domain-context.

  Score stabilization removed — IL needs softmax to concentrate.

Why hierarchical (multi-domain assumption):
  In realistic multi-domain VNE, a domain's individual nodes are not
  globally visible. Only domain-level aggregates + inter-domain links
  are exchanged. This architecture matches that.

  mp_vne handles inter-domain by post-hoc shortest_path at commit time;
  it has NO inter-domain representation during decision-making. v3's
  inter-domain GCN is a genuine signal mp_vne lacks.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class GCNEncoder(nn.Module):
    """GCN with LayerNorm + residual. Variable depth."""

    def __init__(self, node_feat_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_feat_size if i == 0 else hidden_size
            self.layers.append(nn.Linear(in_dim, hidden_size, bias=True))
            self.norms.append(nn.LayerNorm(hidden_size))

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = X
        for i, (W, ln) in enumerate(zip(self.layers, self.norms)):
            H_new = torch.relu(W(A_norm @ H))
            # Residual after first layer (when dims align)
            H = ln(H + H_new) if i > 0 else ln(H_new)
        return H


class VNodeSelfAttention(nn.Module):
    """Multi-head self-attention across vnodes."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, vnode_embeds: torch.Tensor) -> torch.Tensor:
        x = vnode_embeds.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x.squeeze(0)


class HierarchicalSubstrateEncoder(nn.Module):
    """Two-level substrate encoder for multi-domain VNE.

    Per-domain GCN (intra) → domain summary (pool) → inter-domain GCN
    over domain summaries. Returns BOTH levels:
      - intra_snode_embeds[d]: (N_d, intra_hidden) per domain
      - domain_embeds:         (D, inter_hidden)

    No individual node is visible from outside its domain.
    """

    def __init__(self, node_feat_size: int = 5,
                 intra_hidden: int = 32, inter_hidden: int = 32,
                 intra_layers: int = 3, inter_layers: int = 2):
        super().__init__()
        self.intra_gcn = GCNEncoder(node_feat_size, intra_hidden, num_layers=intra_layers)
        # Inter GCN's input dim = intra_hidden (domain summary)
        self.inter_gcn = GCNEncoder(intra_hidden, inter_hidden, num_layers=inter_layers)
        self.intra_hidden = intra_hidden
        self.inter_hidden = inter_hidden

    def forward(
        self,
        domain_node_feats: List[torch.Tensor],    # length D — each (N_d, F)
        domain_adj_mats: List[torch.Tensor],      # length D — each (N_d, N_d)
        inter_domain_adj: torch.Tensor,           # (D, D) normalized
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Level 1: per-domain intra GCN
        intra_snode_embeds: List[torch.Tensor] = []
        domain_summaries: List[torch.Tensor] = []
        for X, A in zip(domain_node_feats, domain_adj_mats):
            embeds = self.intra_gcn(X, A)              # (N_d, intra_hidden)
            intra_snode_embeds.append(embeds)
            # Max-pool summary — keeps strongest per-feature signal
            domain_summaries.append(embeds.max(dim=0).values)
        domain_summaries_t = torch.stack(domain_summaries)  # (D, intra_hidden)

        # Level 2: inter-domain GCN over domain summaries
        domain_embeds = self.inter_gcn(domain_summaries_t, inter_domain_adj)  # (D, inter_hidden)

        return intra_snode_embeds, domain_embeds


class JointCandidateHead(nn.Module):
    """V5 cross-attention scorer with explicit cost features (PreCost-aware).

    Per (vnode, snode) input augments the snode rep with:
      - cpu_slack (1)        : available_cpu − vnode.cpu_demand, normalized
      - cpu_cost (1)         : vnode.cpu_demand × snode.cpu_price, normalized
                               (the EXACT PreCost node-term — NN gets it
                               explicitly instead of having to relearn the
                               cpu × price product from raw features)
      - is_boundary (1)      : 1 if snode has any inter-domain link

    A learnable scalar bias on cpu_cost is added directly to the score
    (initialized NEGATIVE so higher cost → lower score, mimicking PreCost
    by default; NN learns DEVIATIONS from this prior).
    """

    def __init__(self, vnode_dim: int, snode_dim: int, hidden_size: int = 64,
                 num_heads: int = 4, aux_dim: int = 3):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.aux_dim = aux_dim
        self.query_proj = nn.Linear(vnode_dim, hidden_size, bias=True)
        self.key_proj = nn.Linear(snode_dim + aux_dim, hidden_size, bias=True)
        self.residual = nn.Sequential(
            nn.Linear(vnode_dim + snode_dim + aux_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        # PreCost-style hard prior. cost_bias initialized NEGATIVE so higher
        # cpu_cost decreases score by default.
        self.cost_bias = nn.Parameter(torch.tensor(-3.0))
        self.boundary_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        vnode_embed: torch.Tensor,    # (vnode_dim,)
        snode_embeds: torch.Tensor,   # (N, snode_dim)
        cpu_slack: torch.Tensor,      # (N,)
        cost_features: torch.Tensor,  # (N, 2) — [cpu_cost_norm, is_boundary]
    ) -> torch.Tensor:
        n = snode_embeds.shape[0]
        denom = cpu_slack.abs().max().clamp(min=1.0)
        slack_norm = (cpu_slack / denom).unsqueeze(-1)

        aux = torch.cat([slack_norm, cost_features], dim=-1)             # (N, 3)
        key_input = torch.cat([snode_embeds, aux], dim=-1)
        q = self.query_proj(vnode_embed)
        k = self.key_proj(key_input)

        q_h = q.view(self.num_heads, self.head_dim)
        k_h = k.view(n, self.num_heads, self.head_dim)
        attn_per_head = (k_h * q_h.unsqueeze(0)).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = attn_per_head.mean(dim=-1)

        q_expand = vnode_embed.unsqueeze(0).expand(n, -1)
        mlp_input = torch.cat([q_expand, snode_embeds, aux], dim=-1)
        residual = self.residual(mlp_input).squeeze(-1)

        # PreCost-style explicit prior: NN starts with mp_vne-like ranking
        # and learns deviations.
        explicit_prior = self.cost_bias * cost_features[:, 0] + self.boundary_bias * cost_features[:, 1]

        scores = attn + residual + explicit_prior
        scores = scores.masked_fill(cpu_slack < 0, -1e4)
        return scores


class PolicyNetwork(nn.Module):
    """V3 policy with hierarchical substrate encoder + joint VN encoder."""

    def __init__(
        self,
        vnode_feat_size: int = 5,
        vlink_feat_size: int = 5,
        gcn_node_feat_size: int = 5,
        gcn_hidden: int = 32,        # intra_hidden = inter_hidden
        vn_hidden: int = 32,
        hidden_size: int = 64,
        num_heads: int = 4,
        intra_layers: int = 3,
        inter_layers: int = 2,
    ):
        super().__init__()
        self.substrate_enc = HierarchicalSubstrateEncoder(
            node_feat_size=gcn_node_feat_size,
            intra_hidden=gcn_hidden,
            inter_hidden=gcn_hidden,
            intra_layers=intra_layers,
            inter_layers=inter_layers,
        )
        self.vn_gcn = GCNEncoder(vnode_feat_size, vn_hidden, num_layers=2)
        self.vnode_attn = VNodeSelfAttention(vn_hidden, num_heads=num_heads)

        # Cand head's snode_dim = intra_hidden + inter_hidden
        snode_ctx_dim = gcn_hidden * 2
        self.candidate_head = JointCandidateHead(
            vnode_dim=vn_hidden,
            snode_dim=snode_ctx_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        # node_head: vn_embed + max-pool of substrate context for this vnode
        node_input_size = vn_hidden + snode_ctx_dim
        self.node_head = nn.Sequential(
            nn.Linear(node_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # link_head: vlink feats + src/dst vnode embeddings (only its endpoints)
        link_input_size = vlink_feat_size + 2 * vn_hidden
        self.link_head = nn.Sequential(
            nn.Linear(link_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        value_input_size = snode_ctx_dim + vn_hidden
        self.value_head = nn.Sequential(
            nn.Linear(value_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        vnode_feats: torch.Tensor,                                # (V, vnode_feat_size)
        vlink_feats: torch.Tensor,                                # (E_vn, vlink_feat_size)
        vn_adj: torch.Tensor,                                     # (V, V)
        vlink_endpoints: torch.Tensor,                            # (E_vn, 2) long
        all_domain_feats: List[torch.Tensor],                     # length D — (N_d, F) each
        all_domain_adjs: List[torch.Tensor],                      # length D
        inter_domain_adj: torch.Tensor,                           # (D, D) normalized
        per_vnode_allowed_domain_idx: List[List[int]],            # length V — domain indices allowed per vnode
        per_vnode_cpu_slacks: Optional[List[torch.Tensor]] = None,
        per_vnode_cost_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        num_vnodes = vnode_feats.shape[0]
        num_vlinks = vlink_feats.shape[0]

        # 1. Hierarchical substrate encoding
        intra_snode_embeds, domain_embeds = self.substrate_enc(
            all_domain_feats, all_domain_adjs, inter_domain_adj,
        )
        # intra_snode_embeds: List[(N_d, intra_hidden)]
        # domain_embeds: (D, inter_hidden)

        # 2. Build per-vnode snode pool: concat allowed domains' [intra | inter] features
        per_vnode_pools: List[torch.Tensor] = []
        per_vnode_ctx: List[torch.Tensor] = []
        for i, allowed_idx in enumerate(per_vnode_allowed_domain_idx):
            if not allowed_idx:
                per_vnode_pools.append(torch.empty(0, 2 * self.substrate_enc.intra_hidden))
                per_vnode_ctx.append(torch.zeros(2 * self.substrate_enc.intra_hidden))
                continue
            parts = []
            for d in allowed_idx:
                intra = intra_snode_embeds[d]                         # (N_d, intra_hidden)
                inter = domain_embeds[d].unsqueeze(0).expand(intra.shape[0], -1)  # (N_d, inter_hidden)
                parts.append(torch.cat([intra, inter], dim=-1))       # (N_d, 2*hidden)
            pool = torch.cat(parts, dim=0)                            # (N_allowed_i, 2*hidden)
            per_vnode_pools.append(pool)
            per_vnode_ctx.append(pool.max(dim=0).values)

        substrate_ctx = torch.stack(per_vnode_ctx)                    # (V, 2*hidden)

        # 3. VN encoding: GCN + cross-vnode self-attention
        vn_embeds = self.vn_gcn(vnode_feats, vn_adj)                  # (V, vn_hidden)
        vn_embeds = self.vnode_attn(vn_embeds)                        # (V, vn_hidden)

        # 4. node_head: per-vnode score
        node_input = torch.cat([vn_embeds, substrate_ctx], dim=1)
        node_scores = self.node_head(node_input).squeeze(-1)

        # 5. link_head: per-vlink, sees only its endpoints
        if num_vlinks > 0:
            src_emb = vn_embeds[vlink_endpoints[:, 0]]
            dst_emb = vn_embeds[vlink_endpoints[:, 1]]
            link_input = torch.cat([vlink_feats, src_emb, dst_emb], dim=1)
            link_scores = self.link_head(link_input).squeeze(-1)
        else:
            link_scores = torch.zeros(0)

        # 6. Joint cand_head — V5 needs cost_features too
        candidate_scores: Optional[List[torch.Tensor]] = None
        if per_vnode_cpu_slacks is not None:
            candidate_scores = []
            for i in range(num_vnodes):
                pool = per_vnode_pools[i]
                slack = per_vnode_cpu_slacks[i]
                if pool.shape[0] == 0:
                    candidate_scores.append(torch.empty(0))
                    continue
                cost_feat = (per_vnode_cost_features[i]
                             if per_vnode_cost_features is not None
                             else torch.zeros(pool.shape[0], 2))
                scores = self.candidate_head(vn_embeds[i], pool, slack, cost_feat)
                candidate_scores.append(scores)

        # 7. Value head
        substrate_pool = substrate_ctx.max(dim=0).values
        vn_pool = vn_embeds.max(dim=0).values
        state_repr = torch.cat([substrate_pool, vn_pool], dim=-1)
        value = self.value_head(state_repr).squeeze(-1)

        return node_scores, link_scores, candidate_scores, value
