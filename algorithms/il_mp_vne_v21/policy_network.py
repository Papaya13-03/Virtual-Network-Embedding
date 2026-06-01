"""V21 policy — strong attention-based encoder, no ordering heads.

Drops:
  - node_head and link_head (PSO is order-invariant; proven by code reading +
    flat ordering-RL experiment).
  - Plackett-Luce sampling at rollout (no learnable ordering).

Reinvests capacity in:
  - GAT-style attention aggregation with edge features (delay, bw_price, etc.)
    so the encoder actually uses the link information that vanilla GCN's scalar
    edge-weight ignored.
  - Attention-pool replacing max-pool for domain summaries (the V6 bottleneck
    that collapses per-domain detail).
  - Multi-layer cross-attention candidate head (vnode query, snode key/value)
    with explicit cost-bias priors retained from V6.
  - Designed N-agnostic — same network works at 100, 200, 500 substrate nodes.

Forward signature MATCHES V6's PolicyNetwork so ILMPVNEV6's `_forward_policy`
can be inherited unchanged. node_scores/link_scores are returned as zeros for
API compatibility (the V6 algorithm code expects 4-tuple).
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class GATBlock(nn.Module):
    """Multi-head graph attention with optional edge features, pre-LayerNorm,
    residual. Aggregation: a_ij = softmax_j( edge_score(i, j) ) masked by A.

    For N-agnostic scaling, this operates on (N, D) tensors with dense
    (N, N) adjacency mask — fine for N ≤ ~200 per pass.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int = 4, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.edge_dim = edge_dim

        self.norm = nn.LayerNorm(in_dim)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention scoring vectors per head (GAT paper Eq. 3).
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        if edge_dim > 0:
            self.W_e = nn.Linear(edge_dim, out_dim, bias=False)
            self.a_edge = nn.Parameter(torch.empty(num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.a_edge)
        else:
            self.W_e = None
            self.a_edge = None

        # Residual projection if dims change.
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, X: torch.Tensor, A_mask: torch.Tensor,
                E: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X: (N, in_dim). A_mask: (N, N) {0/1 or float weight, >0 means edge}.
        E: (N, N, edge_dim) edge features, or None.
        Returns: (N, out_dim).
        """
        if X.shape[0] == 0:
            return X.new_zeros(0, self.head_dim * self.num_heads)

        X_norm = self.norm(X)
        Wh = self.W(X_norm).view(-1, self.num_heads, self.head_dim)  # (N, H, D)

        # Per-head scalar contribution from src/dst.
        src_score = (Wh * self.a_src.unsqueeze(0)).sum(-1)   # (N, H)
        dst_score = (Wh * self.a_dst.unsqueeze(0)).sum(-1)   # (N, H)
        # Edge logit: e_ij = src_i + dst_j (+ edge_term).  Shape (N, N, H).
        logits = src_score.unsqueeze(1) + dst_score.unsqueeze(0)

        if E is not None and self.W_e is not None:
            We = self.W_e(E).view(E.shape[0], E.shape[1], self.num_heads, self.head_dim)
            edge_score = (We * self.a_edge.view(1, 1, self.num_heads, self.head_dim)).sum(-1)
            logits = logits + edge_score

        logits = F.leaky_relu(logits, negative_slope=0.2)
        mask = (A_mask > 0).unsqueeze(-1).expand_as(logits)
        logits = logits.masked_fill(~mask, float("-inf"))
        # Guard against rows with all-masked (no edges): set to uniform self-loop.
        all_masked = (~mask).all(dim=1, keepdim=True).expand_as(logits)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)

        attn = torch.softmax(logits, dim=1)                  # softmax over j (sources for i)
        attn = self.dropout(attn)
        # Aggregate: out_i = Σ_j attn[i, j, h] * Wh[j, h, :]
        out = torch.einsum("ijh,jhd->ihd", attn, Wh)          # (N, H, D)
        out = out.reshape(-1, self.num_heads * self.head_dim)  # (N, out_dim)

        return self.res_proj(X) + out


class AttentionPool(nn.Module):
    """Set-Transformer-style PMA: a learnable seed query attends over the set
    → 1 vector. Size-invariant pool that preserves more info than max."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, dim) → (dim,)."""
        if X.shape[0] == 0:
            return self.seed.squeeze(0).detach().clone()
        x = self.norm(X).unsqueeze(0)               # (1, N, D)
        q = self.seed.unsqueeze(0)                  # (1, 1, D)
        out, _ = self.attn(q, x, x)                 # (1, 1, D)
        out = out.squeeze(0).squeeze(0)
        return out + self.ffn(out)


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


class EnhancedSubstrateEncoder(nn.Module):
    """Two-level multi-domain encoder with GAT + attention-pool.

    Intra: 4-layer GAT (with edge features) per domain → (N_d, hidden) snode
    embeddings.
    Pool: attention-pool over each domain's snodes → (hidden,) domain summary.
    Inter: 3-layer GAT on (D, hidden) domain summaries with inter-domain edge
    features → (D, hidden) cross-domain-aware domain embeddings.
    """

    def __init__(self, node_feat_size: int = 7, hidden: int = 64,
                 num_heads: int = 4,
                 intra_edge_dim: int = 1, inter_edge_dim: int = 1,
                 intra_layers: int = 4, inter_layers: int = 3):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(node_feat_size, hidden)
        self.intra_layers = nn.ModuleList([
            GATBlock(hidden, hidden, num_heads=num_heads, edge_dim=intra_edge_dim)
            for _ in range(intra_layers)
        ])
        self.intra_norm = nn.LayerNorm(hidden)
        self.pool = AttentionPool(hidden, num_heads=num_heads)
        self.inter_layers = nn.ModuleList([
            GATBlock(hidden, hidden, num_heads=num_heads, edge_dim=inter_edge_dim)
            for _ in range(inter_layers)
        ])
        self.inter_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        domain_node_feats: List[torch.Tensor],   # length D, each (N_d, F)
        domain_adj_mats: List[torch.Tensor],     # length D, each (N_d, N_d)
        inter_domain_adj: torch.Tensor,          # (D, D) — used as both mask and edge weight
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        intra_snode_embeds: List[torch.Tensor] = []
        domain_summaries: List[torch.Tensor] = []
        for X, A in zip(domain_node_feats, domain_adj_mats):
            H = self.input_proj(X)
            # Edge feature = A weight (single scalar — could expand later).
            E = A.unsqueeze(-1) if A.numel() > 0 else None
            for layer in self.intra_layers:
                H = layer(H, A, E)
            H = self.intra_norm(H)
            intra_snode_embeds.append(H)
            domain_summaries.append(self.pool(H))
        D_emb = torch.stack(domain_summaries)                # (D, hidden)
        E_inter = inter_domain_adj.unsqueeze(-1)
        for layer in self.inter_layers:
            D_emb = layer(D_emb, inter_domain_adj, E_inter)
        D_emb = self.inter_norm(D_emb)
        return intra_snode_embeds, D_emb


class EnhancedVNEncoder(nn.Module):
    """VN encoder = GAT (with edge features = vlink_feats) + 1 transformer
    block over vnodes (global cross-vnode attention)."""

    def __init__(self, vnode_feat_size: int = 7, vlink_feat_size: int = 5,
                 hidden: int = 64, num_heads: int = 4, gat_layers: int = 3):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(vnode_feat_size, hidden)
        self.gat_layers = nn.ModuleList([
            GATBlock(hidden, hidden, num_heads=num_heads, edge_dim=vlink_feat_size)
            for _ in range(gat_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.global_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden),
        )

    def forward(self, vnode_feats: torch.Tensor, vlink_feats: torch.Tensor,
                vn_adj: torch.Tensor, vlink_endpoints: torch.Tensor) -> torch.Tensor:
        V = vnode_feats.shape[0]
        H = self.input_proj(vnode_feats)

        # Build edge-feature tensor (V, V, vlink_feat_size). Default zeros.
        if vlink_endpoints.numel() > 0:
            E = vn_adj.new_zeros(V, V, vlink_feats.shape[1]) if vlink_feats.numel() > 0 else None
            if E is not None:
                # Symmetric assignment (undirected).
                src = vlink_endpoints[:, 0]
                dst = vlink_endpoints[:, 1]
                E[src, dst] = vlink_feats
                E[dst, src] = vlink_feats
        else:
            E = None

        for layer in self.gat_layers:
            H = layer(H, vn_adj, E)
        H = self.norm(H)

        # Global vnode self-attention (no mask).
        x = H.unsqueeze(0)
        attn_out, _ = self.global_attn(x, x, x)
        H = (x + attn_out).squeeze(0)
        H = H + self.ffn(self.ffn_norm(H))
        return H


# --------------------------------------------------------------------------
# Candidate head
# --------------------------------------------------------------------------


class MultiLayerCrossAttnCandHead(nn.Module):
    """Per-snode scorer. Snodes act as queries (distinct per position) and
    iteratively refine via:
      - self-attention among snodes (so they compete/cooperate),
      - cross-attention to the vnode context token,
      - FFN.
    Final per-snode hidden → scalar logit + V6-style cost-bias priors.

    Earlier version had the broadcast-vnode-as-query bug (all P queries
    identical → all attn outputs identical → uniform logits across the pool,
    so cand_head provided no signal). The fix is to project snode features as
    the per-position query while keeping vnode_embed as the cross-attention
    context.
    """

    def __init__(self, vnode_dim: int, snode_dim: int,
                 hidden: int = 64, num_heads: int = 4,
                 num_layers: int = 2, aux_dim: int = 4):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.aux_dim = aux_dim  # [cpu_slack, cpu_cost, is_boundary, link_cost]

        # Project vnode embed → 1 context token used by every cross-attn layer.
        self.vnode_proj = nn.Linear(vnode_dim, hidden)
        # Project per-snode features → initial per-snode query.
        self.snode_proj = nn.Linear(snode_dim + aux_dim, hidden)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "ln_self": nn.LayerNorm(hidden),
                "self_attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                "ln_cross": nn.LayerNorm(hidden),
                "cross_attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                "ln_ffn": nn.LayerNorm(hidden),
                "ffn": nn.Sequential(
                    nn.Linear(hidden, hidden * 2), nn.GELU(),
                    nn.Linear(hidden * 2, hidden),
                ),
            }))
        # Bilinear shortcut: dot product between vnode_proj and each snode_proj
        # (mirrors V6's design — gives a strong per-snode signal even before
        # the heavy multi-layer head learns anything).
        self.bilinear_scale = (hidden ** -0.5)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # PreCost-equivalent explicit priors (kept from V6).
        self.cost_bias = nn.Parameter(torch.tensor(-3.0))
        self.boundary_bias = nn.Parameter(torch.tensor(0.0))
        self.link_cost_bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, vnode_embed: torch.Tensor, snode_pool: torch.Tensor,
                cpu_slack: Optional[torch.Tensor],
                cost_feat: Optional[torch.Tensor]) -> torch.Tensor:
        P = snode_pool.shape[0]
        if P == 0:
            return snode_pool.new_zeros(0)

        if cpu_slack is None:
            cpu_slack = snode_pool.new_zeros(P, 1)
        elif cpu_slack.dim() == 1:
            cpu_slack = cpu_slack.unsqueeze(-1)
        if cost_feat is None:
            cost_feat = snode_pool.new_zeros(P, 3)

        # Normalise cpu_slack (V6 trick: prevent scale blow-up).
        denom = cpu_slack.abs().max().clamp(min=1.0)
        slack_norm = cpu_slack / denom
        aux = torch.cat([slack_norm, cost_feat], dim=-1)        # (P, 4)
        snode_full = torch.cat([snode_pool, aux], dim=-1)       # (P, snode_dim+4)

        # Initial per-snode rep (DISTINCT per position) and context token.
        snode_h = self.snode_proj(snode_full).unsqueeze(0)      # (1, P, hidden)
        ctx_token = self.vnode_proj(vnode_embed).view(1, 1, -1)  # (1, 1, hidden)

        # Bilinear shortcut (q·k): a strong head-aligned per-snode signal.
        q = ctx_token.squeeze(0).squeeze(0)                     # (hidden,)
        k_proj = snode_h.squeeze(0)                             # (P, hidden)
        bilinear_score = (k_proj * q.unsqueeze(0)).sum(-1) * self.bilinear_scale  # (P,)

        # Multi-layer refinement: self-attn among snodes + cross-attn to vnode.
        for layer in self.layers:
            x = layer["ln_self"](snode_h)
            self_out, _ = layer["self_attn"](x, x, x)
            snode_h = snode_h + self_out

            x = layer["ln_cross"](snode_h)
            cross_out, _ = layer["cross_attn"](x, ctx_token, ctx_token)
            snode_h = snode_h + cross_out

            snode_h = snode_h + layer["ffn"](layer["ln_ffn"](snode_h))

        h = snode_h.squeeze(0)                                  # (P, hidden)
        residual_logits = self.score_head(h).squeeze(-1)        # (P,)

        # Explicit PreCost-style prior (cost down-weighting).
        cpu_cost = cost_feat[:, 0]
        is_boundary = cost_feat[:, 1]
        link_cost = cost_feat[:, 2]
        prior = (self.cost_bias * cpu_cost
                 + self.boundary_bias * is_boundary
                 + self.link_cost_bias * link_cost)

        scores = bilinear_score + residual_logits + prior
        # Mask infeasible snodes (V6 trick).
        slack_1d = cpu_slack.squeeze(-1) if cpu_slack.dim() > 1 else cpu_slack
        scores = scores.masked_fill(slack_1d < 0, -1e4)
        return scores


# --------------------------------------------------------------------------
# Full policy
# --------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """V21 policy. Forward signature matches V6's so ILMPVNEV6._forward_policy
    is reusable. node_scores and link_scores are returned as zeros (unused —
    PSO is order-invariant and direct-decoding uses original VN order)."""

    def __init__(self, vnode_feat_size: int = 7, vlink_feat_size: int = 5,
                 gcn_node_feat_size: int = 7,
                 hidden: int = 64, num_heads: int = 4,
                 intra_layers: int = 4, inter_layers: int = 3,
                 vn_gat_layers: int = 3, cand_layers: int = 2,
                 # Compat kwargs (accepted, ignored — V6 init passes these).
                 gcn_hidden: Optional[int] = None,
                 hidden_size: Optional[int] = None):
        super().__init__()
        # Honour legacy kw if caller passes hidden_size.
        if hidden_size is not None:
            hidden = hidden_size
        self.hidden = hidden
        self.substrate_enc = EnhancedSubstrateEncoder(
            node_feat_size=gcn_node_feat_size, hidden=hidden, num_heads=num_heads,
            intra_layers=intra_layers, inter_layers=inter_layers,
        )
        self.vn_enc = EnhancedVNEncoder(
            vnode_feat_size=vnode_feat_size, vlink_feat_size=vlink_feat_size,
            hidden=hidden, num_heads=num_heads, gat_layers=vn_gat_layers,
        )
        # snode context per pool = [intra_embed | domain_embed] = 2 * hidden.
        self.cand_head = MultiLayerCrossAttnCandHead(
            vnode_dim=hidden, snode_dim=2 * hidden,
            hidden=hidden, num_heads=num_heads, num_layers=cand_layers,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        vnode_feats: torch.Tensor,
        vlink_feats: torch.Tensor,
        vn_adj: torch.Tensor,
        vlink_endpoints: torch.Tensor,
        all_domain_feats: List[torch.Tensor],
        all_domain_adjs: List[torch.Tensor],
        inter_domain_adj: torch.Tensor,
        per_vnode_allowed_domain_idx: List[List[int]],
        per_vnode_cpu_slacks: Optional[List[torch.Tensor]] = None,
        per_vnode_cost_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        V = vnode_feats.shape[0]
        E = vlink_feats.shape[0]

        # 1. Substrate encoding (intra GAT → attention-pool → inter GAT).
        intra_snode_embeds, domain_embeds = self.substrate_enc(
            all_domain_feats, all_domain_adjs, inter_domain_adj,
        )

        # 2. Per-vnode candidate pool = concat allowed domains' [intra | domain].
        per_vnode_pools: List[torch.Tensor] = []
        per_vnode_ctx: List[torch.Tensor] = []
        for allowed in per_vnode_allowed_domain_idx:
            if not allowed:
                per_vnode_pools.append(vnode_feats.new_zeros(0, 2 * self.hidden))
                per_vnode_ctx.append(vnode_feats.new_zeros(2 * self.hidden))
                continue
            parts = []
            for d in allowed:
                intra = intra_snode_embeds[d]
                inter = domain_embeds[d].unsqueeze(0).expand(intra.shape[0], -1)
                parts.append(torch.cat([intra, inter], dim=-1))
            pool = torch.cat(parts, dim=0)
            per_vnode_pools.append(pool)
            per_vnode_ctx.append(pool.max(dim=0).values)
        substrate_ctx = torch.stack(per_vnode_ctx)              # (V, 2*hidden)

        # 3. VN encoding.
        vn_embeds = self.vn_enc(vnode_feats, vlink_feats, vn_adj, vlink_endpoints)

        # 4. node_scores / link_scores — unused; zeros for API compatibility.
        node_scores = vnode_feats.new_zeros(V)
        link_scores = vlink_feats.new_zeros(E) if E > 0 else vlink_feats.new_zeros(0)

        # 5. Candidate scoring via multi-layer cross-attention.
        candidate_scores: Optional[List[torch.Tensor]] = None
        if per_vnode_cpu_slacks is not None:
            candidate_scores = []
            for i in range(V):
                pool = per_vnode_pools[i]
                slack = per_vnode_cpu_slacks[i] if i < len(per_vnode_cpu_slacks) else None
                cost = (per_vnode_cost_features[i] if per_vnode_cost_features is not None
                        and i < len(per_vnode_cost_features) else None)
                if pool.shape[0] == 0:
                    candidate_scores.append(vnode_feats.new_zeros(0))
                    continue
                candidate_scores.append(
                    self.cand_head(vn_embeds[i], pool, slack, cost)
                )

        # 6. Value (critic).
        substrate_pool = substrate_ctx.max(dim=0).values        # (2*hidden,)
        # Use vn pool of size hidden too — combine with substrate_pool[:hidden]
        # to keep value_head input = 2*hidden.
        vn_pool = vn_embeds.max(dim=0).values                   # (hidden,)
        state_repr = torch.cat([substrate_pool[:self.hidden], vn_pool], dim=-1)  # (2*hidden,)
        value = self.value_head(state_repr).squeeze(-1)

        return node_scores, link_scores, candidate_scores, value
