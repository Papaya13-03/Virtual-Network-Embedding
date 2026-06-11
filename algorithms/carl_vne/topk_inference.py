"""V16 — Fix the 3 inference bottlenecks that capped V6→V14 under mp_vne.

Deep analysis revealed mp_vne wins acceptance + cost not because of better model,
but because of richer INFERENCE pipeline:

  1. PSO candidate pool: 50 per vnode (top_5 × num_domains)
     vs ours: 5 per vnode (global top_5)
     → mp_vne's PSO explores 10× wider space.

  2. Fitness function: full per-link cost
        link_cost = Σ_path (transmission_delay + bw_price × bw)
     vs ours (V6): hop-count proxy
        link_cost = len(path) × bw
     → mp_vne optimizes the right objective.

  3. bw_required in shortest_path: full bw
     vs ours (V6): min(1.0, bw × 0.1)  ← only 10% of demand
     → ours accepts mappings that fail at commit time.

V16 fixes ALL three. Same V6 weights, no retraining.
"""
from typing import Dict, List, Tuple

import torch

from algorithms.carl_vne.base_vne import (
    BaseVNE as _V6Base,
    BaseVNEDirect as _V6Direct,
    BaseVNEPSO as _V6PSO,
)
from algorithms.carl_vne.multi_restart import MultiRestartMixin


DEFAULT_PER_DOMAIN_K = 5    # candidates per (vnode, allowed_domain)


class TopKInferenceMixin:
    """Override candidate selection + fitness to match mp_vne's inference pipeline."""

    PER_DOMAIN_K = DEFAULT_PER_DOMAIN_K

    # ---- 1. Per-domain top-K candidate selection ----
    def rank_all_nn(self, vn, sample: bool = False, sample_cand: bool = False):
        # sample      → stochastic vnode/vlink ordering (Plackett-Luce).
        # sample_cand → stochastic per-domain candidate pick (Categorical for
        #               K=1) instead of argmax top-K. Enables RL exploration of
        #               candidate selection in the PSO regime. Default False so
        #               eval (rank_all_nn(sample=True)) stays deterministic
        #               (argmax per domain) — unchanged behaviour.
        node_scores, link_scores, cand_scores, cand_pools, value = self._forward_policy(vn)
        vnodes = list(vn.nodes.values())
        link_items = list(vn.links.items())

        if sample:
            ordered_vnodes, node_lp, node_ent = self._plackett_luce_sample(node_scores, vnodes)
            ordered_links, link_lp, link_ent = self._plackett_luce_sample(link_scores, link_items)
        else:
            ordered_vnodes = self._greedy_sort(node_scores, vnodes)
            ordered_links = self._greedy_sort(link_scores, link_items)
            node_lp, link_lp = [], []
            node_ent, link_ent = [], []

        # Build snode → domain_id lookup (cached on GC).
        snode_to_dom: Dict[str, str] = {}
        for lc in self.global_controller.local_controllers:
            for sn_id in lc.domain.network.nodes.keys():
                snode_to_dom[sn_id] = lc.domain.id

        orig_idx = {v.id: i for i, v in enumerate(vnodes)}
        candidate_nodes: List[List] = []
        candidate_weights: List[List[float]] = []
        cand_lp: List = []
        cand_ent: List = []

        for v in ordered_vnodes:
            i = orig_idx[v.id]
            scores_i = cand_scores[i]
            pool_i = cand_pools[i]

            # Group pool indices by domain.
            by_dom: Dict[str, List[int]] = {}
            for j, snode in enumerate(pool_i):
                d = snode_to_dom.get(snode.id, "_unknown")
                by_dom.setdefault(d, []).append(j)

            # In each domain, pick K candidates by NN score.
            combined_idx: List[int] = []
            dom_lps: List = []   # per-domain selection log-probs (sample_cand)
            dom_ents: List = []
            for dom_id, indices in by_dom.items():
                if not indices:
                    continue
                idx_t = torch.tensor(indices, dtype=torch.long)
                dom_scores = scores_i[idx_t]
                k = min(self.PER_DOMAIN_K, len(indices))
                if sample_cand:
                    # Explore: sample the per-domain candidate from softmax over
                    # ALL feasible snodes in the domain. The cand_head's ranking
                    # is what's being trained (reinforce snodes that embed well).
                    dist = torch.distributions.Categorical(logits=dom_scores)
                    if k == 1:
                        s = dist.sample()
                        dom_lps.append(dist.log_prob(s))
                        dom_ents.append(dist.entropy())
                        chosen = [int(s.item())]
                    else:
                        # Gumbel top-k (PL) sample without replacement.
                        u = torch.rand_like(dom_scores).clamp_min(1e-12)
                        gumbel = dom_scores - torch.log(-torch.log(u))
                        chosen = torch.topk(gumbel, k).indices.tolist()
                        lsm = torch.log_softmax(dom_scores, dim=0)
                        dom_lps.extend(lsm[c] for c in chosen)
                        dom_ents.append(-(lsm.exp() * lsm).sum())
                    for ti in chosen:
                        combined_idx.append(indices[ti])
                else:
                    top_in_dom = torch.topk(dom_scores, k).indices.tolist()
                    for ti in top_in_dom:
                        combined_idx.append(indices[ti])

            # Dedupe while preserving order.
            seen = set()
            picked: List[int] = []
            for j in combined_idx:
                if j not in seen:
                    picked.append(j)
                    seen.add(j)

            candidate_nodes.append([pool_i[j] for j in picked])

            # PSO weights: softmax over NN's scores for picked indices.
            if picked:
                ps = scores_i[torch.tensor(picked, dtype=torch.long)].detach()
                weights = torch.softmax(ps, dim=0).tolist()
            else:
                weights = []
            candidate_weights.append(weights)

            if sample_cand:
                cand_lp.extend(dom_lps)
                cand_ent.extend(dom_ents)
            elif sample:
                # Legacy degenerate path (argmax picks; kept for API consistency).
                if picked:
                    picked_scores = scores_i[torch.tensor(picked, dtype=torch.long)]
                    log_softmax = torch.log_softmax(picked_scores, dim=0)
                    cand_lp.extend([log_softmax[i] for i in range(len(picked))])
                    entropy = -(log_softmax.exp() * log_softmax).sum()
                    cand_ent.append(entropy)

        return ordered_vnodes, ordered_links, candidate_nodes, candidate_weights, {
            "node": node_lp, "link": link_lp, "cand": cand_lp,
        }, {
            "node": node_ent, "link": link_ent, "cand": cand_ent,
        }, value

    # ---- 2 + 3. Proper fitness: full bw + delay + bw_price ----
    def _fitness(self, particle_idx, candidates, vlink_indices, ordered_vnodes, cand_weights=None):
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float("inf")

        node_cost = sum(
            vnode.cpu_demand * snode.cpu_price
            for vnode, snode in zip(ordered_vnodes, mapping)
        )

        sorted_vlinks = sorted(vlink_indices, key=lambda x: x["bw"], reverse=True)
        link_cost = 0.0
        for vlink_info in sorted_vlinks:
            src = mapping[vlink_info["src_idx"]]
            dst = mapping[vlink_info["dst_idx"]]
            bw = vlink_info["bw"]
            # V16: FULL bw_required (was min(1.0, bw*0.1)).
            path = self.global_controller.shortest_path(src, dst, bw_required=bw)
            if not path:
                return float("inf")
            # V16: per-link cost = transmission_delay + bw_price × bw  (mp_vne formula).
            link_cost += sum(
                l.transmission_delay + l.bandwidth_price * bw
                for l in path
            )

        base = node_cost + link_cost

        # Keep policy-bias term so cand_head ranking still influences PSO.
        if cand_weights is not None:
            alpha = self.config.get("pso", {}).get("policy_bias_alpha", 50.0)
            sum_w = sum(
                cand_weights[i][idx] if idx < len(cand_weights[i]) else 0.0
                for i, idx in enumerate(particle_idx)
            )
            base -= alpha * sum_w
        return base


class TopKVNE(TopKInferenceMixin, _V6Base):
    def __init__(self):
        super().__init__()
        self.name = "CARL-VNE-TopK"


class TopKVNEDirect(TopKInferenceMixin, _V6Direct):
    def __init__(self):
        super().__init__()
        self.name = "CARL-VNE-TopK-Direct"


class TopKVNEPSO(TopKInferenceMixin, MultiRestartMixin, _V6PSO):
    """V16 PSO inference + multi-restart (inherits from V10's mixin)."""
    def __init__(self):
        super().__init__()
        self.name = "CARL-VNE-TopK-PSO"
