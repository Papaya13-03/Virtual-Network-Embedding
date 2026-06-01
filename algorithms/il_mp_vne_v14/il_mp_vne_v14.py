"""V14 — Hybrid Top-K candidates at inference.

Insight: V10 wins rev/cost vs mp_vne (+7%) but LOSES acceptance (28.9% vs 32.8%).
Cause: NN's top-K candidates sometimes miss feasible snodes that PreCost would
pick. mp_vne's PreCost-based candidates have broader feasibility coverage.

Fix: at inference, AUGMENT NN's top-K with PreCost top-K candidates per vnode.
PSO then searches in this larger union. Best of both worlds:
  - Acceptance ≥ mp_vne (PreCost-feasible snodes guaranteed present)
  - Cost ≤ V10 (NN's good candidates still preferred)

V14 inherits V10 (V6 model + multi-restart PSO). Only override is candidate
generation: union(NN_top_K, PreCost_top_K) → wider pool for PSO.
"""
import torch
from typing import List, Tuple, Dict

from algorithms.il_mp_vne_v10.il_mp_vne_v10 import (
    ILMPVNEV10 as _V10Base,
    ILMPVNEV10Direct as _V10Direct,
    ILMPVNEV10PSO as _V10PSO,
)


# Tunable: K for NN top, K for PreCost top.
DEFAULT_K_NN = 5
DEFAULT_K_PRECOST = 5


class _HybridTopKMixin:
    """Override `rank_all_nn` so candidates_per_vnode = union(NN_top, PreCost_top)."""

    K_NN = DEFAULT_K_NN
    K_PRECOST = DEFAULT_K_PRECOST

    def _precost_topk(self, vnode, pool, k):
        """Return indices into pool of k cheapest snodes by PreCost node term:
          cpu_demand × cpu_price (PreCost node-cost, no link term).
        Infeasible snodes (insufficient CPU) excluded.
        """
        costs = []
        for i, snode in enumerate(pool):
            avail = getattr(snode, "available_cpu", snode.cpu_capacity)
            if avail < vnode.cpu_demand:
                continue
            cost = vnode.cpu_demand * snode.cpu_price
            costs.append((cost, i))
        costs.sort(key=lambda x: x[0])
        return [i for _, i in costs[:k]]

    def rank_all_nn(self, vn, sample: bool = False):
        # Run forward pass like parent (V10 inherits V6's rank_all_nn).
        node_scores, link_scores, cand_scores, cand_pools, value = self._forward_policy(vn)
        vnodes = list(vn.nodes.values())
        link_items = list(vn.links.items())

        # Ordering (same as parent): Plackett-Luce sample or greedy.
        if sample:
            ordered_vnodes, node_lp, node_ent = self._plackett_luce_sample(node_scores, vnodes)
            ordered_links, link_lp, link_ent = self._plackett_luce_sample(link_scores, link_items)
        else:
            ordered_vnodes = self._greedy_sort(node_scores, vnodes)
            ordered_links = self._greedy_sort(link_scores, link_items)
            node_lp, link_lp = [], []
            node_ent, link_ent = [], []

        orig_idx = {v.id: i for i, v in enumerate(vnodes)}
        candidate_nodes: List[List] = []
        candidate_weights: List[List[float]] = []
        cand_lp: List = []
        cand_ent: List = []

        for v in ordered_vnodes:
            i = orig_idx[v.id]
            scores_i = cand_scores[i]
            pool_i = cand_pools[i]

            # NN top-K (sampled or greedy)
            if sample:
                picked_nn, lps, ents = self._plackett_luce_topk(scores_i, self.K_NN)
                cand_lp.extend(lps)
                cand_ent.extend(ents)
            else:
                picked_nn = self._topk_greedy(scores_i, self.K_NN)

            # PreCost top-K (greedy by cpu_demand × cpu_price)
            picked_precost = self._precost_topk(v, pool_i, self.K_PRECOST)

            # Union (preserve NN's ordering first, then add PreCost-only)
            seen = set(picked_nn)
            combined = list(picked_nn)
            for idx in picked_precost:
                if idx not in seen:
                    combined.append(idx)
                    seen.add(idx)

            candidate_nodes.append([pool_i[j] for j in combined])

            # Weights: softmax of NN's scores for combined picks (PreCost-only
            # picks get the NN score they happen to have — lower than top-K).
            if combined:
                picked_scores = scores_i[torch.tensor(combined, dtype=torch.long)].detach()
                weights = torch.softmax(picked_scores, dim=0).tolist()
            else:
                weights = []
            candidate_weights.append(weights)

        return ordered_vnodes, ordered_links, candidate_nodes, candidate_weights, {
            "node": node_lp, "link": link_lp, "cand": cand_lp,
        }, {
            "node": node_ent, "link": link_ent, "cand": cand_ent,
        }, value


class ILMPVNEV14(_HybridTopKMixin, _V10Base):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V14"


class ILMPVNEV14Direct(_HybridTopKMixin, _V10Direct):
    """Direct mode doesn't use candidates_list — hybrid is a no-op here."""
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V14-Direct"


class ILMPVNEV14PSO(_HybridTopKMixin, _V10PSO):
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V14-PSO"
