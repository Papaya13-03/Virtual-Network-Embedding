"""V21 — cand-focused architecture (no ordering heads), GAT encoder, multi-
layer cross-attention cand head. Inherits ILMPVNEV6 plumbing (PSO, commit,
controller) but swaps in V21Policy and overrides rank_all_nn / rank_direct to
use the VN's original node/link order (PSO is order-invariant; direct-decode
no longer learns ordering since node/link heads are gone).

PSO inference uses per-domain top-1 (K=1) — same regime as V17/V19. The
candidate head is the only learning target; train via the V19 recipe:
imitation_pretrain.py for IL + ppo_finetune.py --rollout direct --target cand
for RL fine-tune.
"""
from typing import Dict, List, Optional, Tuple

import torch

from algorithms.il_mp_vne_v6.il_mp_vne_v6 import ILMPVNEV6
from algorithms.il_mp_vne_v21.policy_network import PolicyNetwork as V21Policy
from problem.virtual_network import VirtualNetwork


class ILMPVNEV21(ILMPVNEV6):
    """V21 base. Inherits V6 plumbing; replaces policy with V21Policy and
    drops PL ordering. Use ILMPVNEV21PSO for PSO inference or
    ILMPVNEV21Direct for autoregressive cand decoding."""

    PER_DOMAIN_K = 1     # match V17/V19 regime for fair comparison

    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V21"
        # Swap in V21 policy (V6 init created a V6 PolicyNetwork; replace it).
        pn_cfg = self.config.get("policy_network", {})
        self.policy = V21Policy(
            vnode_feat_size=7,
            vlink_feat_size=5,
            gcn_node_feat_size=7,
            hidden=pn_cfg.get("v21_hidden", 64),
            num_heads=pn_cfg.get("v21_num_heads", 4),
            intra_layers=pn_cfg.get("v21_intra_layers", 4),
            inter_layers=pn_cfg.get("v21_inter_layers", 3),
            vn_gat_layers=pn_cfg.get("v21_vn_layers", 3),
            cand_layers=pn_cfg.get("v21_cand_layers", 2),
        )
        # Rebind trainer to the new policy so any solve()-time online code
        # references the right params (no eval/PPO path uses trainer.step,
        # so this is just structural cleanliness).
        try:
            self.trainer.policy = self.policy
            self.trainer.optimizer = type(self.trainer.optimizer)(
                self.policy.parameters(),
                lr=self.trainer.optimizer.param_groups[0]["lr"],
            )
        except Exception:
            pass
        self._pretrained = False  # any prior V6 auto-load is moot.

    # ------------------------------------------------------------------
    # Helper: snode_id → domain_id lookup (cached on the global controller).
    # ------------------------------------------------------------------

    def _snode_to_domain(self) -> Dict[str, str]:
        cache_key = "_v21_snode_to_dom"
        cached = getattr(self.global_controller, cache_key, None)
        if cached is not None:
            return cached
        mapping: Dict[str, str] = {}
        for lc in self.global_controller.local_controllers:
            for sn_id in lc.domain.network.nodes.keys():
                mapping[sn_id] = lc.domain.id
        setattr(self.global_controller, cache_key, mapping)
        return mapping

    # ------------------------------------------------------------------
    # rank_all_nn — PSO path. Original VN order + per-domain top-K
    # (optionally SAMPLED for RL exploration via sample_cand).
    # ------------------------------------------------------------------

    def rank_all_nn(self, vn: VirtualNetwork, sample: bool = False,
                    sample_cand: bool = False):
        node_scores, link_scores, cand_scores, cand_pools, value = \
            self._forward_policy(vn)
        # Original VN order — node/link heads are absent so any ordering signal
        # would be from zero scores. PSO is order-invariant for its mapping,
        # so this is the correct, deterministic choice.
        ordered_vnodes = list(vn.nodes.values())
        ordered_links = list(vn.links.items())
        # Empty log_probs/entropies for ordering — there is no ordering policy.
        node_lp: List[torch.Tensor] = []
        link_lp: List[torch.Tensor] = []
        node_ent: List[torch.Tensor] = []
        link_ent: List[torch.Tensor] = []

        snode_to_dom = self._snode_to_domain()
        orig_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
        candidate_nodes: List[List] = []
        candidate_weights: List[List[float]] = []
        cand_lp: List[torch.Tensor] = []
        cand_ent: List[torch.Tensor] = []

        for v in ordered_vnodes:
            i = orig_idx[v.id]
            scores_i = cand_scores[i]
            pool_i = cand_pools[i]

            # Group by domain.
            by_dom: Dict[str, List[int]] = {}
            for j, snode in enumerate(pool_i):
                by_dom.setdefault(snode_to_dom.get(snode.id, "_unknown"), []).append(j)

            combined_idx: List[int] = []
            dom_lps: List[torch.Tensor] = []
            dom_ents: List[torch.Tensor] = []
            for indices in by_dom.values():
                if not indices:
                    continue
                idx_t = torch.tensor(indices, dtype=torch.long)
                dom_scores = scores_i[idx_t]
                k = min(self.PER_DOMAIN_K, len(indices))
                if sample_cand:
                    dist = torch.distributions.Categorical(logits=dom_scores)
                    if k == 1:
                        s = dist.sample()
                        dom_lps.append(dist.log_prob(s))
                        dom_ents.append(dist.entropy())
                        chosen = [int(s.item())]
                    else:
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

            seen = set()
            picked: List[int] = []
            for j in combined_idx:
                if j not in seen:
                    picked.append(j)
                    seen.add(j)
            candidate_nodes.append([pool_i[j] for j in picked])

            if picked:
                ps = scores_i[torch.tensor(picked, dtype=torch.long)].detach()
                candidate_weights.append(torch.softmax(ps, dim=0).tolist())
            else:
                candidate_weights.append([])

            if sample_cand:
                cand_lp.extend(dom_lps)
                cand_ent.extend(dom_ents)

        return (
            ordered_vnodes, ordered_links, candidate_nodes, candidate_weights,
            {"node": node_lp, "link": link_lp, "cand": cand_lp},
            {"node": node_ent, "link": link_ent, "cand": cand_ent},
            value,
        )

    # ------------------------------------------------------------------
    # rank_direct — direct autoregressive cand decoding. Original VN order.
    # ------------------------------------------------------------------

    def rank_direct(self, vn: VirtualNetwork, sample: bool = False,
                    sample_order: Optional[bool] = None,
                    sample_cand: Optional[bool] = None):
        # sample_order is ignored (no ordering policy). sample_cand defaults to
        # `sample` if not given (V6 contract).
        if sample_cand is None:
            sample_cand = sample

        node_scores, link_scores, cand_scores, cand_pools, value = \
            self._forward_policy(vn)
        ordered_vnodes = list(vn.nodes.values())
        ordered_links = list(vn.links.items())
        orig_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}

        mapping: Dict[str, str] = {}
        used: set = set()
        cand_lp: List[torch.Tensor] = []
        cand_ent: List[torch.Tensor] = []
        failed = False

        for v in ordered_vnodes:
            i = orig_idx[v.id]
            scores_i = cand_scores[i].clone()
            pool_i = cand_pools[i]

            # Mask snodes already used.
            if used:
                mask_vals = torch.tensor(
                    [float("-inf") if pool_i[j].id in used else 0.0
                     for j in range(len(pool_i))],
                    dtype=scores_i.dtype,
                )
                scores_i = scores_i + mask_vals

            if not torch.isfinite(scores_i).any():
                failed = True
                break

            if sample_cand:
                # Gumbel-Max sample with analytic Categorical log_prob.
                gumbel = -torch.log(-torch.log(
                    torch.rand_like(scores_i).clamp_min(1e-12)))
                pos = torch.argmax(scores_i + gumbel)
                dist = torch.distributions.Categorical(logits=scores_i)
                cand_lp.append(dist.log_prob(pos))
                cand_ent.append(dist.entropy())
            else:
                pos = torch.argmax(scores_i)

            chosen = pool_i[int(pos.item())]
            mapping[v.id] = chosen.id
            used.add(chosen.id)

        log_probs = {"node": [], "link": [], "cand": cand_lp}
        entropies = {"node": [], "link": [], "cand": cand_ent}
        if failed:
            return ordered_vnodes, ordered_links, None, log_probs, entropies, value
        return ordered_vnodes, ordered_links, mapping, log_probs, entropies, value


class ILMPVNEV21Direct(ILMPVNEV21):
    """Direct autoregressive deploy — cand_head decides each snode (argmax at
    eval). Use when the cand_head is the only learned component and we want
    the cleanest RL→deploy match."""

    def __init__(self):
        super().__init__()
        self.config["inference_mode"] = "direct"
        self.name = "IL-MP-VNE-V21-Direct"


class ILMPVNEV21PSO(ILMPVNEV21):
    """PSO deploy with V17/V19 per-domain K=1 regime; cand_head proposes one
    candidate per allowed domain, PSO picks among them."""

    NUM_RESTARTS = 3   # match V16/V17 for fair comparison

    def __init__(self):
        super().__init__()
        self.config["inference_mode"] = "pso"
        self.name = "IL-MP-VNE-V21-PSO"
