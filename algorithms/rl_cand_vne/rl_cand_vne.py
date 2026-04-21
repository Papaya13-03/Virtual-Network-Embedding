import hashlib
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import yaml

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from algorithms.rl_cand_vne.feature_extraction import (
    build_vn_adjacency,
    extract_domain_features,
    extract_vnode_features,
)
from algorithms.rl_cand_vne.policy_network import PolicyNetwork
from algorithms.rl_cand_vne.trainer import Trainer
from problem.embedding_solution import EmbeddingSolution
from problem.request import VirtualNetworkRequest
from problem.substrate_network import SubstrateNetwork
from problem.virtual_network import VirtualNetwork


def _default_config() -> Dict:
    return {
        "policy_network": {"hidden_size": 64, "num_gcn_layers": 2},
        "training": {
            "pretrain_episodes": 5000, "inline_pretrain_episodes": 500,
            "batch_size": 16, "online_k": 10, "baseline_window": 100,
            "lam_sup": 1.0, "warmup_fraction": 0.2,
            "u_max_cpu": 0.8, "u_max_bw": 0.8, "warmup_M_max": 20,
            "R_penalty": 2.0, "learning_rate": 0.001,
            "checkpoint_every": 500, "online_save_every": 100,
            "vn_min_nodes": 2, "vn_max_nodes": 8,
            "vn_min_cpu": 1.0, "vn_max_cpu": 30.0,
            "vn_min_bw": 5.0, "vn_max_bw": 80.0, "vn_link_prob": 0.5,
            "allowed_domains": {"p_all": 0.5, "p_single": 0.3, "p_subset": 0.2,
                                "subset_min": 2, "subset_max": 3},
        },
        "candidates": {"K": 5},
        "pso": {"num_particles": 20, "num_iterations": 15,
                "w": 0.7, "c1": 1.5, "c2": 1.5, "mutation_rate": 0.1},
        "checkpoint": {"path": "checkpoints/rl_cand_vne.pt", "require_hash_match": False},
    }


def substrate_structure_hash(sn) -> str:
    """Compute a stable hash of the substrate network structure (topology + capacities)."""
    if hasattr(sn, "domains"):
        # MultiDomainNetwork
        nodes = []
        links = []
        for domain in sn.domains.values():
            for nid, n in domain.network.nodes.items():
                nodes.append((nid, n.cpu_capacity, n.cpu_price, n.processing_delay))
            for (u, v), lk in domain.network.links.items():
                links.append((u, v, lk.bandwidth_capacity, lk.bandwidth_price, lk.transmission_delay))
        for (u, v), lk in sn.inter_domain_links.items():
            links.append((u, v, lk.bandwidth_capacity, lk.bandwidth_price, lk.transmission_delay))
        domains = sorted(sn.domains.keys())
    else:
        # Bare SubstrateNetwork
        nodes = [(nid, n.cpu_capacity, n.cpu_price, n.processing_delay)
                 for nid, n in sn.nodes.items()]
        links = [(u, v, lk.bandwidth_capacity, lk.bandwidth_price, lk.transmission_delay)
                 for (u, v), lk in sn.links.items()]
        domains = []
    payload = {
        "nodes": sorted(nodes),
        "links": sorted(links),
        "domains": domains,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RLCandVNE:
    """
    RL-based VNE using a candidate-selection policy (domain head + snode head per vnode).
    Untrained policy uses PSO + _commit_mapping_ordered from OAMPVNE for the embedding step.
    """

    def __init__(self):
        self.name = "RL-Cand-VNE"
        self._active_mappings: "OrderedDict[str, Dict]" = OrderedDict()
        self._request_count = 0
        self._initialized = False
        self._pretrained = False
        self._episodes_trained = 0

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "rl_cand_vne.yaml",
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = _default_config()

        pn_cfg = self.config["policy_network"]
        self.policy = PolicyNetwork(
            vnode_feat_size=5, snode_feat_size=5,
            hidden=pn_cfg["hidden_size"], K=self.config["candidates"]["K"],
        )
        self.trainer = Trainer(
            self.policy,
            lr=self.config["training"]["learning_rate"],
            lam_sup=self.config["training"]["lam_sup"],
            baseline_window=self.config["training"]["baseline_window"],
        )
        self.global_controller: GlobalController = None  # set lazily in solve()
        self._baseline_helper = OAMPVNE()  # PSO + commit engine

    # ---------- Candidate building from the policy output ----------

    def _build_policy_inputs(self, vn: VirtualNetwork) -> Tuple[torch.Tensor, torch.Tensor, List, List[float]]:
        vnode_feats = extract_vnode_features(vn)
        vn_adj = build_vn_adjacency(vn)

        domain_cache = {}
        for lc in self.global_controller.local_controllers:
            domain_cache[lc.domain.id] = (lc.domain, *extract_domain_features(lc.domain))

        domain_inputs_per_vnode = []
        cpu_demands = []
        vnodes = list(vn.nodes.values())
        for vnode in vnodes:
            allowed_ids = vnode.allowed_domains or [lc.domain.id for lc in self.global_controller.local_controllers]
            allowed_triples = []
            for did in allowed_ids:
                if did not in domain_cache:
                    continue
                domain_obj, X, A = domain_cache[did]
                node_ids = list(domain_obj.network.nodes.keys())
                avail = torch.tensor([
                    getattr(domain_obj.network.nodes[nid], "available_cpu",
                            domain_obj.network.nodes[nid].cpu_capacity)
                    for nid in node_ids
                ], dtype=torch.float32)
                allowed_triples.append((X, A, avail))
            if not allowed_triples:
                # Fallback: all domains
                for did, (domain_obj, X, A) in domain_cache.items():
                    node_ids = list(domain_obj.network.nodes.keys())
                    avail = torch.tensor([
                        getattr(domain_obj.network.nodes[nid], "available_cpu",
                                domain_obj.network.nodes[nid].cpu_capacity)
                        for nid in node_ids
                    ], dtype=torch.float32)
                    allowed_triples.append((X, A, avail))
            domain_inputs_per_vnode.append(allowed_triples)
            cpu_demands.append(float(vnode.cpu_demand))

        return vnode_feats, vn_adj, domain_inputs_per_vnode, cpu_demands

    def _resolve_candidate_snodes(
        self, vn: VirtualNetwork, policy_out: Dict,
    ) -> List[List]:
        """Translate policy output (indices) into lists of actual SubstrateNode objects."""
        vnodes = list(vn.nodes.values())
        candidate_nodes: List[List] = []
        for i, vnode in enumerate(vnodes):
            allowed_ids = vnode.allowed_domains or [lc.domain.id for lc in self.global_controller.local_controllers]
            d_idx = policy_out["chosen_domains"][i]
            did = allowed_ids[d_idx]
            lc = next(lc for lc in self.global_controller.local_controllers if lc.domain.id == did)
            domain_node_list = list(lc.domain.network.nodes.values())
            snode_indices = policy_out["chosen_snodes"][i]
            candidate_nodes.append([domain_node_list[j] for j in snode_indices])
        return candidate_nodes

    # ---------- Core solve ----------

    def solve(self, sn, req: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(sn)
            # Share our GlobalController with the PSO/commit helper
            self._baseline_helper.global_controller = self.global_controller
            self._initialized = True

        # First call: if no checkpoint on disk, run inline fallback pretraining.
        if not getattr(self, "_pretrained", False):
            ckpt_path = self.config.get("checkpoint", {}).get("path", "")
            h = substrate_structure_hash(sn) if sn is not None else ""
            if ckpt_path and os.path.exists(ckpt_path):
                self.load_checkpoint(ckpt_path, expected_hash=h)
            elif int(self.config["training"].get("inline_pretrain_episodes", 0)) > 0:
                self.pretrain_inline(sn)
            else:
                self._pretrained = True  # skip training; untrained policy

        self._release_expired(req.arrival_time)
        self.global_controller.clear_caches()

        vn = req.virtual_network
        solution = EmbeddingSolution(vnr_id=req.id, is_successful=False)

        self.policy.train()
        vnode_feats, vn_adj, dip, demands = self._build_policy_inputs(vn)
        policy_out = self.policy(
            vnode_feats=vnode_feats, vn_adj_norm=vn_adj,
            domain_inputs_per_vnode=dip, cpu_demands=demands, sample=True,
        )
        candidate_nodes = self._resolve_candidate_snodes(vn, policy_out)

        if any(not c for c in candidate_nodes):
            return solution

        # Delegate PSO to OAMPVNE baseline helper using our candidate sets.
        # OAMPVNE.pso signature: (candidates, request, vlink_indices, ordered_vnodes)
        ordered_vnodes = list(vn.nodes.values())
        vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
        vlink_indices = [
            {"src_idx": vnode_to_idx[vl.source],
             "dst_idx": vnode_to_idx[vl.target],
             "bw": vl.bandwidth_demand}
            for vl in vn.links.values()
        ]
        best_particle = self._baseline_helper.pso(candidate_nodes, req, vlink_indices, ordered_vnodes)
        mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        try:
            vlink_paths = self._baseline_helper._commit_mapping_ordered(mapping, vn)
        except Exception:
            return solution
        if not vlink_paths:
            return solution

        cost = self._composite_cost(mapping, vn, vlink_paths)

        solution.is_successful = True
        solution.node_mapping = mapping
        solution.embedding_cost = cost
        solution.link_mapping = {
            (v_src, v_dst): [
                ([(l.source, l.target) for l in path_links], bw)
                for (path_links, bw) in path_list
            ]
            for (v_src, v_dst), path_list in vlink_paths.items()
        }

        self._active_mappings[req.id] = {
            "mapping": mapping, "vnetwork": vn,
            "vlink_paths": vlink_paths,
            "expire_time": req.arrival_time + req.lifetime,
        }
        return solution

    def _composite_cost(self, mapping: Dict[str, str], vn: VirtualNetwork, vlink_paths: Dict) -> float:
        cost = 0.0
        for v_id, s_id in mapping.items():
            vnode = vn.nodes[v_id]
            _, snode = self.global_controller._find_snode(s_id)
            if snode:
                cost += vnode.cpu_demand * snode.cpu_price
                cost += snode.processing_delay
        for (_, _), paths in vlink_paths.items():
            for path_links, bw in paths:
                for link in path_links:
                    cost += bw * link.bandwidth_price
                    cost += link.transmission_delay
        return cost

    # ---------- Checkpoint I/O ----------

    def save_checkpoint(self, path: str, substrate_hash: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "config": self.config,
            "substrate_hash": substrate_hash,
            "episodes_trained": getattr(self, "_episodes_trained", 0),
            "baseline_buffer": list(self.trainer._baseline_buf),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, expected_hash: str = "") -> bool:
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(payload["policy_state_dict"])
        stored_hash = payload.get("substrate_hash", "")
        if expected_hash and stored_hash and expected_hash != stored_hash:
            require = self.config.get("checkpoint", {}).get("require_hash_match", False)
            if require:
                raise ValueError(
                    f"Substrate hash mismatch: expected {expected_hash}, stored {stored_hash}"
                )
            print(
                f"[rl_cand_vne] WARNING: substrate hash mismatch "
                f"(expected {expected_hash[:8]}, stored {stored_hash[:8]}). Continuing."
            )
        # Restore baseline buffer best-effort
        try:
            for r in payload.get("baseline_buffer", []):
                self.trainer._baseline_buf.append(float(r))
        except Exception:
            pass
        self._pretrained = True
        return True

    # ---------- Inline pretraining (fallback when no checkpoint) ----------

    def pretrain_inline(self, sn) -> None:
        from algorithms.rl_cand_vne.state_sampler import sample_substrate_state
        from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains

        if self.global_controller is None:
            self.global_controller = GlobalController(sn)
            self._baseline_helper.global_controller = self.global_controller

        train_cfg = self.config["training"]
        episodes = int(train_cfg["inline_pretrain_episodes"])
        batch_size = int(train_cfg["batch_size"])
        vn_kwargs = {
            "min_nodes": train_cfg["vn_min_nodes"], "max_nodes": train_cfg["vn_max_nodes"],
            "min_cpu": train_cfg["vn_min_cpu"], "max_cpu": train_cfg["vn_max_cpu"],
            "min_bw": train_cfg["vn_min_bw"], "max_bw": train_cfg["vn_max_bw"],
            "link_prob": train_cfg["vn_link_prob"],
        }
        domain_ids = [lc.domain.id for lc in self.global_controller.local_controllers]
        ad = train_cfg["allowed_domains"]

        self.policy.train()
        for ep in range(episodes):
            sample_substrate_state(
                self.global_controller, sn,
                warmup_fraction=train_cfg["warmup_fraction"],
                u_max_cpu=train_cfg["u_max_cpu"], u_max_bw=train_cfg["u_max_bw"],
                M_max=train_cfg["warmup_M_max"], vn_kwargs=vn_kwargs,
            )
            vn = generate_random_vn_with_domains(
                min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
                min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
                min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
                link_prob=vn_kwargs["link_prob"],
                domain_ids=domain_ids,
                p_all=ad["p_all"], p_single=ad["p_single"], p_subset=ad["p_subset"],
                subset_min=ad["subset_min"], subset_max=ad["subset_max"],
            )
            req = VirtualNetworkRequest(
                id=f"pt_{ep}", virtual_network=vn,
                arrival_time=0.0, lifetime=float("inf"),
            )
            reward, committed, dom_lps, sn_lps, success = self._training_episode(req)
            self.trainer.record(
                domain_log_probs=dom_lps,
                snode_log_probs_per_vnode=sn_lps,
                reward=reward,
                committed_snode_indices=committed,
                success=success,
            )
            if (ep + 1) % batch_size == 0:
                self.trainer.update()
            # Rollback any allocation this episode caused
            self.global_controller.reset_allocations()
            self.global_controller.clear_caches()

        if self.trainer.buffer:
            self.trainer.update()
        self._pretrained = True

    def _training_episode(self, req: VirtualNetworkRequest):
        """Return (reward, committed_snode_indices_or_None, dom_lps, sn_lps, success)."""
        vn = req.virtual_network
        vnode_feats, vn_adj, dip, demands = self._build_policy_inputs(vn)
        out = self.policy(
            vnode_feats=vnode_feats, vn_adj_norm=vn_adj,
            domain_inputs_per_vnode=dip, cpu_demands=demands, sample=True,
        )
        candidate_nodes = self._resolve_candidate_snodes(vn, out)
        R_penalty = self.config["training"]["R_penalty"]

        if any(not c for c in candidate_nodes):
            return (
                -R_penalty, None,
                out["domain_log_probs"], out["snode_log_probs_per_vnode"],
                False,
            )

        ordered_vnodes = list(vn.nodes.values())
        vnode_to_idx = {v.id: i for i, v in enumerate(ordered_vnodes)}
        vlink_indices = [
            {
                "src_idx": vnode_to_idx[vl.source],
                "dst_idx": vnode_to_idx[vl.target],
                "bw": vl.bandwidth_demand,
            }
            for vl in vn.links.values()
        ]
        try:
            best_particle = self._baseline_helper.pso(
                candidate_nodes, req, vlink_indices, ordered_vnodes
            )
        except Exception:
            return (
                -R_penalty, None,
                out["domain_log_probs"], out["snode_log_probs_per_vnode"],
                False,
            )

        mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }
        try:
            vlink_paths = self._baseline_helper._commit_mapping_ordered(mapping, vn)
            if not vlink_paths:
                raise ValueError("no paths")
        except Exception:
            return (
                -R_penalty, None,
                out["domain_log_probs"], out["snode_log_probs_per_vnode"],
                False,
            )

        cost = self._composite_cost(mapping, vn, vlink_paths)
        revenue = sum(nd.cpu_demand for nd in vn.nodes.values()) + \
                  sum(vl.bandwidth_demand for vl in vn.links.values())
        reward = -cost / max(revenue, 1e-6)
        committed_indices = list(best_particle)
        return (
            reward, committed_indices,
            out["domain_log_probs"], out["snode_log_probs_per_vnode"],
            True,
        )

    def _release_expired(self, now: float) -> None:
        expired = [rid for rid, d in self._active_mappings.items() if d["expire_time"] <= now]
        for rid in expired:
            data = self._active_mappings.pop(rid)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"],
            )
