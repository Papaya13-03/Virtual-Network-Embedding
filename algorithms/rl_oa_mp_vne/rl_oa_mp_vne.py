# algorithms/rl_oa_mp_vne/rl_oa_mp_vne.py
import os
import random
import yaml
import torch
from typing import List, Dict, Tuple
from collections import OrderedDict

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.oa_mp_vne.local_controller import LocalController
from algorithms.rl_oa_mp_vne.policy_network import PolicyNetwork
from algorithms.rl_oa_mp_vne.trainer import RankingTrainer
from algorithms.rl_oa_mp_vne.vn_generator import generate_random_vn
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class RLOAMPVNE:
    """
    RL-enhanced Order-Aware Multi-Path VNE.

    Uses a Policy Network (GCN + REINFORCE) to rank virtual nodes and links
    instead of the hand-crafted heuristic in OA-MP-VNE.
    Pre-trains on synthetic virtual networks, then continues learning
    online every k requests.
    """

    def __init__(self):
        self.name = "RL-OA-MP-VNE"
        self._active_mappings: Dict[str, Dict] = OrderedDict()
        self._request_count = 0
        self._pretrained = False

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "rl_oa_mp_vne.yaml",
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {
                "pso": {
                    "num_particles": 20, "num_iterations": 15,
                    "w": 0.7, "c1": 1.5, "c2": 1.5, "mutation_rate": 0.1,
                },
                "policy_network": {
                    "hidden_size": 64, "gcn_hidden": 32,
                    "learning_rate": 0.001, "gamma": 0.99,
                },
                "candidates": {"K": 10},
                "training": {
                    "pretrain_episodes": 800, "batch_size": 16, "online_k": 10,
                    "vn_min_nodes": 2, "vn_max_nodes": 8,
                    "vn_min_cpu": 1.0, "vn_max_cpu": 30.0,
                    "vn_min_bw": 5.0, "vn_max_bw": 80.0,
                    "vn_link_prob": 0.5,
                },
            }

        pn_cfg = self.config.get("policy_network", {})
        self.policy = PolicyNetwork(
            vnode_feat_size=5,
            vlink_feat_size=5,
            gcn_node_feat_size=5,
            gcn_hidden=pn_cfg.get("gcn_hidden", 32),
            hidden_size=pn_cfg.get("hidden_size", 64),
        )
        self.trainer = RankingTrainer(
            self.policy,
            lr=pn_cfg.get("learning_rate", 0.001),
            gamma=pn_cfg.get("gamma", 0.99),
            batch_size=self.config.get("training", {}).get("batch_size", 16),
        )
        self.global_controller = None

    # ---- Controller Init ----

    def _init_controller(self, substrate_network: SubstrateNetwork) -> None:
        self.global_controller = GlobalController(substrate_network)

    # ---- Feature Extraction ----

    def _extract_domain_features(self, lc: LocalController) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract node features and normalized adjacency for one domain.
        Returns: X (num_nodes, 5), A_norm (num_nodes, num_nodes)
        """
        net = lc.domain.network
        node_ids = list(net.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Node features: [avail_cpu/capacity, cpu_price, proc_delay, degree, avg_neighbor_bw]
        X = torch.zeros(n, 5)
        degrees = [0] * n
        neighbor_bw = [0.0] * n

        for (u, v), link in net.links.items():
            if u in id_to_idx:
                degrees[id_to_idx[u]] += 1
                neighbor_bw[id_to_idx[u]] += getattr(link, 'available_bw', link.bandwidth_capacity)
            if v in id_to_idx:
                degrees[id_to_idx[v]] += 1
                neighbor_bw[id_to_idx[v]] += getattr(link, 'available_bw', link.bandwidth_capacity)

        max_degree = max(degrees) if degrees and max(degrees) > 0 else 1
        max_cap = max(nd.cpu_capacity for nd in net.nodes.values()) or 1.0
        max_bw = max(neighbor_bw) if neighbor_bw and max(neighbor_bw) > 0 else 1.0

        for i, nid in enumerate(node_ids):
            node = net.nodes[nid]
            avail = getattr(node, 'available_cpu', node.cpu_capacity)
            X[i, 0] = avail / max_cap
            X[i, 1] = node.cpu_price / 10.0  # normalize price
            X[i, 2] = node.processing_delay / 10.0  # normalize delay
            X[i, 3] = degrees[i] / max_degree
            X[i, 4] = (neighbor_bw[i] / degrees[i] / max_bw) if degrees[i] > 0 else 0.0

        # Adjacency with self-loops + symmetric normalization
        A = torch.zeros(n, n)
        for (u, v), link in net.links.items():
            if u in id_to_idx and v in id_to_idx:
                bw_ratio = getattr(link, 'available_bw', link.bandwidth_capacity) / link.bandwidth_capacity
                A[id_to_idx[u], id_to_idx[v]] = bw_ratio
                A[id_to_idx[v], id_to_idx[u]] = bw_ratio

        # Add self-loops
        A = A + torch.eye(n)
        # D^{-1/2} A D^{-1/2}
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        return X, A_norm

    def _extract_vnode_features(self, vn: VirtualNetwork) -> torch.Tensor:
        """
        Extract features for each virtual node.
        Returns: (num_vnodes, 5) tensor.
        Features: [cpu_demand_norm, degree_norm, adj_bw_norm, num_nodes_norm, num_links_norm]
        """
        vnodes = list(vn.nodes.values())
        n = len(vnodes)
        feats = torch.zeros(n, 5)

        degrees = {nd.id: 0 for nd in vnodes}
        adj_bw = {nd.id: 0.0 for nd in vnodes}
        for vlink in vn.links.values():
            degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
            degrees[vlink.target] = degrees.get(vlink.target, 0) + 1
            adj_bw[vlink.source] = adj_bw.get(vlink.source, 0.0) + vlink.bandwidth_demand
            adj_bw[vlink.target] = adj_bw.get(vlink.target, 0.0) + vlink.bandwidth_demand

        max_cpu = max(nd.cpu_demand for nd in vnodes) or 1.0
        max_deg = max(degrees.values()) or 1
        max_bw = max(adj_bw.values()) if adj_bw and max(adj_bw.values()) > 0 else 1.0

        for i, nd in enumerate(vnodes):
            feats[i, 0] = nd.cpu_demand / max_cpu
            feats[i, 1] = degrees[nd.id] / max_deg
            feats[i, 2] = adj_bw[nd.id] / max_bw
            feats[i, 3] = len(vn.nodes) / 20.0
            feats[i, 4] = len(vn.links) / 40.0

        return feats

    def _extract_vlink_features(self, vn: VirtualNetwork) -> torch.Tensor:
        """
        Extract features for each virtual link.
        Returns: (num_vlinks, 5) tensor.
        Features: [bw_norm, src_degree_norm, dst_degree_norm, src_cpu_norm, dst_cpu_norm]
        """
        vlinks = list(vn.links.values())
        m = len(vlinks)
        feats = torch.zeros(m, 5)

        degrees = {nid: 0 for nid in vn.nodes}
        for vlink in vlinks:
            degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
            degrees[vlink.target] = degrees.get(vlink.target, 0) + 1

        max_bw = max(vl.bandwidth_demand for vl in vlinks) or 1.0
        max_deg = max(degrees.values()) or 1
        max_cpu = max(nd.cpu_demand for nd in vn.nodes.values()) or 1.0

        for i, vl in enumerate(vlinks):
            feats[i, 0] = vl.bandwidth_demand / max_bw
            feats[i, 1] = degrees[vl.source] / max_deg
            feats[i, 2] = degrees[vl.target] / max_deg
            feats[i, 3] = vn.nodes[vl.source].cpu_demand / max_cpu
            feats[i, 4] = vn.nodes[vl.target].cpu_demand / max_cpu

        return feats

    def _get_domain_inputs(
        self, vn: VirtualNetwork,
    ) -> Tuple[list, list, List[List[SubstrateNode]], List[torch.Tensor]]:
        """
        For each vnode, collect:
          - Xs, As: per-domain GCN inputs for allowed domains (used by all heads)
          - pool: flattened SubstrateNode list across allowed domains
                  (index-aligned with the candidate head's score vector)
          - cpu_slack: per-pool-snode (available_cpu - vnode.cpu_demand) as a tensor
        """
        domain_cache = {}
        domain_node_lists = {}
        for lc in self.global_controller.local_controllers:
            X, A = self._extract_domain_features(lc)
            domain_cache[lc.domain.id] = (X, A)
            domain_node_lists[lc.domain.id] = list(lc.domain.network.nodes.values())

        all_domain_ids = [lc.domain.id for lc in self.global_controller.local_controllers]

        per_vnode_Xs: list = []
        per_vnode_As: list = []
        per_vnode_pools: List[List[SubstrateNode]] = []
        per_vnode_slacks: List[torch.Tensor] = []

        for vnode in vn.nodes.values():
            allowed = vnode.allowed_domains or all_domain_ids
            allowed = [d for d in allowed if d in domain_cache] or all_domain_ids

            Xs = tuple(domain_cache[d][0] for d in allowed)
            As = tuple(domain_cache[d][1] for d in allowed)

            pool: List[SubstrateNode] = []
            slacks: List[float] = []
            for d in allowed:
                for snode in domain_node_lists[d]:
                    avail = getattr(snode, "available_cpu", snode.cpu_capacity)
                    pool.append(snode)
                    slacks.append(avail - vnode.cpu_demand)

            per_vnode_Xs.append(Xs)
            per_vnode_As.append(As)
            per_vnode_pools.append(pool)
            per_vnode_slacks.append(torch.tensor(slacks, dtype=torch.float32))

        return per_vnode_Xs, per_vnode_As, per_vnode_pools, per_vnode_slacks

    # ---- NN-Based Ranking ----

    @staticmethod
    def _plackett_luce_sample(scores: torch.Tensor, items: list) -> Tuple[list, List[torch.Tensor]]:
        """Sample a full permutation from scores using the Plackett-Luce model."""
        remaining_scores = scores.clone()
        remaining_indices = list(range(len(items)))
        ordered_indices = []
        log_probs = []

        for _ in range(len(items)):
            dist = torch.distributions.Categorical(logits=remaining_scores)
            chosen_pos = dist.sample()
            log_probs.append(dist.log_prob(chosen_pos))

            chosen_idx = remaining_indices[chosen_pos.item()]
            ordered_indices.append(chosen_idx)

            mask = torch.ones(len(remaining_indices), dtype=torch.bool)
            mask[chosen_pos.item()] = False
            remaining_scores = remaining_scores[mask]
            remaining_indices = [ri for j, ri in enumerate(remaining_indices) if j != chosen_pos.item()]

        return [items[i] for i in ordered_indices], log_probs

    @staticmethod
    def _plackett_luce_topk(
        scores: torch.Tensor, k: int,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """Sample up to k distinct indices by Plackett-Luce. Infeasible entries
        (-inf in `scores`) are skipped; if fewer than k feasible entries exist,
        sampling stops early. Returns parallel lists of length <= k."""
        finite_count = int(torch.isfinite(scores).sum().item())
        k = min(k, finite_count)
        if k == 0:
            return [], []

        remaining = scores.clone()
        remaining_indices = list(range(scores.shape[0]))
        chosen: List[int] = []
        log_probs: List[torch.Tensor] = []

        for _ in range(k):
            dist = torch.distributions.Categorical(logits=remaining)
            pos = dist.sample()
            log_probs.append(dist.log_prob(pos))

            pos_item = pos.item()
            chosen.append(remaining_indices[pos_item])

            mask = torch.ones(len(remaining_indices), dtype=torch.bool)
            mask[pos_item] = False
            remaining = remaining[mask]
            remaining_indices = [ri for j, ri in enumerate(remaining_indices) if j != pos_item]

        return chosen, log_probs

    @staticmethod
    def _topk_greedy(scores: torch.Tensor, k: int) -> List[int]:
        """Deterministic top-k selection; skips -inf entries."""
        finite_count = int(torch.isfinite(scores).sum().item())
        k = min(k, finite_count)
        if k == 0:
            return []
        topk = torch.topk(scores, k).indices.tolist()
        return topk

    @staticmethod
    def _greedy_sort(scores: torch.Tensor, items: list) -> list:
        """Sort items by scores descending (no log_probs)."""
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        return [items[i] for i in sorted_indices]

    def _forward_policy(
        self, vn: VirtualNetwork,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[List[SubstrateNode]]]:
        """Single forward pass returning all three heads' outputs plus the
        per-vnode SubstrateNode pools (index-aligned with candidate scores)."""
        vnode_feats = self._extract_vnode_features(vn)
        vlink_feats = self._extract_vlink_features(vn)
        per_vnode_Xs, per_vnode_As, per_vnode_pools, per_vnode_slacks = self._get_domain_inputs(vn)
        node_scores, link_scores, cand_scores = self.policy(
            vnode_feats, vlink_feats,
            per_vnode_Xs, per_vnode_As,
            per_vnode_cpu_slacks=per_vnode_slacks,
        )
        return node_scores, link_scores, cand_scores, per_vnode_pools

    def rank_all_nn(
        self, vn: VirtualNetwork, sample: bool = False,
    ) -> Tuple[
        List[VirtualNode],
        List[Tuple[Tuple[str, str], VirtualLink]],
        List[List[SubstrateNode]],
        Dict[str, List[torch.Tensor]],
    ]:
        """
        Rank vnodes + vlinks and pick top-K candidate snodes per vnode with a
        single forward pass.
        Returns:
            ordered_vnodes, ordered_vlinks, candidate_nodes (aligned with
            ordered_vnodes), log_probs (Dict of log probs for each component).
        """
        node_scores, link_scores, cand_scores, cand_pools = self._forward_policy(vn)
        vnodes = list(vn.nodes.values())
        link_items = list(vn.links.items())
        K = int(self.config.get("candidates", {}).get("K", 5))

        if sample:
            ordered_vnodes, node_lp = self._plackett_luce_sample(node_scores, vnodes)
            ordered_links, link_lp = self._plackett_luce_sample(link_scores, link_items)
        else:
            ordered_vnodes = self._greedy_sort(node_scores, vnodes)
            ordered_links = self._greedy_sort(link_scores, link_items)
            node_lp, link_lp = [], []

        orig_idx = {v.id: i for i, v in enumerate(vnodes)}
        candidate_nodes: List[List[SubstrateNode]] = []
        cand_lp: List[torch.Tensor] = []
        for v in ordered_vnodes:
            i = orig_idx[v.id]
            scores_i = cand_scores[i]
            pool_i = cand_pools[i]
            if sample:
                picked, lps = self._plackett_luce_topk(scores_i, K)
                cand_lp.extend(lps)
            else:
                picked = self._topk_greedy(scores_i, K)
            candidate_nodes.append([pool_i[j] for j in picked])

        return ordered_vnodes, ordered_links, candidate_nodes, {
            "node": node_lp,
            "link": link_lp,
            "cand": cand_lp
        }

    # ---- Pre-Training ----

    def _pretrain(self, substrate_network: SubstrateNetwork) -> None:
        """Pre-train on synthetic virtual networks."""
        train_cfg = self.config.get("training", {})
        episodes = train_cfg.get("pretrain_episodes", 200)
        batch_size = train_cfg.get("batch_size", 16)

        print(f"  [RL-OA-MP-VNE] Pre-training on {episodes} synthetic VNs...")
        self.policy.train()

        for ep in range(episodes):
            vn = generate_random_vn(
                min_nodes=train_cfg.get("vn_min_nodes", 2),
                max_nodes=train_cfg.get("vn_max_nodes", 8),
                min_cpu=train_cfg.get("vn_min_cpu", 1.0),
                max_cpu=train_cfg.get("vn_max_cpu", 30.0),
                min_bw=train_cfg.get("vn_min_bw", 5.0),
                max_bw=train_cfg.get("vn_max_bw", 80.0),
                link_prob=train_cfg.get("vn_link_prob", 0.5),
            )

            # Reset substrate for each episode
            self.global_controller.reset_allocations()
            self.global_controller.clear_caches()

            # NN-ranked ordering + candidate selection (single forward pass)
            ordered_vnodes, ordered_links, candidate_nodes, all_log_probs = \
                self.rank_all_nn(vn, sample=True)

            # Try embedding with this ordering + learned candidates
            reward = self._try_embedding(vn, ordered_vnodes, ordered_links, candidate_nodes)
            self.trainer.record(all_log_probs, reward)

            if (ep + 1) % batch_size == 0:
                loss_dict = self.trainer.update()
                if (ep + 1) % (batch_size * 5) == 0:
                    print(f"    Episode {ep + 1}/{episodes}, reward={loss_dict['avg_reward']:.4f}, loss={loss_dict['total_loss']:.4f} (n:{loss_dict['node_loss']:.4f}, l:{loss_dict['link_loss']:.4f}, c:{loss_dict['cand_loss']:.4f})")

        # Final update for remaining buffer
        if self.trainer.buffer:
            self.trainer.update()

        self.global_controller.reset_allocations()
        self.global_controller.clear_caches()
        self._pretrained = True
        print(f"  [RL-OA-MP-VNE] Pre-training complete.")

    def _try_embedding(
        self, vn: VirtualNetwork,
        ordered_vnodes: List[VirtualNode],
        ordered_links: List[Tuple[Tuple[str, str], VirtualLink]],
        candidate_nodes: List[List[SubstrateNode]],
    ) -> float:
        """
        Attempt a full embedding with the given ordering + learned candidates.
        Returns reward. Does NOT permanently allocate — rolls back after evaluation.
        """
        if any(not c for c in candidate_nodes):
            return -1.0

        # Build index maps
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(ordered_vnodes)}
        vlink_indices = []
        for vlink in vn.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # PSO search
        best_particle = self._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        # Try commit (will rollback internally on failure)
        try:
            vlink_paths = self._commit_mapping_ordered(mapping, vn, ordered_links)
            if not vlink_paths:
                return -1.0

            # Compute reward
            revenue = sum(nd.cpu_demand for nd in vn.nodes.values()) + \
                      sum(vl.bandwidth_demand for vl in vn.links.values())
            cost = self._compute_cost(mapping, vn, vlink_paths)
            reward = revenue / (cost + 1e-6)

            # Rollback — pre-training should not permanently allocate
            self.global_controller.release_mapping(mapping, vn, vlink_paths)
            return reward

        except ValueError:
            return -1.0

    def _compute_cost(self, mapping: Dict[str, str], vn: VirtualNetwork, vlink_paths: Dict) -> float:
        """Total embedding cost aligned with the evaluation metric in
        evaluation/visualize_results.py: node CPU cost (cpu_demand*cpu_price)
        plus per-vlink bw*hop_count. Keeping reward consistent with the
        evaluation metric avoids training/test-time objective mismatch."""
        cost = 0.0
        for vnode_id, snode_id in mapping.items():
            vnode = vn.nodes[vnode_id]
            _, snode = self.global_controller._find_snode(snode_id)
            if snode:
                cost += vnode.cpu_demand * snode.cpu_price
        for (v_src, v_dst), paths in vlink_paths.items():
            for path_links, bw in paths:
                cost += bw * len(path_links)
        return cost

    # ---- PSO (reused from oa_mp_vne) ----

    def _repair_particle(self, particle: List[int], candidates: List[List[SubstrateNode]]) -> List[int]:
        used_snode_ids = set()
        for i in range(len(particle)):
            snode = candidates[i][particle[i]]
            if snode.id in used_snode_ids:
                found = False
                alternatives = list(range(len(candidates[i])))
                random.shuffle(alternatives)
                for alt_idx in alternatives:
                    if candidates[i][alt_idx].id not in used_snode_ids:
                        particle[i] = alt_idx
                        snode = candidates[i][alt_idx]
                        found = True
                        break
                if not found:
                    pass
            used_snode_ids.add(snode.id)
        return particle

    def _pso(self, candidates: List[List[SubstrateNode]],
             vlink_indices: List[Dict], ordered_vnodes: List[VirtualNode]) -> List[int]:
        pso_cfg = self.config.get("pso", {})
        num_particles = pso_cfg.get("num_particles", 20)
        num_iterations = pso_cfg.get("num_iterations", 15)
        w = pso_cfg.get("w", 0.7)
        c1 = pso_cfg.get("c1", 1.5)
        c2 = pso_cfg.get("c2", 1.5)
        mutation_rate = pso_cfg.get("mutation_rate", 0.1)
        num_vnode = len(candidates)

        population = [
            self._repair_particle(
                [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)],
                candidates,
            )
            for _ in range(num_particles)
        ]
        velocities = [[0.0] * num_vnode for _ in range(num_particles)]

        pbest = [p[:] for p in population]
        pbest_score = [self._fitness(p, candidates, vlink_indices, ordered_vnodes) for p in population]

        gbest_idx = pbest_score.index(min(pbest_score))
        gbest = pbest[gbest_idx][:]
        gbest_score = pbest_score[gbest_idx]

        for _ in range(num_iterations):
            for i in range(num_particles):
                for j in range(num_vnode):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - population[i][j])
                        + c2 * r2 * (gbest[j] - population[i][j])
                    )
                    new_idx = int(round(population[i][j] + velocities[i][j])) % len(candidates[j])
                    population[i][j] = new_idx

                if random.random() < mutation_rate:
                    mut_idx = random.randint(0, num_vnode - 1)
                    population[i][mut_idx] = random.randint(0, len(candidates[mut_idx]) - 1)

                self._repair_particle(population[i], candidates)

                score = self._fitness(population[i], candidates, vlink_indices, ordered_vnodes)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best = min(pbest_score)
            if current_best < gbest_score:
                gbest = pbest[pbest_score.index(current_best)][:]
                gbest_score = current_best

        return gbest

    def _fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]],
                 vlink_indices: List[Dict], ordered_vnodes: List[VirtualNode]) -> float:
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float('inf')

        node_cost = sum(
            vnode.cpu_demand * snode.cpu_price
            for vnode, snode in zip(ordered_vnodes, mapping)
        )

        sorted_vlinks = sorted(vlink_indices, key=lambda x: x["bw"], reverse=True)
        link_cost = 0.0
        for vlink_info in sorted_vlinks:
            src_node = mapping[vlink_info["src_idx"]]
            dst_node = mapping[vlink_info["dst_idx"]]
            path = self.global_controller.shortest_path(
                src_node, dst_node, bw_required=min(1.0, vlink_info["bw"] * 0.1)
            )
            if not path:
                return float('inf')
            # Hop-count-based cost (matches _compute_cost and the evaluation metric)
            link_cost += len(path) * vlink_info["bw"]

        return node_cost + link_cost

    # ---- Commit Mapping (reused from oa_mp_vne, with custom link order) ----

    def _commit_mapping_ordered(
        self, mapping: Dict[str, str], vn: VirtualNetwork,
        ordered_links: List[Tuple[Tuple[str, str], VirtualLink]] = None,
    ) -> Dict[tuple, list]:
        """
        Commit mapping with links processed in the given order.
        If ordered_links is None, falls back to BW-descending order.
        """
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Duplicate substrate node in mapping")

        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, list] = {}

        if ordered_links is None:
            link_items = list(vn.links.items())
            link_items.sort(key=lambda item: item[1].bandwidth_demand, reverse=True)
            ordered_links = link_items

        try:
            # Allocate CPU
            for vnode_id, snode_id in mapping.items():
                vnode = vn.nodes[vnode_id]
                _, snode = self.global_controller._find_snode(snode_id)
                if not snode:
                    raise ValueError(f"Node {snode_id} not found")
                if getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on {snode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # Allocate bandwidth in NN-ranked order with MP-VNE multi-path splitting.
            # A single shortest path may not have enough BW on its bottleneck link;
            # splitting the demand across up to `max_paths` paths unlocks feasible
            # embeddings that single-path would reject and tends to use shorter hops.
            for vlink_key, vlink in ordered_links:
                src_snode_id = mapping[vlink.source]
                dst_snode_id = mapping[vlink.target]
                _, src_snode = self.global_controller._find_snode(src_snode_id)
                _, dst_snode = self.global_controller._find_snode(dst_snode_id)

                demand_remaining = vlink.bandwidth_demand
                allocated_paths = []
                max_paths = 5

                while demand_remaining > 0.001 and len(allocated_paths) < max_paths:
                    min_required = min(demand_remaining * 0.1, 1.0, demand_remaining)
                    path = self.global_controller.shortest_path(
                        src_snode, dst_snode, bw_required=min_required, use_cache=False,
                    )
                    if not path:
                        break
                    path_bw = min(getattr(l, 'available_bw', l.bandwidth_capacity) for l in path)
                    allocated = min(demand_remaining, path_bw)
                    for link in path:
                        link.available_bw -= allocated
                        link_key = (link.source, link.target)
                        allocated_bw[link_key] = allocated_bw.get(link_key, 0) + allocated
                    allocated_paths.append((path, allocated))
                    demand_remaining -= allocated

                if demand_remaining > 0.001:
                    raise ValueError(f"Insufficient multi-path BW for vlink {vlink.source}->{vlink.target}")

                vlink_paths[vlink_key] = allocated_paths

        except Exception as e:
            for snode_id, cpu in allocated_cpu.items():
                _, snode = self.global_controller._find_snode(snode_id)
                if snode:
                    snode.available_cpu += cpu
            for link_key, bw in allocated_bw.items():
                link = self.global_controller._find_link(*link_key)
                if link:
                    link.available_bw += bw
            raise e

        return vlink_paths

    # ---- Lifecycle ----

    def _release_expired(self, current_time: float) -> None:
        expired_ids = [rid for rid, data in self._active_mappings.items()
                       if data["expire_time"] <= current_time]
        for expired_id in expired_ids:
            data = self._active_mappings.pop(expired_id)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"]
            )

    # ---- Main Solve ----

    def solve(self, substrate_network: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        # Initialize controller on first call
        if self.global_controller is None:
            self._init_controller(substrate_network)

        # Pre-train on first call
        if not self._pretrained:
            self._pretrain(substrate_network)

        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        # NN-ranked ordering + candidate selection (on-policy: log_probs match the executed action)
        self.policy.train()
        ordered_vnodes, ordered_links, candidate_nodes, log_probs = \
            self.rank_all_nn(vnetwork, sample=True)

        if any(not c for c in candidate_nodes):
            self._record_online(log_probs, -1.0)
            return solution

        # Build index maps
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(ordered_vnodes)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # PSO optimization
        best_particle = self._pso(candidate_nodes, vlink_indices, ordered_vnodes)
        best_mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        # Commit
        try:
            vlink_paths = self._commit_mapping_ordered(best_mapping, vnetwork, ordered_links)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            self._record_online(log_probs, -1.0)
            return solution

        cost = self._compute_cost(best_mapping, vnetwork, vlink_paths)
        revenue = sum(nd.cpu_demand for nd in vnetwork.nodes.values()) + \
                  sum(vl.bandwidth_demand for vl in vnetwork.links.values())
        reward = revenue / (cost + 1e-6)

        solution.node_mapping = best_mapping
        solution.embedding_cost = cost

        formatted_link_mapping = {}
        for (v_src, v_dst), allocated_paths in vlink_paths.items():
            formatted_paths = []
            for path_links, allocated_bw in allocated_paths:
                link_tuples = [(l.source, l.target) for l in path_links]
                formatted_paths.append((link_tuples, allocated_bw))
            formatted_link_mapping[(v_src, v_dst)] = formatted_paths
        solution.link_mapping = formatted_link_mapping

        self._active_mappings[virtual_request.id] = {
            "mapping": best_mapping,
            "vnetwork": vnetwork,
            "vlink_paths": vlink_paths,
            "expire_time": virtual_request.arrival_time + virtual_request.lifetime,
        }

        # Online learning — log_probs are on-policy (from the sampled ordering used above)
        self._record_online(log_probs, reward)

        return solution

    def _record_online(self, log_probs: Dict[str, List[torch.Tensor]], reward: float) -> None:
        """Record on-policy experience and update every k requests."""
        self._request_count += 1
        self.trainer.record(log_probs, reward)

        online_k = self.config.get("training", {}).get("online_k", 10)
        if self._request_count % online_k == 0 and self.trainer.buffer:
            self.trainer.update()
