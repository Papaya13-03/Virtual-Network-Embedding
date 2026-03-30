import random
import os
import yaml
from typing import List, Dict, Tuple
from collections import OrderedDict

from algorithms.oa_mp_vne.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class OAMPVNE:
    def __init__(self):
        self.name = "OA-MP-VNE"
        self._active_mappings: Dict[str, Dict] = OrderedDict()

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "oa_mp_vne.yaml"
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {
                "pso": {
                    "num_particles": 20, "num_iterations": 15,
                    "w": 0.7, "c1": 1.5, "c2": 1.5, "mutation_rate": 0.1
                },
                "ordering": {"w_degree": 0.4, "w_cpu": 0.3, "w_bw": 0.3}
            }

    # ---- Ordering Stage ----

    def rank_virtual_nodes(self, vnetwork: VirtualNetwork) -> List[VirtualNode]:
        """
        Rank virtual nodes by composite score:
          score = w_degree * norm_degree + w_cpu * norm_cpu + w_bw * norm_adj_bw
        Higher score = embed first (most constrained).
        """
        ordering_cfg = self.config.get("ordering", {})
        w_degree = ordering_cfg.get("w_degree", 0.4)
        w_cpu = ordering_cfg.get("w_cpu", 0.3)
        w_bw = ordering_cfg.get("w_bw", 0.3)

        vnodes = list(vnetwork.nodes.values())
        if len(vnodes) <= 1:
            return vnodes

        # Compute raw metrics
        degrees: Dict[str, int] = {n.id: 0 for n in vnodes}
        adj_bw: Dict[str, float] = {n.id: 0.0 for n in vnodes}
        for vlink in vnetwork.links.values():
            degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
            degrees[vlink.target] = degrees.get(vlink.target, 0) + 1
            adj_bw[vlink.source] = adj_bw.get(vlink.source, 0.0) + vlink.bandwidth_demand
            adj_bw[vlink.target] = adj_bw.get(vlink.target, 0.0) + vlink.bandwidth_demand

        max_degree = max(degrees.values()) or 1
        max_cpu = max(n.cpu_demand for n in vnodes) or 1.0
        max_bw = max(adj_bw.values()) or 1.0

        scored = []
        for n in vnodes:
            norm_deg = degrees[n.id] / max_degree
            norm_cpu = n.cpu_demand / max_cpu
            norm_bw = adj_bw[n.id] / max_bw
            score = w_degree * norm_deg + w_cpu * norm_cpu + w_bw * norm_bw
            scored.append((score, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored]

    def rank_virtual_links(self, vnetwork: VirtualNetwork) -> List[Tuple[Tuple[str, str], VirtualLink]]:
        """
        Sort virtual links by bandwidth demand descending.
        Most demanding links get best paths first.
        """
        link_items = list(vnetwork.links.items())
        link_items.sort(key=lambda item: item[1].bandwidth_demand, reverse=True)
        return link_items

    # ---- Lifecycle ----

    def _release_expired(self, current_time: float) -> None:
        expired_ids = [rid for rid, data in self._active_mappings.items()
                       if data["expire_time"] <= current_time]
        for expired_id in expired_ids:
            data = self._active_mappings.pop(expired_id)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"]
            )

    def solve(self, substrate_network: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        self.global_controller = GlobalController(substrate_network)
        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        # Step 1: Order-aware node ranking
        ordered_vnodes = self.rank_virtual_nodes(vnetwork)

        # Step 2: Get candidates in the ranked order
        candidate_nodes = []
        for vnode in ordered_vnodes:
            candidates = []
            for lc in self.global_controller.local_controllers:
                if not vnode.allowed_domains or lc.domain.id in vnode.allowed_domains:
                    candidates.extend(lc.get_candidates(vnode))
            candidate_nodes.append(candidates)

        if any(not c for c in candidate_nodes):
            return solution

        # Build index maps based on ordered_vnodes
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(ordered_vnodes)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand
            })

        # Step 3: PSO optimization (same as MP-VNE, but over the ordered candidates)
        best_particle_idx = self.pso(candidate_nodes, virtual_request, vlink_indices, ordered_vnodes)

        # Build mapping
        best_mapping = {
            ordered_vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle_idx)
        }

        # Step 4: Order-aware link embedding
        try:
            vlink_paths = self._commit_mapping_ordered(best_mapping, vnetwork)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            return solution

        cost = self.fitness(best_particle_idx, candidate_nodes, virtual_request, vlink_indices, ordered_vnodes)

        solution.node_mapping = best_mapping
        solution.is_successful = True
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

        return solution

    def _commit_mapping_ordered(self, mapping: Dict[str, str], vnetwork: VirtualNetwork) -> Dict[tuple, list]:
        """
        Like GlobalController.commit_mapping, but processes virtual links
        in bandwidth-descending order so the most demanding links get
        the best substrate paths.
        """
        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, list] = {}

        try:
            # Allocate CPU
            for vnode_id, snode_id in mapping.items():
                vnode = vnetwork.nodes[vnode_id]
                _, snode = self.global_controller._find_snode(snode_id)
                if not snode:
                    raise ValueError(f"Node {snode_id} not found")
                if getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on {snode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # Allocate bandwidth in ORDER (highest BW demand first)
            ranked_links = self.rank_virtual_links(vnetwork)
            for vlink_key, vlink in ranked_links:
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
                        src_snode, dst_snode, bw_required=min_required, use_cache=False
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
            # Rollback
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

    # ---- PSO ----

    def pso(self, candidates: List[List[SubstrateNode]], request: VirtualNetworkRequest,
            vlink_indices: List[Dict], ordered_vnodes: List[VirtualNode]) -> List[int]:
        pso_config = self.config.get("pso", {})
        num_particles = pso_config.get("num_particles", 20)
        num_iterations = pso_config.get("num_iterations", 15)
        w = pso_config.get("w", 0.7)
        c1 = pso_config.get("c1", 1.5)
        c2 = pso_config.get("c2", 1.5)
        mutation_rate = pso_config.get("mutation_rate", 0.1)
        num_vnode = len(candidates)

        population = [
            [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)]
            for _ in range(num_particles)
        ]
        velocities = [[0.0] * num_vnode for _ in range(num_particles)]

        pbest = [p[:] for p in population]
        pbest_score = [self.fitness(p, candidates, request, vlink_indices, ordered_vnodes) for p in population]

        gbest_idx = pbest_score.index(min(pbest_score))
        gbest = pbest[gbest_idx][:]
        gbest_score = pbest_score[gbest_idx]

        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}...")
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

                score = self.fitness(population[i], candidates, request, vlink_indices, ordered_vnodes)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best = min(pbest_score)
            if current_best < gbest_score:
                gbest = pbest[pbest_score.index(current_best)][:]
                gbest_score = current_best

        return gbest

    def fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]],
                request: VirtualNetworkRequest, vlink_indices: List[Dict],
                ordered_vnodes: List[VirtualNode]) -> float:
        """
        Order-aware fitness: simulates sequential node placement in ranked order.
        """
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]

        # Penalize duplicate substrate node mappings
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float('inf')

        # Node cost
        node_cost = sum(
            vnode.cpu_demand * snode.cpu_price
            for vnode, snode in zip(ordered_vnodes, mapping)
        )

        # Link cost — evaluate links sorted by bandwidth demand (descending)
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
            link_cost += sum(
                l.transmission_delay + l.bandwidth_price * vlink_info["bw"]
                for l in path
            )

        return node_cost + link_cost
