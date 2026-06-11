import random
import time
import uuid
import os
import yaml
from typing import List, Dict
from collections import OrderedDict

from algorithms.mp_vne.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class MPVNELegacy:
    def __init__(self):
        self.name = "MP-VNE-Legacy"
        self._active_mappings: Dict[str, Dict] = OrderedDict()  # request_id -> {"mapping", "vlink_paths", "expire_time"}
        
        # Load configs
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "mp_vne.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            # Fallback to defaults if file not found
            self.config = {
                "pso": {
                    "num_particles": 20,
                    "num_iterations": 15,
                    "w": 0.7,
                    "c1": 1.5,
                    "c2": 1.5,
                    "mutation_rate": 0.1
                },
                "candidate_selection": {
                    "top_k": 5
                }
            }

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

        top_k = self.config.get("candidate_selection", {}).get("top_k", 5)
        candidate_nodes = self.global_controller.process_request(vnetwork, top_k=top_k)
        if any(not c for c in candidate_nodes):
            return solution  # Failed to find candidates for a virtual node
            
        # Pre-calculate virtual link endpoint indices to speed up fitness calculation
        vnodes_list = list(vnetwork.nodes.values())
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(vnodes_list)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand
            })
            
        best_particle_idx = self.pso(candidate_nodes, virtual_request, vlink_indices)

        # Build mapping dictionaries
        vnodes = list(vnetwork.nodes.values())
        best_mapping = {
            vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle_idx)
        }
        
        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vnetwork)
            # Post-mapping verification
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            return solution  # Mapping failed

        cost = self.fitness(best_particle_idx, candidate_nodes, virtual_request, vlink_indices)
        
        # Populate solution
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

    # ---------------- PSO & mapping ----------------
    def pso(self, candidates: List[List[SubstrateNode]], request: VirtualNetworkRequest, vlink_indices: List[Dict]) -> List[int]:
        pso_config = self.config.get("pso", {})
        num_particles: int = pso_config.get("num_particles", 20)
        num_iterations: int = pso_config.get("num_iterations", 15)
        w: float = pso_config.get("w", 0.7)
        c1: float = pso_config.get("c1", 1.5)
        c2: float = pso_config.get("c2", 1.5)
        mutation_rate: float = pso_config.get("mutation_rate", 0.1)
        num_vnode: int = len(candidates)

        population: List[List[int]] = [
            [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)]
            for _ in range(num_particles)
        ]
        velocities: List[List[float]] = [[0.0 for _ in range(num_vnode)] for _ in range(num_particles)]

        pbest: List[List[int]] = [p[:] for p in population]
        pbest_score: List[float] = [self.fitness(p, candidates, request, vlink_indices) for p in population]

        gbest_idx: int = pbest_score.index(min(pbest_score))
        gbest: List[int] = pbest[gbest_idx][:]
        gbest_score: float = pbest_score[gbest_idx]

        for _ in range(num_iterations):
            # print(f"  Iteration {_ + 1}/{num_iterations}...")
            for i in range(num_particles):
                for j in range(num_vnode):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - population[i][j])
                        + c2 * r2 * (gbest[j] - population[i][j])
                    )
                    new_idx: int = int(round(population[i][j] + velocities[i][j])) % len(candidates[j])
                    population[i][j] = new_idx

                if random.random() < mutation_rate:
                    mut_idx = random.randint(0, num_vnode - 1)
                    population[i][mut_idx] = random.randint(0, len(candidates[mut_idx]) - 1)

                score: float = self.fitness(population[i], candidates, request, vlink_indices)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best_score: float = min(pbest_score)
            if current_best_score < gbest_score:
                gbest = pbest[pbest_score.index(current_best_score)][:]
                gbest_score = current_best_score
            
        return gbest

    def fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]], request: VirtualNetworkRequest, vlink_indices: List[Dict]) -> float:
        vnetwork: VirtualNetwork = request.virtual_network
        vnodes: List[VirtualNode] = list(vnetwork.nodes.values())

        mapping: List[SubstrateNode] = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        
        # Penalize if multiple virtual nodes map to the same substrate node
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float('inf')

        node_cost: float = sum(vnode.cpu_demand * snode.cpu_price for vnode, snode in zip(vnodes, mapping))
        link_cost: float = 0.0
        
        for vlink_info in vlink_indices:
            src_node: SubstrateNode = mapping[vlink_info["src_idx"]]
            dst_node: SubstrateNode = mapping[vlink_info["dst_idx"]]
            
            # Single-path constraint: PSO must see the same feasibility filter
            # that commit will apply, otherwise it will keep picking mappings
            # that pass fitness but fail commit. Probe with the FULL demand.
            path = self.global_controller.shortest_path(
                src_node, dst_node, bw_required=vlink_info["bw"],
            )
            if not path:
                return float('inf')
            link_cost += sum(l.transmission_delay + l.bandwidth_price * vlink_info["bw"] for l in path)

        return node_cost + link_cost
