import random
import time
import uuid
from typing import List, Dict
from collections import OrderedDict

from src.algorithms.MP_VNE.global_controller import GlobalController
from src.types.substrate import SubstrateNetwork, SubstrateNode
from src.types.virtual import VirtualNetwork, VirtualNode, VirtualLink
from src.types.request import VirtualRequest


class MP_VNE:
    def __init__(self, snetwork: SubstrateNetwork) -> None:
        self.global_controller: GlobalController = GlobalController(snetwork)
        self._active_mappings: Dict[str, Dict] = OrderedDict()  # request_id -> {"mapping", "vlinks", "vlink_paths", "expire_time"}

    def handle_mapping_request(self, request: VirtualRequest, current_time: float):
        request_id = str(uuid.uuid4())
        vnetwork = request["vnetwork"]
        lifetime = request.get("lifetime", 1000)

        candidate_nodes = self.global_controller.process_request(vnetwork)
        best_particle_idx = self.pso(candidate_nodes, request)

        best_mapping = {
            vnode: candidate_nodes[i][idx]
            for i, (vnode, idx) in enumerate(zip(vnetwork.nodes, best_particle_idx))
        }
        vlinks = getattr(vnetwork, "links", [])
        print("vlinks;;;;;", vlinks)
        # Commit và lấy snapshot path
        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vlinks=vlinks)
        except ValueError:
            raise

        # Lưu thông tin mapping
        self._active_mappings[request_id] = {
            "mapping": best_mapping,
            "vlinks": vlinks,
            "vlink_paths": vlink_paths,  # snapshot path
            "expire_time": current_time + lifetime
        }

        print(vlink_paths)

        cost = self.fitness(best_particle_idx, candidate_nodes, request)
        return request_id, cost, self._active_mappings[request_id]

    def release_expired_requests(self, current_time: float) -> None:
        """Giải phóng các mapping hết lifetime"""
        expired_ids = [rid for rid, info in self._active_mappings.items() if info["expire_time"] <= current_time]
        for rid in expired_ids:
            info = self._active_mappings.pop(rid)
            self.global_controller.release_mapping(info["mapping"], info["vlink_paths"])  # dùng snapshot path

    # ---------------- PSO & mapping ----------------
    def pso(self, candidates: List[List[SubstrateNode]], request: VirtualRequest) -> List[int]:
        num_particles: int = 50
        num_iterations: int = 30
        num_vnode: int = len(candidates)

        population: List[List[int]] = [
            [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)]
            for _ in range(num_particles)
        ]
        velocities: List[List[float]] = [[0.0 for _ in range(num_vnode)] for _ in range(num_particles)]

        pbest: List[List[int]] = [p[:] for p in population]
        pbest_score: List[float] = [self.fitness(p, candidates, request) for p in population]

        gbest_idx: int = pbest_score.index(min(pbest_score))
        gbest: List[int] = pbest[gbest_idx][:]
        gbest_score: float = pbest_score[gbest_idx]

        w, c1, c2 = 0.7, 1.5, 1.5

        for _ in range(num_iterations):
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

                if random.random() < 0.1:
                    mut_idx = random.randint(0, num_vnode - 1)
                    population[i][mut_idx] = random.randint(0, len(candidates[mut_idx]) - 1)

                score: float = self.fitness(population[i], candidates, request)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            current_best_score: float = min(pbest_score)
            if current_best_score < gbest_score:
                gbest = pbest[pbest_score.index(current_best_score)][:]
                gbest_score = current_best_score
            
        return gbest

    def fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]], request: VirtualRequest) -> float:
        vnetwork: VirtualNetwork = request["vnetwork"]
        vnodes: List[VirtualNode] = vnetwork.nodes
        vlinks: List[VirtualLink] = getattr(vnetwork, "vlinks", [])

        mapping: List[SubstrateNode] = [candidates[i][idx] for i, idx in enumerate(particle_idx)]

        node_cost: float = sum(vnode.cpu_demand * snode.cost_per_unit for vnode, snode in zip(vnodes, mapping))
        link_cost: float = 0.0
        for vlink in vlinks:
            src_node: SubstrateNode = mapping[vnodes.index(vlink.src)]
            dst_node: SubstrateNode = mapping[vnodes.index(vlink.dst)]
            link_cost += self.global_controller.shortest_path_cost(src_node, dst_node, bw_required=vlink.bandwidth)

        return node_cost + link_cost