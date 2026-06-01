"""mp_vne_MR — mp_vne (extension) + multi-restart PSO (K=3).

Purpose: fair-compare V16 vs mp_vne where the ONLY difference is the ranking
criterion (NN vs PreCost). Everything else equalized:
  - Candidate strategy: per-domain top-K (same as V16, V16 inherited from mp_vne)
  - Fitness function: full bw_required + transmission_delay + bw_price (same)
  - Multi-restart PSO: K=3 (same as V16)
"""
import random

from algorithms.mp_vne.mp_vne import MPVNE


class MPVNEMR(MPVNE):
    NUM_RESTARTS = 3

    def __init__(self):
        super().__init__()
        self.name = "MP-VNE-MR"

    def solve(self, substrate_network, virtual_request):
        # Override solve to wrap PSO with K-restart, picking best by fitness.
        from problem.embedding_solution import EmbeddingSolution
        from algorithms.mp_vne.global_controller import GlobalController

        # Same GC cache fix as mp_vne_v2.
        if getattr(self, "global_controller", None) is None:
            self.global_controller = GlobalController(substrate_network)
        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        top_k = self.config.get("candidate_selection", {}).get("top_k", 5)
        candidate_nodes = self.global_controller.process_request(vnetwork, top_k=top_k)
        if any(not c for c in candidate_nodes):
            return solution

        vnodes_list = list(vnetwork.nodes.values())
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(vnodes_list)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # Multi-restart PSO: K independent calls, pick by fitness.
        best_particle = None
        best_score = float("inf")
        master_seed = getattr(self, "_master_seed", 42)
        for k in range(self.NUM_RESTARTS):
            random.seed(k * 1337 + master_seed)
            particle = self.pso(candidate_nodes, virtual_request, vlink_indices)
            score = self.fitness(particle, candidate_nodes, virtual_request, vlink_indices)
            if score < best_score:
                best_score = score
                best_particle = particle
        best_particle_idx = best_particle if best_particle is not None else particle

        vnodes = list(vnetwork.nodes.values())
        best_mapping = {
            vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle_idx)
        }

        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vnetwork)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            return solution

        cost = self.fitness(best_particle_idx, candidate_nodes, virtual_request, vlink_indices)

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
