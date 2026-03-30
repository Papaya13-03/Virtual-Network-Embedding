import random
import os
import yaml
from typing import List, Dict
from collections import OrderedDict

from algorithms.tarp_vne.tarp_controller import TARPGlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class TARPVNE:
    """
    TARP-VNE: Topology-Aware Resource-Preserving Virtual Network Embedding.

    Improvements over MP-VNE:
    1. Topology-aware node ranking (NRS = NRC + LRC + BC + TP)
    2. Multi-objective fitness (cost + fragmentation + topology + balance)
    3. Adaptive PSO (decreasing inertia, crossover c1/c2, diversity mutation)
    4. Topology-informed initialization (greedy + diversified + random)
    5. Load-aware multi-path routing (congestion-penalized FW during commit)
    6. Accurate fitness: uses full bandwidth in path evaluation (not 10% approx)
    """

    def __init__(self):
        self.name = "TARP-VNE"
        self._active_mappings: Dict[str, Dict] = OrderedDict()

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "tarp_vne.yaml"
        )
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {
                "pso": {
                    "num_particles": 25, "num_iterations": 12,
                    "w_max": 0.9, "w_min": 0.4,
                    "c1_max": 2.5, "c1_min": 0.5,
                    "c2_max": 2.5, "c2_min": 0.5,
                    "base_mutation_rate": 0.1, "elite_fraction": 0.2,
                },
                "ranking": {
                    "top_k": 30, "w_nrc": 0.30, "w_lrc": 0.25,
                    "w_bc": 0.20, "w_tp": 0.25,
                },
                "fitness": {
                    "alpha": 1.0, "beta": 0.15, "gamma": 0.3, "delta": 0.1,
                },
                "routing": {"congestion_weight": 2.0, "max_paths": 5},
                "initialization": {
                    "greedy_fraction": 0.3, "diversified_fraction": 0.4,
                },
            }

    # ════════════════════════════════════════════════════════
    # Public interface
    # ════════════════════════════════════════════════════════

    def _release_expired(self, current_time: float) -> None:
        expired_ids = [rid for rid, data in self._active_mappings.items()
                       if data["expire_time"] <= current_time]
        for expired_id in expired_ids:
            data = self._active_mappings.pop(expired_id)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"]
            )

    def solve(self, substrate_network: SubstrateNetwork,
              virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:

        congestion_w = self.config.get("routing", {}).get("congestion_weight", 2.0)
        self.global_controller = TARPGlobalController(substrate_network, congestion_w)
        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        # Phase 1: Precompute structural features
        bc = self.global_controller.compute_betweenness_centrality()
        self.global_controller.build_node_domain_map()
        self.global_controller.precompute_node_info()

        # Phase 2: Candidate selection with topology-aware ranking
        candidate_nodes = self._ranked_candidates(vnetwork, bc)
        if any(not c for c in candidate_nodes):
            return solution

        # Pre-calculate virtual link endpoint indices
        vnodes_list = list(vnetwork.nodes.values())
        vnode_to_idx = {vnode.id: i for i, vnode in enumerate(vnodes_list)}
        vlink_indices = []
        for vlink in vnetwork.links.values():
            vlink_indices.append({
                "src_idx": vnode_to_idx[vlink.source],
                "dst_idx": vnode_to_idx[vlink.target],
                "bw": vlink.bandwidth_demand,
            })

        # Phases 3-4: Adaptive PSO with multi-objective fitness
        best_particle_idx = self._adaptive_pso(
            candidate_nodes, virtual_request, vlink_indices, vnetwork
        )

        # Build node mapping
        vnodes = list(vnetwork.nodes.values())
        best_mapping = {
            vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle_idx)
        }

        # Phase 5: Commit with load-aware multi-path routing
        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vnetwork)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            return solution

        cost = self._embedding_cost(best_particle_idx, candidate_nodes,
                                    virtual_request, vlink_indices)

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

    # ════════════════════════════════════════════════════════
    # Phase 2: Topology-aware candidate ranking
    # ════════════════════════════════════════════════════════

    def _ranked_candidates(self, vnetwork: VirtualNetwork,
                           bc: Dict[str, float]) -> List[List[SubstrateNode]]:
        """
        For each virtual node, collect substrate candidates, score them
        with NRS (Node Ranking Score), and return top-K ranked.
        """
        cfg = self.config.get("ranking", {})
        top_k = cfg.get("top_k", 30)
        w_nrc = cfg.get("w_nrc", 0.30)
        w_lrc = cfg.get("w_lrc", 0.25)
        w_bc = cfg.get("w_bc", 0.20)
        w_tp = cfg.get("w_tp", 0.25)

        all_candidates = []
        for vnode in vnetwork.nodes.values():
            # Collect raw candidates (CPU-feasible, domain-feasible)
            raw = []
            for lc in self.global_controller.local_controllers:
                if not vnode.allowed_domains or lc.domain.id in vnode.allowed_domains:
                    raw.extend(lc.get_candidates(vnode))

            # Score each candidate with NRS
            scored = []
            for snode in raw:
                avail_cpu = getattr(snode, 'available_cpu', snode.cpu_capacity)
                nrc = avail_cpu / snode.cpu_capacity if snode.cpu_capacity > 0 else 0
                lrc, _ = self.global_controller.get_node_lrc_and_degree(snode.id)
                bc_score = bc.get(snode.id, 0.0)
                tp = self.global_controller.get_topological_proximity(
                    snode.id, vnode, vnetwork
                )
                nrs = w_nrc * nrc + w_lrc * lrc + w_bc * bc_score + w_tp * tp
                scored.append((nrs, snode))

            # Sort by NRS descending, keep top-K
            scored.sort(key=lambda x: -x[0])
            all_candidates.append([snode for _, snode in scored[:top_k]])

        return all_candidates

    # ════════════════════════════════════════════════════════
    # Phase 3: Topology-informed PSO initialization
    # ════════════════════════════════════════════════════════

    def _greedy_seed(self, candidates: List[List[SubstrateNode]],
                     vnetwork: VirtualNetwork) -> List[int]:
        """
        Seed one particle greedily: assign highest-ranked candidates to
        highest-degree virtual nodes first (candidates are already ranked).
        """
        num_vnodes = len(candidates)
        vnodes = list(vnetwork.nodes.values())

        vnode_degrees = []
        for i, vnode in enumerate(vnodes):
            degree = sum(1 for l in vnetwork.links.values()
                         if l.source == vnode.id or l.target == vnode.id)
            vnode_degrees.append((i, degree))
        vnode_degrees.sort(key=lambda x: -x[1])

        particle = [0] * num_vnodes
        used_snodes = set()

        for vnode_idx, _ in vnode_degrees:
            assigned = False
            for cand_idx in range(len(candidates[vnode_idx])):
                snode = candidates[vnode_idx][cand_idx]
                if snode.id not in used_snodes:
                    particle[vnode_idx] = cand_idx
                    used_snodes.add(snode.id)
                    assigned = True
                    break
            if not assigned and candidates[vnode_idx]:
                particle[vnode_idx] = random.randint(0, len(candidates[vnode_idx]) - 1)

        return particle

    def _diversified_seed(self, candidates: List[List[SubstrateNode]]) -> List[int]:
        """
        Seed one particle with domain diversity: for each virtual node,
        pick a candidate from a random domain to spread load.
        """
        num_vnodes = len(candidates)
        particle = [0] * num_vnodes
        used_snodes = set()

        for i in range(num_vnodes):
            # Group candidates by domain
            domain_groups: Dict[str, List[int]] = {}
            for cand_idx, snode in enumerate(candidates[i]):
                domain_id = self.global_controller.get_node_domain(snode.id)
                domain_groups.setdefault(domain_id, []).append(cand_idx)

            domains = list(domain_groups.keys())
            random.shuffle(domains)
            assigned = False

            for d in domains:
                cands = domain_groups[d]
                random.shuffle(cands)
                for cand_idx in cands:
                    if candidates[i][cand_idx].id not in used_snodes:
                        particle[i] = cand_idx
                        used_snodes.add(candidates[i][cand_idx].id)
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned and candidates[i]:
                particle[i] = random.randint(0, len(candidates[i]) - 1)

        return particle

    # ════════════════════════════════════════════════════════
    # Phase 4: Adaptive PSO
    # ════════════════════════════════════════════════════════

    def _adaptive_pso(self, candidates: List[List[SubstrateNode]],
                      request: VirtualNetworkRequest,
                      vlink_indices: List[Dict],
                      vnetwork: VirtualNetwork) -> List[int]:
        pso = self.config.get("pso", {})
        num_particles = pso.get("num_particles", 25)
        num_iterations = pso.get("num_iterations", 12)
        w_max = pso.get("w_max", 0.9)
        w_min = pso.get("w_min", 0.4)
        c1_max = pso.get("c1_max", 2.5)
        c1_min = pso.get("c1_min", 0.5)
        c2_max = pso.get("c2_max", 2.5)
        c2_min = pso.get("c2_min", 0.5)
        base_mut = pso.get("base_mutation_rate", 0.1)
        elite_frac = pso.get("elite_fraction", 0.2)

        init_cfg = self.config.get("initialization", {})
        greedy_frac = init_cfg.get("greedy_fraction", 0.3)
        div_frac = init_cfg.get("diversified_fraction", 0.4)

        num_vnode = len(candidates)

        # ── Topology-informed initialization ──
        num_greedy = max(1, int(num_particles * greedy_frac))
        num_diversified = max(1, int(num_particles * div_frac))
        num_random = num_particles - num_greedy - num_diversified

        population: List[List[int]] = []
        for _ in range(num_greedy):
            population.append(self._greedy_seed(candidates, vnetwork))
        for _ in range(num_diversified):
            population.append(self._diversified_seed(candidates))
        for _ in range(num_random):
            population.append(
                [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)]
            )

        velocities = [[0.0] * num_vnode for _ in range(num_particles)]

        # Evaluate initial population
        pbest = [p[:] for p in population]
        pbest_score = [
            self._multi_objective_fitness(p, candidates, request, vlink_indices, vnetwork)
            for p in population
        ]

        gbest_idx = pbest_score.index(min(pbest_score))
        gbest = pbest[gbest_idx][:]
        gbest_score = pbest_score[gbest_idx]

        num_elite = max(1, int(num_particles * elite_frac))

        # ── Main PSO loop ──
        for t in range(num_iterations):
            print(f"  TARP-PSO Iteration {t + 1}/{num_iterations}...")

            # Adaptive parameters
            progress = t / max(1, num_iterations - 1)
            w = w_max - (w_max - w_min) * progress
            c1 = c1_max - (c1_max - c1_min) * progress  # exploration → exploitation
            c2 = c2_min + (c2_max - c2_min) * progress   # weak → strong social pull

            # Diversity-aware mutation rate
            diversity = self._compute_diversity(population)
            mutation_rate = base_mut * max(0.5, 1.0 - diversity)

            # Identify elite particles (top by fitness)
            ranked = sorted(range(num_particles), key=lambda i: pbest_score[i])
            elite_set = set(ranked[:num_elite])

            for i in range(num_particles):
                # Velocity and position update
                for j in range(num_vnode):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - population[i][j])
                        + c2 * r2 * (gbest[j] - population[i][j])
                    )
                    new_idx = int(round(population[i][j] + velocities[i][j]))
                    population[i][j] = new_idx % len(candidates[j])

                # Diversity-aware mutation (elites exempt)
                if i not in elite_set and random.random() < mutation_rate:
                    mut_j = random.randint(0, num_vnode - 1)
                    population[i][mut_j] = random.randint(
                        0, len(candidates[mut_j]) - 1
                    )

                # Evaluate fitness
                score = self._multi_objective_fitness(
                    population[i], candidates, request, vlink_indices, vnetwork
                )
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score

            # Update global best
            current_best = min(pbest_score)
            if current_best < gbest_score:
                gbest = pbest[pbest_score.index(current_best)][:]
                gbest_score = current_best

        return gbest

    # ════════════════════════════════════════════════════════
    # Multi-objective fitness function
    # ════════════════════════════════════════════════════════

    def _multi_objective_fitness(self, particle_idx: List[int],
                                 candidates: List[List[SubstrateNode]],
                                 request: VirtualNetworkRequest,
                                 vlink_indices: List[Dict],
                                 vnetwork: VirtualNetwork) -> float:
        """
        F = α·C_embed + β·R_frag + γ·T_align + δ·L_balance

        Key difference from MP-VNE: uses FULL bandwidth for path
        computation, not the 10% approximation.
        """
        fcfg = self.config.get("fitness", {})
        alpha = fcfg.get("alpha", 1.0)
        beta = fcfg.get("beta", 0.15)
        gamma = fcfg.get("gamma", 0.3)
        delta = fcfg.get("delta", 0.1)

        vnodes = list(vnetwork.nodes.values())
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]

        # ── Injectivity check ──
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping):
            return float('inf')

        # ── 1. Embedding cost with accurate bandwidth ──
        node_cost = sum(
            vn.cpu_demand * sn.cpu_price for vn, sn in zip(vnodes, mapping)
        )

        link_cost = 0.0
        hop_total = 0

        for vl in vlink_indices:
            src_node = mapping[vl["src_idx"]]
            dst_node = mapping[vl["dst_idx"]]

            # FULL bandwidth (critical improvement over MP-VNE's bw*0.1)
            path = self.global_controller.shortest_path(
                src_node, dst_node, bw_required=vl["bw"]
            )
            if not path:
                return float('inf')

            link_cost += sum(
                l.transmission_delay + l.bandwidth_price * vl["bw"] for l in path
            )
            hop_total += len(path)

        c_embed = node_cost + link_cost

        # ── 2. Resource fragmentation penalty (node-level) ──
        r_frag = 0.0
        for vn, sn in zip(vnodes, mapping):
            avail = getattr(sn, 'available_cpu', sn.cpu_capacity)
            cap = sn.cpu_capacity
            if cap > 0:
                util_after = 1.0 - (avail - vn.cpu_demand) / cap
                r_frag += util_after ** 2

        # ── 3. Topology alignment (total hop count) ──
        t_align = float(hop_total)

        # ── 4. Load balance across domains ──
        domain_counts: Dict[str, int] = {}
        for sn in mapping:
            d = self.global_controller.get_node_domain(sn.id)
            domain_counts[d] = domain_counts.get(d, 0) + 1

        l_balance = 0.0
        if len(domain_counts) > 1:
            counts = list(domain_counts.values())
            mean_c = sum(counts) / len(counts)
            variance = sum((c - mean_c) ** 2 for c in counts) / len(counts)
            l_balance = variance ** 0.5

        return alpha * c_embed + beta * r_frag + gamma * t_align + delta * l_balance

    # ════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════

    def _embedding_cost(self, particle_idx: List[int],
                        candidates: List[List[SubstrateNode]],
                        request: VirtualNetworkRequest,
                        vlink_indices: List[Dict]) -> float:
        """Pure embedding cost (for solution reporting, no penalty terms)."""
        vnetwork = request.virtual_network
        vnodes = list(vnetwork.nodes.values())
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]

        node_cost = sum(
            vn.cpu_demand * sn.cpu_price for vn, sn in zip(vnodes, mapping)
        )
        link_cost = 0.0

        for vl in vlink_indices:
            src_node = mapping[vl["src_idx"]]
            dst_node = mapping[vl["dst_idx"]]
            path = self.global_controller.shortest_path(
                src_node, dst_node, bw_required=vl["bw"]
            )
            if path:
                link_cost += sum(
                    l.transmission_delay + l.bandwidth_price * vl["bw"]
                    for l in path
                )

        return node_cost + link_cost

    @staticmethod
    def _compute_diversity(population: List[List[int]]) -> float:
        """Normalized average pairwise Hamming distance ∈ [0, 1]."""
        n = len(population)
        if n < 2:
            return 1.0
        dims = len(population[0])
        if dims == 0:
            return 1.0

        total_dist = 0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += sum(
                    1 for k in range(dims) if population[i][k] != population[j][k]
                )
                pairs += 1

        return (total_dist / pairs / dims) if pairs > 0 else 1.0
