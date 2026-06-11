"""MP-VNE V4 — paper-faithful PSO operators + paper hyperparameters.

Paper: https://arxiv.org/pdf/2202.12830 (Zhang et al., "MP-VNE")

V4 implements PSO exactly as paper describes:

  Eq. 3:  v_new = α·v + β·rand1·δ(x_pb, x) + γ·rand2·δ(x_gb, x)
  Eq. 4:  x_new = x + v_new

  where δ(a,b) = 0 if a == b else 1     ← paper page 8 point 7

Paper hyperparams (page 13):
  particles=10, iterations=50, α=0.3, β=0.3, γ=0.4, mutation=10%
  + paper Algorithm 2 says "resetParticalPosition" on mutation (whole particle).

Candidate selection: paper-accurate top-K total (mp_vne_v2 strategy).
Multi-restart: K=3 to be fair vs V16/V17.
"""
import random
from typing import List, Dict

from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution

from algorithms.mp_vne.global_controller import GlobalController as GCPerDom  # top-K per domain
from algorithms.mp_vne.legacy import MPVNELegacy


# Paper hyperparameters
PAPER_PARTICLES = 10
PAPER_ITERATIONS = 50
PAPER_ALPHA = 0.3   # inertia
PAPER_BETA = 0.3    # cognitive (c1)
PAPER_GAMMA = 0.4   # social (c2)
PAPER_MUTATION = 0.10

NUM_RESTARTS = 3    # multi-restart wrapper to match V16/V17


class MPVNE(MPVNELegacy):
    NUM_RESTARTS = NUM_RESTARTS

    def __init__(self):
        super().__init__()
        self.name = "MP-VNE"
        # Override config with paper-faithful values.
        self.config["pso"] = {
            "num_particles": PAPER_PARTICLES,
            "num_iterations": PAPER_ITERATIONS,
            "alpha": PAPER_ALPHA,
            "beta": PAPER_BETA,
            "gamma": PAPER_GAMMA,
            "mutation_rate": PAPER_MUTATION,
        }
        self.config["candidate_selection"] = {"top_k": 1}  # top-1 per domain (paper-strict)

    def solve(self, substrate_network, virtual_request):
        # Cache GlobalController (per-domain top-K selection, paper-strict).
        if getattr(self, "global_controller", None) is None:
            self.global_controller = GCPerDom(substrate_network)
        self._release_expired(virtual_request.arrival_time)
        self.global_controller.clear_caches()

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        top_k = self.config["candidate_selection"]["top_k"]
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

        # Multi-restart wrapper (K=3) — fair vs V16/V17.
        best_particle = None
        best_score = float("inf")
        master_seed = getattr(self, "_master_seed", 42)
        for k in range(self.NUM_RESTARTS):
            random.seed(k * 1337 + master_seed)
            particle = self.pso_paper(candidate_nodes, virtual_request, vlink_indices)
            score = self.fitness(particle, candidate_nodes, virtual_request, vlink_indices)
            if score < best_score:
                best_score = score
                best_particle = particle
        if best_particle is None:
            best_particle = particle  # fallback to last

        best_mapping = {
            vnodes_list[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle)
        }

        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vnetwork)
            if not vlink_paths:
                raise ValueError("No paths allocated")
            solution.is_successful = True
        except ValueError:
            return solution

        cost = self.fitness(best_particle, candidate_nodes, virtual_request, vlink_indices)

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

    def pso_paper(self, candidates, request, vlink_indices):
        """Paper-faithful PSO with BOTH operators redefined per paper page 8:

        Point 7 (minus δ):
          a - b = 0 if a == b else 1

        Point 8 (plus, binary threshold at 0.5):
          a + b = 1 if (a + b) > 0.5 else 0

        Eq.3:  v_new = α·v ⊕ β·r1·δ(x_pb, x) ⊕ γ·r2·δ(x_gb, x)
        Eq.4:  x_new = x ⊕ v_new      (interpreted: v=1 → change x toward best)

        Velocity therefore lives in {0, 1} after each update.
        """
        cfg = self.config["pso"]
        num_particles = cfg["num_particles"]
        num_iterations = cfg["num_iterations"]
        alpha = cfg["alpha"]
        beta = cfg["beta"]
        gamma = cfg["gamma"]
        mutation_rate = cfg["mutation_rate"]
        num_vnode = len(candidates)

        # Initialize population (random indices into per-vnode candidate lists).
        population: List[List[int]] = [
            [random.randint(0, len(candidates[j]) - 1) for j in range(num_vnode)]
            for _ in range(num_particles)
        ]
        # Velocity is binary {0, 1} per dimension under paper's redefined +.
        velocities: List[List[int]] = [[0] * num_vnode for _ in range(num_particles)]

        pbest = [p[:] for p in population]
        pbest_score = [self.fitness(p, candidates, request, vlink_indices) for p in population]

        gbest_idx = pbest_score.index(min(pbest_score))
        gbest = pbest[gbest_idx][:]
        gbest_score = pbest_score[gbest_idx]

        for _ in range(num_iterations):
            for i in range(num_particles):
                # Eq.3 with binary δ (minus) and binary threshold + (plus).
                for j in range(num_vnode):
                    r1, r2 = random.random(), random.random()
                    delta_pb = 0 if pbest[i][j] == population[i][j] else 1
                    delta_gb = 0 if gbest[j] == population[i][j] else 1
                    raw_sum = (
                        alpha * velocities[i][j]
                        + beta * r1 * delta_pb
                        + gamma * r2 * delta_gb
                    )
                    # Paper page 8 point 8: plus → 1 if >0.5 else 0.
                    velocities[i][j] = 1 if raw_sum > 0.5 else 0

                # Eq.4: x_new = x ⊕ v_new.
                # When v_new[j] == 1, position moves toward best.
                # Source of move: probabilistic blend of pbest and gbest,
                # weighted by their respective coefficients β and γ.
                for j in range(num_vnode):
                    if velocities[i][j] == 1:
                        if random.random() < beta / (beta + gamma):
                            population[i][j] = pbest[i][j]
                        else:
                            population[i][j] = gbest[j]
                    # velocity 0 → keep x[j]

                # Genetic variation factor (Algorithm 2 line 6-9).
                if random.random() < mutation_rate:
                    population[i] = [
                        random.randint(0, len(candidates[j]) - 1)
                        for j in range(num_vnode)
                    ]

                score = self.fitness(population[i], candidates, request, vlink_indices)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score
                if score < gbest_score:
                    gbest = population[i][:]
                    gbest_score = score

        return gbest
