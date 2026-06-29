"""VNE-PSO — vanilla PSO node placement + Dijkstra link mapping baseline.

A generic particle-swarm VNE heuristic, ported to this repo's multi-domain
data model (``MultiDomainNetwork``) and wired into the standard eval harness so
it can be compared against MC-VNM, MP-VNE and CARL-VNE on the same metrics.

It differs from MP-VNE (which is also PSO-based) in three ways, making it a
distinct, more "vanilla" PSO baseline:
  * the swarm searches over **all** substrate nodes, with no per-domain
    PreCost top-k candidate pruning;
  * continuous index-based velocity PSO (vs MP-VNE's paper-faithful binary PSO);
  * Dijkstra shortest-path link mapping (vs Floyd-Warshall).

Two stages per VNR:
  1. Node mapping (PSO): each particle is a full virtual->substrate node
     assignment (indices into the global substrate-node list). Fitness =
     node CPU cost, with heavy penalties for (a) two virtual nodes of the same
     VN sharing a substrate node and (b) CPU overcommit; ``allowed_domains`` is
     respected when drawing feasible candidates. The best collision-free,
     feasible assignment found becomes the node mapping.
  2. Link mapping (Dijkstra): each virtual link is routed on the cheapest
     substrate path (intra-domain + inter-domain links) whose every hop has
     remaining bandwidth >= demand.

The original reference implementation targeted a non-existent ``src.types``
model and had several multi-domain bugs fixed here:
  * it allowed two virtual nodes of one VN on the same substrate node;
  * it released the wrong bandwidth amount (link capacity, not reserved demand);
  * it ignored ``allowed_domains`` and this repo's inter-domain-link registry.

Resource state (``available_cpu`` / ``available_bw``) is held on the substrate
objects, persists across the request stream, and is released on VNR expiry,
mirroring the other algorithms.
"""
import heapq
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from problem.domain import MultiDomainNetwork, PhysicalDomain
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


COLLISION_PENALTY = 1e6
OVERCOMMIT_PENALTY = 1e6


class VNEPSO:
    def __init__(self, num_particles: int = 20, max_iterations: int = 50,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 patience: int = 8, v_max: int = 4):
        self.name = "VNE-PSO"
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.patience = patience
        self.v_max = v_max

        self.substrate: Optional[MultiDomainNetwork] = None
        self._node_domain: Dict[str, str] = {}      # node_id -> domain_id
        self._snodes: List[SubstrateNode] = []        # global ordered node list
        self._snode_pos: Dict[str, int] = {}          # node_id -> index in _snodes
        self._active_mappings: "OrderedDict[str, Dict]" = OrderedDict()

    # ---------------- setup ----------------
    def _init_controller(self, substrate_network) -> None:
        if not hasattr(substrate_network, "domains"):
            domain = PhysicalDomain(id="domain_0", network=substrate_network)
            md = MultiDomainNetwork(domains={"domain_0": domain})
        else:
            md = substrate_network
        self.substrate = md

        self._node_domain = {}
        self._snodes = []
        for dom_id, domain in md.domains.items():
            for node in domain.network.nodes.values():
                if not hasattr(node, "available_cpu"):
                    node.available_cpu = node.cpu_capacity
                self._node_domain[node.id] = dom_id
                self._snodes.append(node)
            for link in domain.network.links.values():
                if not hasattr(link, "available_bw"):
                    link.available_bw = link.bandwidth_capacity
        for link in md.inter_domain_links.values():
            if not hasattr(link, "available_bw"):
                link.available_bw = link.bandwidth_capacity
        self._snode_pos = {n.id: i for i, n in enumerate(self._snodes)}

    # ---------------- main entry ----------------
    def solve(self, substrate_network, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.substrate is not substrate_network:
            self._init_controller(substrate_network)

        self._release_expired(virtual_request.arrival_time)

        vnetwork = virtual_request.virtual_network
        solution = EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        node_mapping = self._node_mapping(vnetwork)
        if node_mapping is None:
            return solution

        link_mapping = self._link_mapping(vnetwork, node_mapping)
        if link_mapping is None:
            return solution

        self._reserve(node_mapping, link_mapping, vnetwork)
        self._active_mappings[virtual_request.id] = {
            "node_mapping": node_mapping,
            "link_mapping": link_mapping,
            "vnetwork": vnetwork,
            "expire_time": virtual_request.arrival_time + virtual_request.lifetime,
        }

        solution.is_successful = True
        solution.node_mapping = node_mapping
        solution.embedding_cost = self._compute_cost(node_mapping, link_mapping, vnetwork)
        solution.link_mapping = {
            vkey: [([(l.source, l.target) for l in path], bw)]
            for vkey, (path, bw) in link_mapping.items()
        }
        return solution

    # ---------------- node mapping (PSO) ----------------
    def _feasible_candidates(self, vnode: VirtualNode) -> List[int]:
        """Indices (into _snodes) of nodes that satisfy allowed_domains and CPU."""
        out = []
        for i, sn in enumerate(self._snodes):
            if vnode.allowed_domains and self._node_domain[sn.id] not in vnode.allowed_domains:
                continue
            if sn.available_cpu >= vnode.cpu_demand:
                out.append(i)
        return out

    def _node_mapping(self, vnetwork: VirtualNetwork) -> Optional[Dict[str, str]]:
        vnodes = list(vnetwork.nodes.values())
        demands = [v.cpu_demand for v in vnodes]

        cand: List[List[int]] = [self._feasible_candidates(v) for v in vnodes]
        if any(len(c) == 0 for c in cand):
            return None  # some vnode cannot be hosted at all

        n = len(vnodes)
        particles: List[List[int]] = [
            [random.choice(cand[j]) for j in range(n)]
            for _ in range(self.num_particles)
        ]
        velocities: List[List[float]] = [[0.0] * n for _ in range(self.num_particles)]

        def fitness(pos: List[int]) -> float:
            cost = 0.0
            usage: Dict[int, float] = {}
            count: Dict[int, int] = {}
            for j, idx in enumerate(pos):
                usage[idx] = usage.get(idx, 0.0) + demands[j]
                count[idx] = count.get(idx, 0) + 1
                cost += self._snodes[idx].cpu_price * demands[j]
            for idx, used in usage.items():
                if count[idx] > 1:
                    cost += COLLISION_PENALTY  # two vnodes of this VN share one snode
                if used > self._snodes[idx].available_cpu + 1e-9:
                    cost += OVERCOMMIT_PENALTY
            return cost

        pbest = [p[:] for p in particles]
        pbest_fit = [fitness(p) for p in particles]
        gi = pbest_fit.index(min(pbest_fit))
        gbest = pbest[gi][:]
        gbest_fit = pbest_fit[gi]

        stagnant = 0
        for _ in range(self.max_iterations):
            prev = gbest_fit
            for i in range(self.num_particles):
                for j in range(n):
                    r1, r2 = random.random(), random.random()
                    cur = particles[i][j]
                    vel = (self.w * velocities[i][j]
                           + self.c1 * r1 * (pbest[i][j] - cur)
                           + self.c2 * r2 * (gbest[j] - cur))
                    vel = max(-self.v_max, min(self.v_max, vel))
                    velocities[i][j] = vel
                    new_idx = cur + int(round(vel))
                    # snap to a feasible candidate for this vnode
                    if new_idx in cand[j]:
                        particles[i][j] = new_idx
                    else:
                        particles[i][j] = min(cand[j], key=lambda c: abs(c - new_idx))
                fit = fitness(particles[i])
                if fit < pbest_fit[i]:
                    pbest[i] = particles[i][:]
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = particles[i][:]
                        gbest_fit = fit
            stagnant = stagnant + 1 if gbest_fit >= prev else 0
            if stagnant >= self.patience:
                break

        # Reject if the best assignment still collides or overcommits.
        if gbest_fit >= min(COLLISION_PENALTY, OVERCOMMIT_PENALTY):
            return None
        if len(set(gbest)) != n:
            return None
        return {vnodes[j].id: self._snodes[gbest[j]].id for j in range(n)}

    # ---------------- link mapping (Dijkstra) ----------------
    def _link_mapping(
        self, vnetwork: VirtualNetwork, node_mapping: Dict[str, str]
    ) -> Optional[Dict[Tuple[str, str], Tuple[List[SubstrateLink], float]]]:
        result: Dict[Tuple[str, str], Tuple[List[SubstrateLink], float]] = {}
        temp_bw: Dict[int, float] = {}  # id(link) -> tentatively reserved bw
        for vkey, vlink in vnetwork.links.items():
            src_id = node_mapping[vlink.source]
            dst_id = node_mapping[vlink.target]
            path = self._dijkstra(src_id, dst_id, vlink.bandwidth_demand, temp_bw)
            if path is None:
                return None
            for link in path:
                temp_bw[id(link)] = temp_bw.get(id(link), 0.0) + vlink.bandwidth_demand
            result[vkey] = (path, vlink.bandwidth_demand)
        return result

    def _all_links(self) -> List[SubstrateLink]:
        links: List[SubstrateLink] = []
        for domain in self.substrate.domains.values():
            links.extend(domain.network.links.values())
        links.extend(self.substrate.inter_domain_links.values())
        return links

    def _dijkstra(
        self, src_id: str, dst_id: str, bandwidth: float, temp_bw: Dict[int, float]
    ) -> Optional[List[SubstrateLink]]:
        if src_id == dst_id:
            return []

        adj: Dict[str, List[SubstrateLink]] = {}
        for link in self._all_links():
            if link.available_bw - temp_bw.get(id(link), 0.0) >= bandwidth:
                adj.setdefault(link.source, []).append(link)
                adj.setdefault(link.target, []).append(link)

        counter = 0
        heap = [(0.0, counter, src_id, [])]  # (cost, tiebreak, node_id, path)
        visited = set()
        while heap:
            cost, _, node, path = heapq.heappop(heap)
            if node == dst_id:
                return path
            if node in visited:
                continue
            visited.add(node)
            for link in adj.get(node, []):
                nxt = link.target if link.source == node else link.source
                if nxt in visited:
                    continue
                edge = link.transmission_delay + link.bandwidth_price * bandwidth
                counter += 1
                heapq.heappush(heap, (cost + edge, counter, nxt, path + [link]))
        return None

    # ---------------- resources ----------------
    def _reserve(self, node_mapping, link_mapping, vnetwork: VirtualNetwork) -> None:
        for vnode_id, snode_id in node_mapping.items():
            self._find_snode(snode_id).available_cpu -= vnetwork.nodes[vnode_id].cpu_demand
        for path, bw in link_mapping.values():
            for link in path:
                link.available_bw -= bw

    def _release_expired(self, current_time: float) -> None:
        expired = [rid for rid, info in self._active_mappings.items()
                   if info["expire_time"] <= current_time]
        for rid in expired:
            info = self._active_mappings.pop(rid)
            vnetwork = info["vnetwork"]
            for vnode_id, snode_id in info["node_mapping"].items():
                sn = self._find_snode(snode_id)
                if sn is not None:
                    sn.available_cpu += vnetwork.nodes[vnode_id].cpu_demand
            for path, bw in info["link_mapping"].values():
                for link in path:
                    link.available_bw += bw

    # ---------------- cost ----------------
    def _compute_cost(self, node_mapping, link_mapping, vnetwork: VirtualNetwork) -> float:
        node_cost = sum(
            self._find_snode(snode_id).cpu_price * vnetwork.nodes[vnode_id].cpu_demand
            for vnode_id, snode_id in node_mapping.items()
        )
        link_cost = sum(
            link.bandwidth_price * bw
            for path, bw in link_mapping.values() for link in path
        )
        return node_cost + link_cost

    # ---------------- helpers ----------------
    def _find_snode(self, node_id: str) -> Optional[SubstrateNode]:
        dom_id = self._node_domain.get(node_id)
        if dom_id is None:
            return None
        return self.substrate.domains[dom_id].network.nodes.get(node_id)
