"""MC-VNM — multi-domain VN mapping baseline (greedy node + Kruskal/MST link).

This is the MC-VNM heuristic cited in the MP-VNE literature, ported to this
repo's multi-domain data model (``MultiDomainNetwork``) and wired into the
standard eval harness so it can be compared against MP-VNE and CARL-VNE on the
same metrics.

Two stages per VNR:

  1. Node mapping (greedy, resource- and domain-aware):
     each virtual node is placed on the substrate node with the largest
     remaining CPU among nodes that (a) satisfy ``allowed_domains``,
     (b) still have CPU >= demand, and (c) are not already used by another
     virtual node of the *same* VN (single-host-per-vnode constraint).

  2. Link mapping (Kruskal MST + tree path):
     for each virtual link, build a minimum-cost spanning tree over the whole
     multi-domain substrate (intra-domain + inter-domain links) using only
     links whose remaining bandwidth >= demand, grown until the two endpoints
     are connected, then take the unique src->dst path inside that tree.

The original reference implementation targeted a non-existent ``src.types``
model and had three multi-domain bugs that are fixed here:
  * it could map two virtual nodes of one VN to the same substrate node;
  * it never persisted/released resources across the VNR stream;
  * its Kruskal pass ignored this repo's inter-domain-link registry.

Resource state (``available_cpu`` / ``available_bw``) is held directly on the
substrate objects, persists across the request stream, and is released when a
VNR's lifetime expires, mirroring the controllers used by MP-VNE.
"""
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

from problem.domain import MultiDomainNetwork, PhysicalDomain
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution


class MCVNM:
    def __init__(self):
        self.name = "MC-VNM"
        self.substrate: Optional[MultiDomainNetwork] = None
        # node_id -> domain_id, for allowed_domains filtering
        self._node_domain: Dict[str, str] = {}
        # request_id -> {"node_mapping", "link_mapping", "expire_time"}
        self._active_mappings: "OrderedDict[str, Dict]" = OrderedDict()

    # ---------------- setup ----------------
    def _init_controller(self, substrate_network) -> None:
        """Bind to the substrate and initialise per-object resource state once."""
        if not hasattr(substrate_network, "domains"):
            # Wrap a flat SubstrateNetwork as a single-domain MD network.
            domain = PhysicalDomain(id="domain_0", network=substrate_network)
            md = MultiDomainNetwork(domains={"domain_0": domain})
        else:
            md = substrate_network
        self.substrate = md

        self._node_domain = {}
        for dom_id, domain in md.domains.items():
            for node in domain.network.nodes.values():
                if not hasattr(node, "available_cpu"):
                    node.available_cpu = node.cpu_capacity
                self._node_domain[node.id] = dom_id
            for link in domain.network.links.values():
                if not hasattr(link, "available_bw"):
                    link.available_bw = link.bandwidth_capacity
        for link in md.inter_domain_links.values():
            if not hasattr(link, "available_bw"):
                link.available_bw = link.bandwidth_capacity

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

        # Record for later release.
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

    # ---------------- node mapping ----------------
    def _node_mapping(self, vnetwork: VirtualNetwork) -> Optional[Dict[str, str]]:
        """Greedy max-remaining-CPU placement with a distinct-node constraint."""
        mapping: Dict[str, str] = {}
        used: set = set()
        # tentative CPU decrements so two vnodes never overcommit one snode
        temp_cpu: Dict[str, float] = {}

        # Largest demands first: improves feasibility under the greedy rule.
        vnodes = sorted(vnetwork.nodes.values(), key=lambda v: v.cpu_demand, reverse=True)
        for vnode in vnodes:
            best: Optional[SubstrateNode] = None
            best_avail = -1.0
            for dom_id, domain in self.substrate.domains.items():
                if vnode.allowed_domains and dom_id not in vnode.allowed_domains:
                    continue
                for snode in domain.network.nodes.values():
                    if snode.id in used:
                        continue
                    avail = snode.available_cpu - temp_cpu.get(snode.id, 0.0)
                    if avail >= vnode.cpu_demand and avail > best_avail:
                        best_avail = avail
                        best = snode
            if best is None:
                return None
            mapping[vnode.id] = best.id
            used.add(best.id)
            temp_cpu[best.id] = temp_cpu.get(best.id, 0.0) + vnode.cpu_demand
        return mapping

    # ---------------- link mapping ----------------
    def _link_mapping(
        self, vnetwork: VirtualNetwork, node_mapping: Dict[str, str]
    ) -> Optional[Dict[Tuple[str, str], Tuple[List[SubstrateLink], float]]]:
        """Kruskal-MST routing per vlink; tentatively reserves bandwidth so
        concurrent vlinks of the same VN share the substrate consistently."""
        result: Dict[Tuple[str, str], Tuple[List[SubstrateLink], float]] = {}
        temp_bw: Dict[int, float] = {}  # id(link) -> tentatively used bw

        for vkey, vlink in vnetwork.links.items():
            src_id = node_mapping[vlink.source]
            dst_id = node_mapping[vlink.target]
            demand = vlink.bandwidth_demand

            path = self._kruskal_path(src_id, dst_id, demand, temp_bw)
            if path is None:
                return None  # nothing reserved permanently yet; just abort

            for link in path:
                temp_bw[id(link)] = temp_bw.get(id(link), 0.0) + demand
            result[vkey] = (path, demand)
        return result

    def _all_links(self) -> List[SubstrateLink]:
        links: List[SubstrateLink] = []
        for domain in self.substrate.domains.values():
            links.extend(domain.network.links.values())
        links.extend(self.substrate.inter_domain_links.values())
        return links

    def _kruskal_path(
        self, src_id: str, dst_id: str, bandwidth: float, temp_bw: Dict[int, float]
    ) -> Optional[List[SubstrateLink]]:
        """Min-cost spanning tree (Kruskal) over feasible links, then the unique
        tree path between src and dst. Returns None if unreachable."""
        if src_id == dst_id:
            return []

        # 1. Feasible links: remaining bandwidth (minus tentative use) >= demand.
        valid: List[SubstrateLink] = []
        for link in self._all_links():
            avail = link.available_bw - temp_bw.get(id(link), 0.0)
            if avail >= bandwidth:
                valid.append(link)
        if not valid:
            return None

        # 2. Kruskal MST, cheapest links first, until src and dst connect.
        valid.sort(key=lambda l: l.bandwidth_price)
        parent: Dict[str, str] = {}

        def find(u: str) -> str:
            parent.setdefault(u, u)
            root = u
            while parent[root] != root:
                root = parent[root]
            while parent[u] != root:  # path compression
                parent[u], u = root, parent[u]
            return root

        def union(u: str, v: str) -> bool:
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            parent[pu] = pv
            return True

        tree: List[SubstrateLink] = []
        for link in valid:
            if union(link.source, link.target):
                tree.append(link)
            if find(src_id) == find(dst_id):
                break

        if find(src_id) != find(dst_id):
            return None

        # 3. BFS for the unique src->dst path inside the spanning tree.
        adj: Dict[str, List[SubstrateLink]] = {}
        for link in tree:
            adj.setdefault(link.source, []).append(link)
            adj.setdefault(link.target, []).append(link)

        visited = {src_id}
        queue = deque([(src_id, [])])
        while queue:
            node, path = queue.popleft()
            if node == dst_id:
                return path
            for link in adj.get(node, []):
                nxt = link.target if link.source == node else link.source
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [link]))
        return None

    # ---------------- resources ----------------
    def _reserve(self, node_mapping, link_mapping, vnetwork: VirtualNetwork) -> None:
        for vnode_id, snode_id in node_mapping.items():
            snode = self._find_snode(snode_id)
            snode.available_cpu -= vnetwork.nodes[vnode_id].cpu_demand
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
                snode = self._find_snode(snode_id)
                if snode is not None:
                    snode.available_cpu += vnetwork.nodes[vnode_id].cpu_demand
            for path, bw in info["link_mapping"].values():
                for link in path:
                    link.available_bw += bw

    # ---------------- cost ----------------
    def _compute_cost(self, node_mapping, link_mapping, vnetwork: VirtualNetwork) -> float:
        node_cost = 0.0
        for vnode_id, snode_id in node_mapping.items():
            snode = self._find_snode(snode_id)
            node_cost += snode.cpu_price * vnetwork.nodes[vnode_id].cpu_demand
        link_cost = 0.0
        for path, bw in link_mapping.values():
            for link in path:
                link_cost += link.bandwidth_price * bw
        return node_cost + link_cost

    # ---------------- helpers ----------------
    def _find_snode(self, node_id: str) -> Optional[SubstrateNode]:
        dom_id = self._node_domain.get(node_id)
        if dom_id is None:
            return None
        return self.substrate.domains[dom_id].network.nodes.get(node_id)
