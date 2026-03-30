from typing import List, Dict
from collections import deque

from algorithms.mp_vne.global_controller import GlobalController
from algorithms.mp_vne.local_controller import LocalController
from problem.domain import PhysicalDomain, MultiDomainNetwork
from problem.substrate_network import SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNode, VirtualNetwork


class TARPLocalController(LocalController):
    """
    Extended local controller with load-aware routing.

    When use_cache=False (commit phase), uses congestion-penalized
    Floyd-Warshall to steer bandwidth away from congested links.
    When use_cache=True (fitness evaluation), uses standard FW from parent.
    """

    def __init__(self, domain: PhysicalDomain, congestion_weight: float = 2.0):
        self.congestion_weight = congestion_weight
        super().__init__(domain)

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode,
                      bw_required: float = 0.0, use_cache: bool = True) -> List[SubstrateLink]:
        if use_cache:
            return super().shortest_path(src, dst, bw_required, use_cache)
        # Commit phase: load-aware routing
        return self._load_aware_shortest_path(src, dst, bw_required)

    def _load_aware_shortest_path(self, src: SubstrateNode, dst: SubstrateNode,
                                  bw_required: float) -> List[SubstrateLink]:
        if src.id == dst.id:
            return []

        dist, nxt = self._load_aware_fw(bw_required)

        if nxt[src.id][dst.id] is None:
            return []

        path_nodes = [src.id]
        u = src.id
        while u != dst.id:
            u = nxt[u][dst.id]
            path_nodes.append(u)

        path_links = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            link = (self.domain.network.links.get((u, v))
                    or self.domain.network.links.get((v, u)))
            if link:
                path_links.append(link)

        return path_links

    def _load_aware_fw(self, bw_required: float):
        """Floyd-Warshall with congestion penalty. Never cached."""
        nodes = list(self.domain.network.nodes.keys())
        dist = {u: {v: float('inf') for v in nodes} for u in nodes}
        nxt = {u: {v: None for v in nodes} for u in nodes}

        for u in nodes:
            dist[u][u] = 0

        for (u, v), link in self.domain.network.links.items():
            avail_bw = getattr(link, 'available_bw', link.bandwidth_capacity)
            if avail_bw >= bw_required:
                congestion = (1.0 - avail_bw / link.bandwidth_capacity) ** 2
                cost = (link.transmission_delay
                        + link.bandwidth_price * bw_required
                        + self.congestion_weight * congestion)
                if cost < dist[u][v]:
                    dist[u][v] = cost
                    nxt[u][v] = v
                if cost < dist[v][u]:
                    dist[v][u] = cost
                    nxt[v][u] = u

        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        nxt[i][j] = nxt[i][k]

        return dist, nxt


class TARPGlobalController(GlobalController):
    """
    Extended global controller with:
    - Load-aware routing (congestion-penalized FW during commit)
    - Betweenness centrality computation (Brandes' algorithm)
    - Node neighborhood info precomputation for ranking
    """

    def __init__(self, snetwork, congestion_weight: float = 2.0):
        self.congestion_weight = congestion_weight

        # Replicate parent init with TARPLocalControllers
        if not hasattr(snetwork, 'domains'):
            domain = PhysicalDomain(id="domain_1", network=snetwork)
            self.snetwork = MultiDomainNetwork(domains={"domain_1": domain})
            self.snetwork.inter_domain_links = {}
        else:
            self.snetwork = snetwork

        self.local_controllers = [
            TARPLocalController(d, congestion_weight)
            for d in self.snetwork.domains.values()
        ]
        self._id_fw_cache = {}
        self._initialize_resources()

        # TARP-specific caches
        self._bc_cache = None
        self._node_domain_map = None
        self._node_adj_info = None  # Precomputed adjacency bandwidth info

    # ────────────────── Domain Mapping ──────────────────

    def build_node_domain_map(self):
        """Precompute node_id -> domain_id mapping for fast lookups."""
        if self._node_domain_map is not None:
            return self._node_domain_map
        self._node_domain_map = {}
        for lc in self.local_controllers:
            for node_id in lc.domain.network.nodes:
                self._node_domain_map[node_id] = lc.domain.id
        return self._node_domain_map

    def get_node_domain(self, node_id: str) -> str:
        if self._node_domain_map is None:
            self.build_node_domain_map()
        return self._node_domain_map.get(node_id)

    # ────────────────── Node Info Precomputation ──────────────────

    def precompute_node_info(self):
        """
        Precompute per-node adjacency bandwidth data.
        Stores (available_bw, bw_ratio) for each adjacent link.
        """
        self._node_adj_info = {}

        for lc in self.local_controllers:
            for (u, v), link in lc.domain.network.links.items():
                avail = getattr(link, 'available_bw', link.bandwidth_capacity)
                cap = link.bandwidth_capacity
                ratio = avail / cap if cap > 0 else 0.0
                self._node_adj_info.setdefault(u, []).append((avail, ratio))
                self._node_adj_info.setdefault(v, []).append((avail, ratio))

        for (u, v), link in self.snetwork.inter_domain_links.items():
            avail = getattr(link, 'available_bw', link.bandwidth_capacity)
            cap = link.bandwidth_capacity
            ratio = avail / cap if cap > 0 else 0.0
            self._node_adj_info.setdefault(u, []).append((avail, ratio))
            self._node_adj_info.setdefault(v, []).append((avail, ratio))

    def get_node_lrc_and_degree(self, node_id: str):
        """
        Get Link Residual Capacity (mean bw ratio of adjacent links)
        and degree for a substrate node.
        """
        if self._node_adj_info is None:
            self.precompute_node_info()
        bw_data = self._node_adj_info.get(node_id, [])
        if not bw_data:
            return 0.0, 0
        lrc = sum(r for _, r in bw_data) / len(bw_data)
        return lrc, len(bw_data)

    def get_topological_proximity(self, snode_id: str, vnode: VirtualNode,
                                  vnetwork: VirtualNetwork) -> float:
        """
        Fraction of vnode's neighbors whose BW demand can be satisfied
        by at least one of snode's adjacent links.
        """
        if self._node_adj_info is None:
            self.precompute_node_info()

        bw_data = self._node_adj_info.get(snode_id, [])

        v_bw_demands = []
        for vlink in vnetwork.links.values():
            if vlink.source == vnode.id or vlink.target == vnode.id:
                v_bw_demands.append(vlink.bandwidth_demand)

        if not v_bw_demands:
            return 1.0

        mean_bw = sum(v_bw_demands) / len(v_bw_demands)
        sufficient = sum(1 for avail, _ in bw_data if avail >= mean_bw)
        return min(1.0, sufficient / len(v_bw_demands))

    # ────────────────── Betweenness Centrality ──────────────────

    def compute_betweenness_centrality(self) -> Dict[str, float]:
        """Brandes' algorithm for betweenness centrality. Cached."""
        if self._bc_cache is not None:
            return self._bc_cache

        # Build adjacency list for the full substrate graph
        adj: Dict[str, List[str]] = {}
        all_node_ids = set()

        for lc in self.local_controllers:
            for node_id in lc.domain.network.nodes:
                all_node_ids.add(node_id)
                adj.setdefault(node_id, [])
            for (u, v), _ in lc.domain.network.links.items():
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)

        for (u, v), _ in self.snetwork.inter_domain_links.items():
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        all_nodes = list(all_node_ids)
        bc = {n: 0.0 for n in all_nodes}

        for s in all_nodes:
            # BFS from s
            S = []  # Stack of nodes in order of non-decreasing distance
            P = {w: [] for w in all_nodes}  # Predecessors on shortest paths
            sigma = {w: 0 for w in all_nodes}  # Number of shortest paths
            sigma[s] = 1
            d = {w: -1 for w in all_nodes}  # Distance from s
            d[s] = 0
            Q = deque([s])

            while Q:
                v = Q.popleft()
                S.append(v)
                for w in adj.get(v, []):
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            # Back-propagation of dependencies
            delta = {w: 0.0 for w in all_nodes}
            while S:
                w = S.pop()
                for v in P[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    bc[w] += delta[w]

        # Normalize to [0, 1]
        max_bc = max(bc.values()) if bc and max(bc.values()) > 0 else 1.0
        bc = {k: v / max_bc for k, v in bc.items()}

        self._bc_cache = bc
        return bc

    # ────────────────── Load-Aware Routing ──────────────────

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode,
                      bw_required: float = 0.0, use_cache: bool = True) -> List[SubstrateLink]:
        if use_cache:
            # Standard FW for fitness evaluation
            return super().shortest_path(src, dst, bw_required, use_cache)
        # Load-aware FW for commit phase
        return self._load_aware_shortest_path(src, dst, bw_required)

    def _load_aware_shortest_path(self, src: SubstrateNode, dst: SubstrateNode,
                                  bw_required: float) -> List[SubstrateLink]:
        src_domain = self.get_node_domain(src.id)
        dst_domain = self.get_node_domain(dst.id)

        if src_domain and dst_domain and src_domain == dst_domain:
            lc = self._get_local_controller(src_domain)
            return lc._load_aware_shortest_path(src, dst, bw_required)

        # Inter-domain load-aware FW
        dist, nxt = self._load_aware_interdomain_fw(bw_required)

        if nxt[src.id][dst.id] is None:
            return []

        path_nodes = [src.id]
        u = src.id
        while u != dst.id:
            u = nxt[u][dst.id]
            path_nodes.append(u)

        path_links = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            link = self._find_link(u, v)
            if link:
                path_links.append(link)

        return path_links

    def _load_aware_interdomain_fw(self, bw_required: float):
        """Inter-domain Floyd-Warshall with congestion penalty. Never cached."""
        all_nodes = set()
        for lc in self.local_controllers:
            all_nodes.update(lc.domain.network.nodes.keys())
        all_nodes_list = list(all_nodes)

        dist = {u: {v: float('inf') for v in all_nodes_list} for u in all_nodes_list}
        nxt = {u: {v: None for v in all_nodes_list} for u in all_nodes_list}

        for u in all_nodes_list:
            dist[u][u] = 0

        # Intra-domain links with congestion
        for lc in self.local_controllers:
            for (u, v), link in lc.domain.network.links.items():
                avail_bw = getattr(link, 'available_bw', link.bandwidth_capacity)
                if avail_bw >= bw_required:
                    congestion = (1.0 - avail_bw / link.bandwidth_capacity) ** 2
                    cost = (link.transmission_delay
                            + link.bandwidth_price * bw_required
                            + self.congestion_weight * congestion)
                    if cost < dist[u][v]:
                        dist[u][v] = cost
                        nxt[u][v] = v
                    if cost < dist[v][u]:
                        dist[v][u] = cost
                        nxt[v][u] = u

        # Inter-domain links with congestion
        for (u, v), link in self.snetwork.inter_domain_links.items():
            avail_bw = getattr(link, 'available_bw', link.bandwidth_capacity)
            if avail_bw >= bw_required:
                congestion = (1.0 - avail_bw / link.bandwidth_capacity) ** 2
                cost = (link.transmission_delay
                        + link.bandwidth_price * bw_required
                        + self.congestion_weight * congestion)
                if cost < dist[u][v]:
                    dist[u][v] = cost
                    nxt[u][v] = v
                if cost < dist[v][u]:
                    dist[v][u] = cost
                    nxt[v][u] = u

        for k in all_nodes_list:
            for i in all_nodes_list:
                for j in all_nodes_list:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        nxt[i][j] = nxt[i][k]

        return dist, nxt
