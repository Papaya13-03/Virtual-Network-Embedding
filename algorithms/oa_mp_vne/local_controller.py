from typing import List, Dict, Any
from problem.domain import PhysicalDomain
from problem.substrate_network import SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNode

# ---------------- Local Controller ----------------
class LocalController:
    def __init__(self, domain: PhysicalDomain):
        self.domain = domain
        self._fw_cache = {}  # bw_required -> (dist, nxt)
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize available resources if they haven't been set yet."""
        for node in self.domain.network.nodes.values():
            if not hasattr(node, 'available_cpu'):
                node.available_cpu = node.cpu_capacity
        for link in self.domain.network.links.values():
            if not hasattr(link, 'available_bw'):
                link.available_bw = link.bandwidth_capacity

    def clear_cache(self):
        self._fw_cache = {}

    def get_candidates(self, vnode: VirtualNode) -> List[SubstrateNode]:
        if vnode.allowed_domains and self.domain.id not in vnode.allowed_domains:
            return []

        candidates = []
        for node in self.domain.network.nodes.values():
            if getattr(node, 'available_cpu', node.cpu_capacity) >= vnode.cpu_demand:
                candidates.append(node)
        return candidates

    def _compute_floyd_warshall(self, bw_required: float = 0.0, use_cache: bool = True):
        # Round bw_required to avoid precision issues in cache keys
        bw_key = round(bw_required, 4)
        if use_cache and bw_key in self._fw_cache:
            return self._fw_cache[bw_key]

        nodes = list(self.domain.network.nodes.keys())
        dist = {u: {v: float('inf') for v in nodes} for u in nodes}
        nxt = {u: {v: None for v in nodes} for u in nodes}

        for u in nodes:
            dist[u][u] = 0

        for (u, v), link in self.domain.network.links.items():
            avail_bw = getattr(link, 'available_bw', link.bandwidth_capacity)
            if avail_bw >= bw_required:
                cost = link.transmission_delay + link.bandwidth_price * bw_required
                if cost < dist[u][v]:
                    dist[u][v] = cost
                    nxt[u][v] = v
                if cost < dist[v][u]:  # Undirected link support
                    dist[v][u] = cost
                    nxt[v][u] = u

        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        nxt[i][j] = nxt[i][k]

        self._fw_cache[bw_key] = (dist, nxt)
        return dist, nxt

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0, use_cache: bool = True) -> List[SubstrateLink]:
        if src.id == dst.id:
            return []

        dist, nxt = self._compute_floyd_warshall(bw_required, use_cache=use_cache)

        if nxt[src.id][dst.id] is None:
            return []

        path_nodes = [src.id]
        u = src.id
        while u != dst.id:
            u = nxt[u][dst.id]
            path_nodes.append(u)

        path_links = []
        for i in range(len(path_nodes)-1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            link = self.domain.network.links.get((u, v)) or self.domain.network.links.get((v, u))
            if link:
                path_links.append(link)

        return path_links

    def reset_allocations(self):
        for node in self.domain.network.nodes.values():
            node.available_cpu = node.cpu_capacity
        for link in self.domain.network.links.values():
            link.available_bw = link.bandwidth_capacity

    def link_cost(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0) -> float:
        path = self.shortest_path(src, dst, bw_required)
        return sum(link.transmission_delay + link.bandwidth_price * bw_required for link in path)
