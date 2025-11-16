from typing import List
import heapq
from src.types.substrate import SubstrateDomain, SubstrateNode, SubstrateLink

# ---------------- Local Controller ----------------
class LocalController:
    def __init__(self, domain: SubstrateDomain):
        self.domain = domain

    def get_candidates(self, vnode) -> List[SubstrateNode]:
        candidates = []
        for node in self.domain.nodes:
            if node.available_cpu >= vnode.cpu_demand:
                candidates.append(node)
        return candidates

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0) -> List[SubstrateLink]:
        dist = {node: float('inf') for node in self.domain.nodes}
        prev = {node: None for node in self.domain.nodes}
        dist[src] = 0
        pq = [(0, src)]

        while pq:
            cost_u, u = heapq.heappop(pq)
            if u == dst:
                break
            for link in self.domain.links:
                if link.src == u:
                    v = link.dst
                elif link.dst == u:
                    v = link.src
                else:
                    continue
                if link.available_bw < bw_required:
                    continue
                alt = dist[u] + link.delay + link.cost_per_unit
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = link
                    heapq.heappush(pq, (alt, v))

        path = []
        node = dst
        while node != src:
            link = prev[node]
            if link is None:
                return []
            path.append(link)
            node = link.src if link.dst == node else link.dst
        path.reverse()
        return path

    def reset_allocations(self):
        for node in self.domain.nodes:
            node.available_cpu = node.cpu_capacity
        for link in self.domain.links:
            link.available_bw = link.bandwidth

    def link_cost(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0) -> float:
        path = self.shortest_path(src, dst, bw_required)
        return sum(link.delay + link.cost_per_unit * bw_required for link in path)
