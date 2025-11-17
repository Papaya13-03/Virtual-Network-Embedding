from typing import List, Dict
import heapq
from src.algorithms.MP_VNE.local_controller import LocalController
from src.types.substrate import SubstrateNetwork, SubstrateNode, InterLink
from src.types.virtual import VirtualLink, VirtualNetwork, VirtualNode


class GlobalController:
    def __init__(self, snetwork: SubstrateNetwork):
        self.snetwork = snetwork
        self.local_controllers: List[LocalController] = [LocalController(d) for d in snetwork.domains]

    # ---------------- Public interface ----------------
    def process_request(self, request: VirtualNetwork) -> List[List[SubstrateNode]]:
        """Tìm candidate nodes cho từng vnode của request."""
        all_candidates = []
        for vnode in request.nodes:
            candidates = []
            for lc in self.local_controllers:
                candidates.extend(lc.get_candidates(vnode))
            all_candidates.append(candidates)
        return all_candidates

    def commit_mapping(self, mapping: Dict[VirtualNode, SubstrateNode], vlinks: List[VirtualLink] = []) -> Dict:
        """
        Commit resources và trả về snapshot path của các vlink để giải phóng sau này.
        """
        allocated_cpu: Dict[SubstrateNode, float] = {}
        allocated_bw: Dict[InterLink, float] = {}
        vlink_paths: Dict[VirtualLink, List[InterLink]] = {}

        try:
            # --- Allocate CPU ---
            for vnode, snode in mapping.items():
                if snode.available_cpu < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on node {snode.node_id} for vnode {vnode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode] = allocated_cpu.get(snode, 0) + vnode.cpu_demand

            print("vlinks: ", len(vlinks))
            # --- Allocate Bandwidth ---
            for vlink in vlinks:
                src_snode = mapping[vlink.src]
                dst_snode = mapping[vlink.dst]
                path = self.shortest_path(src_snode, dst_snode, bw_required=vlink.bandwidth)

                print("path: ", path)
                if not path:
                    raise ValueError(f"No path found for virtual link {vlink.src.id}->{vlink.dst.id}")
                for link in path:
                    if link.available_bw < vlink.bandwidth:
                        raise ValueError(f"Insufficient BW on link {link.src.node_id}->{link.dst.node_id}")
                    link.available_bw -= vlink.bandwidth
                    allocated_bw[link] = allocated_bw.get(link, 0) + vlink.bandwidth
                vlink_paths[vlink] = path  # lưu snapshot path

        except Exception as e:
            # Rollback
            for snode, cpu in allocated_cpu.items():
                snode.available_cpu += cpu
            for link, bw in allocated_bw.items():
                link.available_bw += bw
            raise e

        return vlink_paths  # trả về để MP_VNE lưu

    def release_mapping(self, mapping: Dict[VirtualNode, SubstrateNode], vlink_paths: Dict[VirtualLink, List[InterLink]]) -> None:
        """
        Giải phóng dựa trên snapshot path đã commit, không tính lại shortest path.
        """
        # Free CPU
        for vnode, snode in mapping.items():
            snode.available_cpu += vnode.cpu_demand

        # Free BW
        for vlink, path in vlink_paths.items():
            for link in path:
                link.available_bw += vlink.bandwidth

    def release_resources(self):
        """Reset toàn bộ resources (dùng khi muốn xóa hết tất cả mapping)."""
        for lc in self.local_controllers:
            lc.reset_allocations()
        for link in self.snetwork.links:
            link.available_bw = link.bandwidth

    # ---------------- Internal helpers ----------------
    def _get_local_controller(self, domain_id: int) -> LocalController:
        for lc in self.local_controllers:
            if lc.domain.domain_id == domain_id:
                return lc
        raise ValueError(f"No LocalController for domain {domain_id}")

    def _interdomain_shortest_path(self, src_boundary: SubstrateNode, dst_boundary: SubstrateNode, bw_required: float = 0.0) -> List[InterLink]:
        """Dijkstra trên InterLink giữa boundary nodes khác domain."""
        graph: Dict[SubstrateNode, List[tuple]] = {}
        nodes = set()

        # Xây graph chỉ gồm inter-domain links đủ BW
        for link in self.snetwork.links:
            if link.available_bw >= bw_required:
                graph.setdefault(link.src, []).append((link.dst, link))
                graph.setdefault(link.dst, []).append((link.src, link))
                nodes.add(link.src)
                nodes.add(link.dst)

        # Thêm đường đi nội bộ boundary nodes từ LocalController
        for lc in self.local_controllers:
            b_nodes = lc.domain.boundary_nodes
            for i in range(len(b_nodes)):
                for j in range(i+1, len(b_nodes)):
                    src_b, dst_b = b_nodes[i], b_nodes[j]
                    path = lc.shortest_path(src_b, dst_b, bw_required=bw_required)
                    if not path:
                        continue
                    total_delay = sum(l.delay for l in path)
                    total_cost = sum(l.cost_per_unit * bw_required for l in path)
                    temp_link = InterLink(src=src_b, dst=dst_b, bandwidth=float('inf'), cost_per_unit=total_cost, delay=total_delay)
                    graph.setdefault(src_b, []).append((dst_b, temp_link))
                    graph.setdefault(dst_b, []).append((src_b, temp_link))
                    nodes.add(src_b)
                    nodes.add(dst_b)

        # Dijkstra
        dist = {n: float('inf') for n in nodes}
        prev_link = {n: None for n in nodes}
        dist[src_boundary] = 0
        pq = [(0, src_boundary)]

        while pq:
            cost_u, u = heapq.heappop(pq)
            if u == dst_boundary:
                break
            for v, link in graph.get(u, []):
                alt = dist[u] + link.delay + link.cost_per_unit * bw_required
                if alt < dist[v]:
                    dist[v] = alt
                    prev_link[v] = link
                    heapq.heappush(pq, (alt, v))

        # Reconstruct path
        path: List[InterLink] = []
        node = dst_boundary
        while node != src_boundary:
            link = prev_link[node]
            if link is None:
                return []  # Không có path
            path.append(link)
            node = link.src if link.dst == node else link.dst
        path.reverse()
        return path

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0) -> List[InterLink]:
        """Return shortest path kết hợp intra-domain và inter-domain."""
        src_domain = self._get_domain_id(src)
        dst_domain = self._get_domain_id(dst)
        if src_domain == dst_domain:
            lc = self._get_local_controller(src_domain)
            return lc.shortest_path(src, dst, bw_required=bw_required)

        # Ghép path: src->boundary + inter-domain + boundary->dst
        lc_src = self._get_local_controller(src_domain)
        lc_dst = self._get_local_controller(dst_domain)
        boundary_src_nodes = lc_src.domain.boundary_nodes
        boundary_dst_nodes = lc_dst.domain.boundary_nodes

        best_cost = float('inf')
        best_path: List[InterLink] = []

        for b_src in boundary_src_nodes:
            path_src = lc_src.shortest_path(src, b_src, bw_required=bw_required)
            for b_dst in boundary_dst_nodes:
                path_dst = lc_dst.shortest_path(b_dst, dst, bw_required=bw_required)
                inter_path = self._interdomain_shortest_path(b_src, b_dst, bw_required=bw_required)
                if not inter_path:
                    continue
                total_path = path_src + inter_path + path_dst
                total_cost = sum(l.delay + l.cost_per_unit * bw_required for l in total_path)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = total_path
        
        print("best_cost: ", best_cost)
        if best_cost == float('inf'):
            raise Exception("Cannot find optimal path")
        
        return best_path

    def _get_domain_id(self, node: SubstrateNode) -> int:
        for lc in self.local_controllers:
            if node in lc.domain.nodes:
                return lc.domain.domain_id
        raise ValueError(f"Node {node.node_id} not found in any domain")