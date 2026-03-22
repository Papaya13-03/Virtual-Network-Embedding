import heapq
from typing import List, Dict, Set, Tuple
from algorithms.srl_vne.local_controller import LocalController
from problem.domain import MultiDomainNetwork, PhysicalDomain
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualLink, VirtualNetwork, VirtualNode

class GlobalController:
    def __init__(self, snetwork):
        if not hasattr(snetwork, 'domains'):
            domain = PhysicalDomain(id="domain_1", network=snetwork)
            self.snetwork = MultiDomainNetwork(domains={"domain_1": domain})
            self.snetwork.inter_domain_links = {}
        else:
            self.snetwork = snetwork
            
        self.local_controllers: List[LocalController] = [LocalController(d) for d in self.snetwork.domains.values()]
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize available resources if they haven't been set yet."""
        for link in self.snetwork.inter_domain_links.values():
            if not hasattr(link, 'available_bw'):
                link.available_bw = link.bandwidth_capacity

    def clear_caches(self):
        pass

    # ---------------- Public interface ----------------
    def process_request_for_node(self, vnode: VirtualNode) -> List[SubstrateNode]:
        """Find candidate nodes for a single vnode across all domains."""
        candidates = []
        for lc in self.local_controllers:
            candidates.extend(lc.get_candidates(vnode))
        return candidates

    def process_request(self, request: VirtualNetwork) -> List[List[SubstrateNode]]:
        """Find candidate nodes for each vnode."""
        all_candidates = []
        for vnode in request.nodes.values():
            all_candidates.append(self.process_request_for_node(vnode))
        return all_candidates

    def commit_mapping(self, mapping: Dict[str, str], request: VirtualNetwork) -> Dict[tuple, List[SubstrateLink]]:
        """
        Commit resources using SINGLE-PATH mapping for SRL-VNE (Dijkstra).
        """
        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, List[SubstrateLink]] = {}

        try:
            # --- Allocate CPU ---
            for vnode_id, snode_id in mapping.items():
                vnode = request.nodes[vnode_id]
                snode_domain_id, snode = self._find_snode(snode_id)
                if getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on node {snode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # --- Allocate Bandwidth (Single-Path via Dijkstra) ---
            for vlink_key, vlink in request.links.items():
                src_snode_id = mapping[vlink.source]
                dst_snode_id = mapping[vlink.target]
                _, src_snode = self._find_snode(src_snode_id)
                _, dst_snode = self._find_snode(dst_snode_id)
                
                path = self.dijkstra_path(src_snode, dst_snode, bw_required=vlink.bandwidth_demand)
                if not path and src_snode_id != dst_snode_id:
                    raise ValueError(f"No path found for vlink {vlink.source}->{vlink.target}")
                
                for link in path:
                    link.available_bw -= vlink.bandwidth_demand
                    link_key = (link.source, link.target)
                    allocated_bw[link_key] = allocated_bw.get(link_key, 0) + vlink.bandwidth_demand
                vlink_paths[vlink_key] = path

        except Exception as e:
            # Rollback
            for snode_id, cpu in allocated_cpu.items():
                _, snode = self._find_snode(snode_id)
                if snode: snode.available_cpu += cpu
            for link_key, bw in allocated_bw.items():
                link = self._find_link(*link_key)
                if link: link.available_bw += bw
            raise e

        return vlink_paths

    def dijkstra_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0) -> List[SubstrateLink]:
        """Shortest path using Dijkstra based on transmission delay + BW price."""
        if src.id == dst.id: return []
        adj = {}
        for lc in self.local_controllers:
            for (u, v), link in lc.domain.network.links.items():
                if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                    cost = link.transmission_delay + link.bandwidth_price * bw_required
                    adj.setdefault(u, []).append((v, cost, link))
                    adj.setdefault(v, []).append((u, cost, link))
        for (u, v), link in self.snetwork.inter_domain_links.items():
            if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                cost = link.transmission_delay + link.bandwidth_price * bw_required
                adj.setdefault(u, []).append((v, cost, link))
                adj.setdefault(v, []).append((u, cost, link))

        distances = {node_id: float('inf') for node_id in adj}; distances[src.id] = 0
        predecessors = {src.id: None}; pq = [(0, src.id)]
        while pq:
            curr_dist, u = heapq.heappop(pq)
            if curr_dist > distances.get(u, float('inf')): continue
            if u == dst.id: break
            for v, cost, link in adj.get(u, []):
                new_dist = curr_dist + cost
                if new_dist < distances.get(v, float('inf')):
                    distances[v] = new_dist
                    predecessors[v] = (u, link)
                    heapq.heappush(pq, (new_dist, v))

        if dst.id not in predecessors or predecessors[dst.id] is None: return []
        path = []; curr = dst.id
        while curr != src.id:
            prev, link = predecessors[curr]
            path.append(link); curr = prev
        return path[::-1]

    def _find_snode(self, node_id: str):
        for lc in self.local_controllers:
            if node_id in lc.domain.network.nodes:
                return lc.domain.id, lc.domain.network.nodes[node_id]
        return None, None
        
    def _find_link(self, src: str, dst: str) -> SubstrateLink:
        for lc in self.local_controllers:
            link = lc.domain.network.links.get((src, dst)) or lc.domain.network.links.get((dst, src))
            if link: return link
        link = self.snetwork.inter_domain_links.get((src, dst)) or self.snetwork.inter_domain_links.get((dst, src))
        return link
