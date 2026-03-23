from typing import List, Dict, Set, Tuple
import json
from .local_controller import LocalController
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
        self._id_fw_cache = {}
        self._initialize_resources()

    def _initialize_resources(self):
        for link in self.snetwork.inter_domain_links.values():
            if not hasattr(link, 'available_bw'):
                link.available_bw = link.bandwidth_capacity

    def clear_caches(self):
        self._id_fw_cache = {}
        for lc in self.local_controllers:
            lc.clear_cache()

    def find_all_candidates(self, request: VirtualNetwork) -> List[List[SubstrateNode]]:
        """Used by PSO to get candidate indices for each vnode."""
        all_candidates = []
        for vnode in request.nodes.values():
            candidates = []
            for lc in self.local_controllers:
                if not vnode.allowed_domains or lc.domain.id in vnode.allowed_domains:
                    candidates.extend(lc.get_candidates(vnode))
            all_candidates.append(candidates)
        return all_candidates

    def commit_mapping(self, mapping: Dict[str, str], request: VirtualNetwork) -> Dict[tuple, List[tuple]]:
        """
        Commit resources with Multi-Path splitting.
        """
        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, List[tuple]] = {}

        try:
            # --- CPU ---
            for vnode_id, snode_id in mapping.items():
                vnode = request.nodes[vnode_id]
                _, snode = self._find_snode(snode_id)
                if not snode or getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on node {snode_id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # --- BW (Multi-Path) ---
            for vlink_key, vlink in request.links.items():
                src_snode_id, dst_snode_id = mapping[vlink.source], mapping[vlink.target]
                _, src_snode = self._find_snode(src_snode_id)
                _, dst_snode = self._find_snode(dst_snode_id)
                
                demand_remaining = vlink.bandwidth_demand
                allocated_paths = []
                max_paths = 5
                
                while demand_remaining > 0.001 and len(allocated_paths) < max_paths:
                    min_required = min(demand_remaining * 0.1, 1.0, demand_remaining)
                    path = self.shortest_path(src_snode, dst_snode, bw_required=min_required, use_cache=False)
                    if not path: break
                        
                    path_bw = min(getattr(l, 'available_bw', l.bandwidth_capacity) for l in path)
                    allocated = min(demand_remaining, path_bw)
                    
                    for link in path:
                        link.available_bw -= allocated
                        link_key = (link.source, link.target)
                        allocated_bw[link_key] = allocated_bw.get(link_key, 0) + allocated
                    
                    allocated_paths.append((path, allocated))
                    demand_remaining -= allocated
                    
                if demand_remaining > 0.001:
                    raise ValueError(f"Insufficient MP-BW for vlink {vlink.source}->{vlink.target}")
                vlink_paths[vlink_key] = allocated_paths

        except Exception as e:
            for sid, cpu in allocated_cpu.items():
                _, snode = self._find_snode(sid); snode.available_cpu += cpu
            for lk, bw in allocated_bw.items():
                link = self._find_link(*lk); link.available_bw += bw
            raise e

        return vlink_paths

    def release_mapping(self, mapping: Dict[str, str], request: VirtualNetwork, vlink_paths: Dict[tuple, List[tuple]]) -> None:
        for vnode_id, snode_id in mapping.items():
            _, snode = self._find_snode(snode_id)
            if snode: snode.available_cpu += request.nodes[vnode_id].cpu_demand

        for (v_src, v_dst), allocated_paths in vlink_paths.items():
            for path, bw in allocated_paths:
                for link in path:
                    link.available_bw += bw

    def reset_allocations(self):
        for lc in self.local_controllers: lc.reset_allocations()
        for link in self.snetwork.inter_domain_links.values():
            link.available_bw = link.bandwidth_capacity

    def _find_snode(self, node_id: str):
        for lc in self.local_controllers:
            if node_id in lc.domain.network.nodes: return lc.domain.id, lc.domain.network.nodes[node_id]
        return None, None
        
    def _find_link(self, src: str, dst: str) -> SubstrateLink:
        for lc in self.local_controllers:
            link = lc.domain.network.links.get((src, dst)) or lc.domain.network.links.get((dst, src))
            if link: return link
        return self.snetwork.inter_domain_links.get((src, dst)) or self.snetwork.inter_domain_links.get((dst, src))

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0, use_cache: bool = True) -> List[SubstrateLink]:
        src_domain, _ = self._find_snode(src.id)
        dst_domain, _ = self._find_snode(dst.id)
        if src_domain == dst_domain:
            for lc in self.local_controllers:
                if lc.domain.id == src_domain: return lc.shortest_path(src, dst, bw_required, use_cache)

        # Inter-domain (simplified FW)
        bw_key = round(bw_required, 4)
        if use_cache and bw_key in self._id_fw_cache:
            dist, nxt = self._id_fw_cache[bw_key]
        else:
            all_nodes = [node_id for lc in self.local_controllers for node_id in lc.domain.network.nodes]
            dist = {u: {v: float('inf') for v in all_nodes} for u in all_nodes}
            nxt = {u: {v: None for v in all_nodes} for u in all_nodes}
            for u in all_nodes: dist[u][u] = 0
            # Intra-domain
            for lc in self.local_controllers:
                for (u, v), link in lc.domain.network.links.items():
                    if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                        cost = link.transmission_delay + link.bandwidth_price * bw_required
                        if cost < dist[u][v]: dist[u][v] = cost; nxt[u][v] = v
                        if cost < dist[v][u]: dist[v][u] = cost; nxt[v][u] = u
            # Inter-domain
            for (u, v), link in self.snetwork.inter_domain_links.items():
                if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                    cost = link.transmission_delay + link.bandwidth_price * bw_required
                    if cost < dist[u][v]: dist[u][v] = cost; nxt[u][v] = v
                    if cost < dist[v][u]: dist[v][u] = cost; nxt[v][u] = u
            # FW
            for k in all_nodes:
                for i in all_nodes:
                    for j in all_nodes:
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]; nxt[i][j] = nxt[i][k]
            self._id_fw_cache[bw_key] = (dist, nxt)

        if nxt[src.id][dst.id] is None: return []
        path = []; curr = src.id
        while curr != dst.id:
            nxt_node = nxt[curr][dst.id]
            path.append(self._find_link(curr, nxt_node)); curr = nxt_node
        return path

    def extract_node_features(self, snode: SubstrateNode, vnode: VirtualNode, progress: float) -> List[float]:
        s_cpu = getattr(snode, 'available_cpu', snode.cpu_capacity)
        s_degree = 0; s_bw_total = 0
        for lc in self.local_controllers:
            if snode.id in lc.domain.network.nodes:
                for (u, v), link in lc.domain.network.links.items():
                    if u == snode.id or v == snode.id:
                        s_degree += 1; s_bw_total += getattr(link, 'available_bw', link.bandwidth_capacity)
        for (u, v), link in self.snetwork.inter_domain_links.items():
            if u == snode.id or v == snode.id:
                s_degree += 1; s_bw_total += getattr(link, 'available_bw', link.bandwidth_capacity)
        return [vnode.cpu_demand / 20.0, s_cpu / 100.0, s_bw_total / 500.0, s_degree / 10.0, progress, 0.5]
