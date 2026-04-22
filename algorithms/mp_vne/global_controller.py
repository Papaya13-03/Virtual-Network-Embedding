from typing import List, Dict, Set, Tuple
from algorithms.mp_vne.local_controller import LocalController
from problem.domain import MultiDomainNetwork, PhysicalDomain
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualLink, VirtualNetwork, VirtualNode

class GlobalController:
    def __init__(self, snetwork):
        if not hasattr(snetwork, 'domains'):
            # Convert single SubstrateNetwork to MultiDomainNetwork for compatibility
            domain = PhysicalDomain(id="domain_1", network=snetwork)
            self.snetwork = MultiDomainNetwork(domains={"domain_1": domain})
            self.snetwork.inter_domain_links = {}
        else:
            self.snetwork = snetwork

        self.local_controllers: List[LocalController] = [LocalController(d) for d in self.snetwork.domains.values()]
        self._id_fw_cache = {}  # bw_required -> (dist, nxt)
        self._boundary_of: Dict[str, Set[str]] = {}          # domain_id -> boundary node ids
        self._boundary_between: Dict[Tuple[str, str], Set[str]] = {}  # (j, m) -> boundary ids of j facing m
        self._build_boundary_index()
        self._initialize_resources()

    def _build_boundary_index(self) -> None:
        for dom_id in self.snetwork.domains:
            self._boundary_of[dom_id] = set()
        for (u, v) in self.snetwork.inter_domain_links.keys():
            u_dom, _ = self._find_snode(u)
            v_dom, _ = self._find_snode(v)
            if u_dom:
                self._boundary_of[u_dom].add(u)
                self._boundary_between.setdefault((u_dom, v_dom), set()).add(u)
            if v_dom:
                self._boundary_of[v_dom].add(v)
                self._boundary_between.setdefault((v_dom, u_dom), set()).add(v)

    def _initialize_resources(self):
        """Initialize available resources if they haven't been set yet."""
        for link in self.snetwork.inter_domain_links.values():
            if not hasattr(link, 'available_bw'):
                link.available_bw = link.bandwidth_capacity

    def clear_caches(self):
        self._id_fw_cache = {}
        for lc in self.local_controllers:
            lc.clear_cache()

    # ---------------- Public interface ----------------
    def process_request(self, request: VirtualNetwork, top_k: int = 2) -> List[List[SubstrateNode]]:
        """Select candidate nodes per vnode using PreCost (MP-VNE paper, Eq. 2).

        For each virtual node, for each allowed domain, rank feasible physical nodes
        by the estimated mapping cost and keep the `top_k` lowest.
        """
        all_domains = list(self.snetwork.domains.keys())

        # Adjacency: vnode_id -> list of (VirtualLink, neighbor VirtualNode)
        adjacency: Dict[str, List[Tuple[VirtualLink, VirtualNode]]] = {vid: [] for vid in request.nodes}
        for vlink in request.links.values():
            if vlink.source in request.nodes and vlink.target in request.nodes:
                adjacency[vlink.source].append((vlink, request.nodes[vlink.target]))
                adjacency[vlink.target].append((vlink, request.nodes[vlink.source]))

        all_candidates: List[List[SubstrateNode]] = []
        for vnode in request.nodes.values():
            adj = adjacency[vnode.id]
            allowed = vnode.allowed_domains or all_domains

            vnode_candidates: List[SubstrateNode] = []
            for dom_id in allowed:
                if dom_id not in self.snetwork.domains:
                    continue
                lc = self._get_local_controller(dom_id)
                feasible = lc.get_candidates(vnode)
                ranked = sorted(
                    feasible,
                    key=lambda sn, d=dom_id: self._compute_precost(vnode, sn, d, adj, all_domains)
                )
                vnode_candidates.extend(ranked[:top_k])
            all_candidates.append(vnode_candidates)
        return all_candidates

    # ---------------- PreCost (Eq. 2 of MP-VNE paper) ----------------
    def _compute_precost(self,
                         vnode: VirtualNode,
                         snode: SubstrateNode,
                         domain_id: str,
                         adjacent_vlinks: List[Tuple[VirtualLink, VirtualNode]],
                         all_domains: List[str]) -> float:
        """Estimated mapping cost of vnode -> snode in domain_id."""
        node_term = vnode.cpu_demand * snode.cpu_price
        if not adjacent_vlinks:
            return node_term

        lc = self._get_local_controller(domain_id)
        boundary_all = self._boundary_of.get(domain_id, set())

        link_term_sum = 0.0
        num_combinations = 0

        for vlink, neighbor in adjacent_vlinks:
            neighbor_domains = neighbor.allowed_domains or all_domains
            bw = vlink.bandwidth_demand

            for m in neighbor_domains:
                if m == domain_id:
                    c_kb = 0.0
                elif (domain_id, m) in self._boundary_between:
                    targets = self._boundary_between[(domain_id, m)]
                    c_kb = self._avg_link_price(lc, snode, targets)
                else:
                    c_kb = self._avg_link_price(lc, snode, boundary_all)

                link_term_sum += bw * c_kb
                num_combinations += 1

        link_term = (link_term_sum / num_combinations) if num_combinations > 0 else 0.0
        return node_term + link_term

    def _avg_link_price(self, lc: LocalController, snode: SubstrateNode, boundary_ids: Set[str]) -> float:
        if not boundary_ids:
            return 0.0
        costs: List[float] = []
        for b_id in boundary_ids:
            if b_id == snode.id:
                costs.append(0.0)
                continue
            b_node = lc.domain.network.nodes.get(b_id)
            if b_node is None:
                continue
            path = lc.shortest_path(snode, b_node, bw_required=0.0)
            if not path:
                continue
            costs.append(sum(l.bandwidth_price for l in path))
        return (sum(costs) / len(costs)) if costs else 0.0

    def commit_mapping(self, mapping: Dict[str, str], request: VirtualNetwork) -> Dict[tuple, List[tuple]]:
        """
        Commit resources and return snapshot path of vlinks.
        Mapping maps VirtualNode ID to SubstrateNode ID.
        """
        allocated_cpu: Dict[str, float] = {}
        allocated_bw: Dict[tuple, float] = {}
        vlink_paths: Dict[tuple, List[tuple]] = {}

        try:
            # --- Allocate CPU ---
            for vnode_id, snode_id in mapping.items():
                vnode = request.nodes[vnode_id]
                snode_domain_id, snode = self._find_snode(snode_id)
                if not snode: 
                    raise ValueError(f"Node {snode_id} not found")
                    
                if getattr(snode, 'available_cpu', snode.cpu_capacity) < vnode.cpu_demand:
                    raise ValueError(f"Insufficient CPU on node {snode.id} for vnode {vnode.id}")
                snode.available_cpu -= vnode.cpu_demand
                allocated_cpu[snode.id] = allocated_cpu.get(snode.id, 0) + vnode.cpu_demand

            # --- Allocate Bandwidth (MP-VNE Splitting) ---
            for vlink_key, vlink in request.links.items():
                src_snode_id = mapping[vlink.source]
                dst_snode_id = mapping[vlink.target]
                _, src_snode = self._find_snode(src_snode_id)
                _, dst_snode = self._find_snode(dst_snode_id)
                
                demand_remaining = vlink.bandwidth_demand
                allocated_paths = []
                max_paths = 5
                
                while demand_remaining > 0.001 and len(allocated_paths) < max_paths:
                    min_required = min(demand_remaining * 0.1, 1.0, demand_remaining)
                    path = self.shortest_path(src_snode, dst_snode, bw_required=min_required, use_cache=False)
                    
                    if not path:
                        break # Cannot fulfill constraints anymore
                        
                    # Calculate how much we can push through this path
                    path_bw = min(getattr(l, 'available_bw', l.bandwidth_capacity) for l in path)
                    allocated = min(demand_remaining, path_bw)
                    
                    for link in path:
                        link.available_bw -= allocated
                        link_key = (link.source, link.target)
                        allocated_bw[link_key] = allocated_bw.get(link_key, 0) + allocated
                        
                    allocated_paths.append((path, allocated))
                    demand_remaining -= allocated
                    
                if demand_remaining > 0.001:
                    raise ValueError(f"Insufficient multi-path BW for vlink {vlink.source}->{vlink.target}")
                    
                vlink_paths[vlink_key] = allocated_paths

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

    def release_mapping(self, mapping: Dict[str, str], request: VirtualNetwork, vlink_paths: Dict[tuple, List[tuple]]) -> None:
        """Release based on snapshot path."""
        # Free CPU
        for vnode_id, snode_id in mapping.items():
            _, snode = self._find_snode(snode_id)
            vnode = request.nodes[vnode_id]
            if snode: snode.available_cpu += vnode.cpu_demand

        # Free BW
        for (v_src, v_dst), allocated_paths in vlink_paths.items():
            for path, bw in allocated_paths:
                for link in path:
                    link.available_bw += bw

    def reset_allocations(self):
        for lc in self.local_controllers:
            lc.reset_allocations()
        for link in self.snetwork.inter_domain_links.values():
            link.available_bw = link.bandwidth_capacity

    # ---------------- Internal helpers ----------------
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

    def _get_local_controller(self, domain_id: str) -> LocalController:
        for lc in self.local_controllers:
            if lc.domain.id == domain_id:
                return lc
        raise ValueError(f"No LocalController for domain {domain_id}")
        
    def _compute_interdomain_floyd_warshall(self, bw_required: float = 0.0, use_cache: bool = True):
        # Round bw_required to avoid precision issues in cache keys
        bw_key = round(bw_required, 4)
        if use_cache and bw_key in self._id_fw_cache:
            return self._id_fw_cache[bw_key]

        all_nodes = set()
        for lc in self.local_controllers:
            all_nodes.update(lc.domain.network.nodes.keys())
        all_nodes_list = list(all_nodes)

        dist = {u: {v: float('inf') for v in all_nodes_list} for u in all_nodes_list}
        nxt = {u: {v: None for v in all_nodes_list} for u in all_nodes_list}

        for u in all_nodes_list:
            dist[u][u] = 0

        # Add intra-domain links
        for lc in self.local_controllers:
            for (u, v), link in lc.domain.network.links.items():
                if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                    cost = link.transmission_delay + link.bandwidth_price * bw_required
                    if cost < dist[u][v]:
                        dist[u][v] = cost
                        nxt[u][v] = v
                    if cost < dist[v][u]:
                        dist[v][u] = cost
                        nxt[v][u] = u
                        
        # Add inter-domain links
        for (u, v), link in self.snetwork.inter_domain_links.items():
            if getattr(link, 'available_bw', link.bandwidth_capacity) >= bw_required:
                cost = link.transmission_delay + link.bandwidth_price * bw_required
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
                        
        self._id_fw_cache[bw_key] = (dist, nxt)
        return dist, nxt

    def shortest_path(self, src: SubstrateNode, dst: SubstrateNode, bw_required: float = 0.0, use_cache: bool = True) -> List[SubstrateLink]:
        src_domain, _ = self._find_snode(src.id)
        dst_domain, _ = self._find_snode(dst.id)
        
        if src_domain == dst_domain:
            lc = self._get_local_controller(src_domain)
            return lc.shortest_path(src, dst, bw_required=bw_required, use_cache=use_cache)

        # Floyd Warshall on the combined graph
        dist, nxt = self._compute_interdomain_floyd_warshall(bw_required, use_cache=use_cache)

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
            link = self._find_link(u, v)
            if link:
                path_links.append(link)
                
        return path_links