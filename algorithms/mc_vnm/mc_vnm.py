from typing import List, Dict, Tuple
from algorithms.mc_vnm.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution

class MCVNM:
    """
    MC-VNM (Minimum Cost Virtual Network Mapping) - Refactored for Multi-Domain
    """
    def __init__(self):
        self.global_controller: GlobalController = None

    def solve(self, substrate: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(substrate)
        
        # 1. Start fresh for each request
        self.global_controller.clear_caches()
        vnetwork = virtual_request.virtual_network
        
        # 2. Node Mapping (Greedy)
        node_mapping = self._greedy_node_mapping(vnetwork)
        if not node_mapping or len(node_mapping) < len(vnetwork.nodes):
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)
            
        # 3. Link Mapping & Resource Commitment (Single-Path)
        try:
            vlink_paths = self.global_controller.commit_mapping(node_mapping, vnetwork)
        except Exception as e:
            # print(f"Link mapping failed: {e}")
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)
            
        # 4. Success - Build Solution
        final_link_mapping = {}
        total_bw_cost = 0
        for vlink_key, path in vlink_paths.items():
            vlink = vnetwork.links[vlink_key]
            # Match MP-VNE format: [(path_links, allocated_bw)]
            path_links = [(l.source, l.target) for l in path]
            final_link_mapping[vlink_key] = [(path_links, vlink.bandwidth_demand)]
            # BW cost = demand * length
            total_bw_cost += vlink.bandwidth_demand * len(path)
            
        # Node cost = sum(cpu * price)
        total_node_cost = 0
        for vnode_id, snode_id in node_mapping.items():
            _, snode = self.global_controller._find_snode(snode_id)
            vnode = vnetwork.nodes[vnode_id]
            total_node_cost += vnode.cpu_demand * getattr(snode, 'cpu_price', 1.0)
            
        return EmbeddingSolution(
            vnr_id=virtual_request.id,
            is_successful=True,
            node_mapping=node_mapping,
            link_mapping=final_link_mapping,
            embedding_cost=total_node_cost + total_bw_cost
        )

    def _greedy_node_mapping(self, vnetwork: VirtualNetwork) -> Dict[str, str]:
        """
        Greedily map vnodes to best candidate snodes.
        """
        mapping = {}
        used_snodes = set()
        
        # Sort vnodes by CPU demand (descending) to map the hardest ones first
        vnodes = sorted(vnetwork.nodes.values(), key=lambda n: n.cpu_demand, reverse=True)
        
        for vnode in vnodes:
            candidates = []
            for lc in self.global_controller.local_controllers:
                # Get candidates that have enough CPU
                for snode in lc.domain.network.nodes.values():
                    if snode.id not in used_snodes and \
                       getattr(snode, 'available_cpu', snode.cpu_capacity) >= vnode.cpu_demand:
                        
                        # Rank candidates based on price and available CPU
                        # Score: price / (available_cpu + 1e-6) -> lower is better
                        score = getattr(snode, 'cpu_price', 1.0) / (getattr(snode, 'available_cpu', snode.cpu_capacity) + 1e-6)
                        candidates.append((snode.id, score))
            
            if not candidates:
                return {} # Failed to map all nodes
                
            # Pick candidate with lowest score
            candidates.sort(key=lambda x: x[1])
            best_snode_id = candidates[0][0]
            
            mapping[vnode.id] = best_snode_id
            used_snodes.add(best_snode_id)
            
        return mapping