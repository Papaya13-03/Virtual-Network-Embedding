import random
import numpy as np
from typing import List, Dict, Tuple
from algorithms.mpq_vne.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution

class MPQVNE:
    """
    MPQ-VNE (Multi-Path Q-Learning VNE) - Refactored for Multi-Domain and Dijkstra
    """
    def __init__(self):
        self.global_controller = None
        self.q_table = {}  # (domain_id, snode_id) -> Q-value
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate

    def solve(self, substrate: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(substrate)
            self._init_q_table()
        
        self.global_controller.clear_caches()
        vnetwork = virtual_request.virtual_network
        
        # 1. Node Mapping (Q-Learning based selection)
        node_mapping = self._q_node_mapping(vnetwork)
        if not node_mapping or len(node_mapping) < len(vnetwork.nodes):
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)
            
        # 2. Link Mapping (Dijkstra-based)
        try:
            vlink_paths = self.global_controller.commit_mapping(node_mapping, vnetwork)
        except Exception:
            self._update_q_table(node_mapping, vnetwork, success=False)
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)
            
        # 3. Success - Update Q-Table and Build Solution
        self._update_q_table(node_mapping, vnetwork, success=True)
        
        final_link_mapping = {}
        total_bw_cost = 0
        for vlink_key, path in vlink_paths.items():
            path_links = [(l.source, l.target) for l in path]
            vlink = vnetwork.links[vlink_key]
            final_link_mapping[vlink_key] = [(path_links, vlink.bandwidth_demand)]
            total_bw_cost += vlink.bandwidth_demand * len(path)
            
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

    def _init_q_table(self):
        for lc in self.global_controller.local_controllers:
            for s_id in lc.domain.network.nodes.keys():
                # Initial Q-value could based on initial resources
                snode = lc.domain.network.nodes[s_id]
                self.q_table[(lc.domain.id, s_id)] = snode.cpu_capacity / (getattr(snode, 'cpu_price', 1.0) + 1e-6)

    def _q_node_mapping(self, vnetwork: VirtualNetwork) -> Dict[str, str]:
        mapping = {}
        used_snodes = set()
        
        vnodes = sorted(vnetwork.nodes.values(), key=lambda n: n.cpu_demand, reverse=True)
        
        for vnode in vnodes:
            candidates = []
            for lc in self.global_controller.local_controllers:
                for snode in lc.domain.network.nodes.values():
                    if snode.id not in used_snodes and \
                       getattr(snode, 'available_cpu', snode.cpu_capacity) >= vnode.cpu_demand:
                        
                        q_val = self.q_table.get((lc.domain.id, snode.id), 0.0)
                        candidates.append((snode.id, q_val))
            
            if not candidates:
                return {}
                
            # Epsilon-greedy selection
            if random.random() < self.epsilon:
                best_snode_id = random.choice(candidates)[0]
            else:
                # Pick candidate with highest Q-value
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_snode_id = candidates[0][0]
            
            mapping[vnode.id] = best_snode_id
            used_snodes.add(best_snode_id)
            
        return mapping

    def _update_q_table(self, mapping: Dict[str, str], vnetwork: VirtualNetwork, success: bool):
        reward = 1.0 if success else -1.0
        
        for vnode_id, snode_id in mapping.items():
            domain_id, snode = self.global_controller._find_snode(snode_id)
            if not snode: continue
            
            key = (domain_id, snode_id)
            old_q = self.q_table.get(key, 0.0)
            
            # Simple reinforcement update
            # Q(s) = Q(s) + alpha * (reward - Q(s))
            new_q = old_q + self.learning_rate * (reward - old_q)
            self.q_table[key] = new_q
