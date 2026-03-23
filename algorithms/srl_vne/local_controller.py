from typing import List, Dict, Any
from problem.domain import PhysicalDomain
from problem.substrate_network import SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNode

# ---------------- Local Controller for SRL-VNE ----------------
class LocalController:
    def __init__(self, domain: PhysicalDomain):
        self.domain = domain
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize available resources if they haven't been set yet."""
        for node in self.domain.network.nodes.values():
            if not hasattr(node, 'available_cpu'):
                node.available_cpu = node.cpu_capacity
        for link in self.domain.network.links.values():
            if not hasattr(link, 'available_bw'):
                link.available_bw = link.bandwidth_capacity

    def get_candidates(self, vnode: VirtualNode) -> List[SubstrateNode]:
        if vnode.allowed_domains and self.domain.id not in vnode.allowed_domains:
            return []
            
        candidates = []
        for node in self.domain.network.nodes.values():
            if getattr(node, 'available_cpu', node.cpu_capacity) >= vnode.cpu_demand:
                candidates.append(node)
        return candidates

    def reset_allocations(self):
        for node in self.domain.network.nodes.values():
            node.available_cpu = node.cpu_capacity
        for link in self.domain.network.links.values():
            link.available_bw = link.bandwidth_capacity
