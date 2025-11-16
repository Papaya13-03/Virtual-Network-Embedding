"""
Base Algorithm Interface for Virtual Network Embedding

All VNE algorithms should inherit from this base class and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from ..types.substrate import PhysicalNetwork
from ..types.virtual import VirtualNetwork
from ..types.vne_request import VNERequest


class BaseVNEAlgorithm(ABC):
    """
    Base class for all VNE algorithms.
    
    All algorithms should:
    - Take physical_network and optionally evaluator as input
    - Implement embed_virtual_network method
    - Implement update_resources and unmap_virtual_network for resource management
    """
    
    def __init__(
        self,
        physical_network: PhysicalNetwork,
        evaluator=None
    ):
        """
        Initialize the algorithm.
        
        Args:
            physical_network: The physical substrate network
            evaluator: Optional evaluator instance (for validation and cost calculation)
        """
        self.physical_network = physical_network
        self.evaluator = evaluator
        self.active_mappings: Dict[int, Dict] = {}
        self.mapping_id_counter = 0
    
    @abstractmethod
    def embed_virtual_network(
        self,
        virtual_network: VirtualNetwork,
        **kwargs
    ) -> Optional[Dict]:
        """
        Embed a virtual network onto the physical network.
        
        Args:
            virtual_network: The virtual network to embed
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Mapping result dictionary with:
                - 'node_mapping': Dict[VirtualNode, PhysicalNode]
                - 'link_mapping': Dict[Tuple[VirtualNode, VirtualNode], List[PhysicalNode]]
                - 'cost': float
            Or None if embedding fails
        """
        pass
    
    def update_resources(
        self,
        virtual_network: VirtualNetwork,
        mapping_result: Dict,
        mapping_id: int
    ):
        """
        Update physical network resources after embedding.
        
        Args:
            virtual_network: The virtual network that was embedded
            mapping_result: Result from embed_virtual_network
            mapping_id: Unique identifier for this mapping
        """
        node_mapping = mapping_result['node_mapping']
        link_mapping = mapping_result['link_mapping']
        
        # Update node resources
        for v_node, p_node in node_mapping.items():
            p_node.used_resource += v_node.resource
        
        # Update link resources
        for (v_node_i, v_node_j), path in link_mapping.items():
            v_link = None
            for i in range(len(virtual_network.nodes)):
                for j in range(len(virtual_network.nodes)):
                    link = virtual_network.links[i][j]
                    if link and link.src == v_node_i and link.dest == v_node_j:
                        v_link = link
                        break
                if v_link:
                    break
            
            if v_link:
                # Handle case where path has only one node (both virtual nodes map to same physical node)
                if len(path) == 1:
                    # No physical link resources to update for same-node mapping
                    continue
                
                for idx in range(len(path) - 1):
                    link = self._find_link(path[idx], path[idx + 1])
                    if link:
                        link.used_bandwidth += v_link.bandwidth
        
        # Store active mapping
        self.active_mappings[mapping_id] = {
            'virtual_network': virtual_network,
            'mapping_result': mapping_result
        }
    
    def unmap_virtual_network(self, mapping_id: int):
        """
        Unmap a virtual network and release resources.
        
        Args:
            mapping_id: Identifier of the mapping to remove
        """
        if mapping_id not in self.active_mappings:
            return
        
        mapping_info = self.active_mappings[mapping_id]
        virtual_network = mapping_info['virtual_network']
        mapping_result = mapping_info['mapping_result']
        
        node_mapping = mapping_result['node_mapping']
        link_mapping = mapping_result['link_mapping']
        
        # Release node resources
        for v_node, p_node in node_mapping.items():
            p_node.used_resource -= v_node.resource
        
        # Release link resources
        for (v_node_i, v_node_j), path in link_mapping.items():
            v_link = None
            for i in range(len(virtual_network.nodes)):
                for j in range(len(virtual_network.nodes)):
                    link = virtual_network.links[i][j]
                    if link and link.src == v_node_i and link.dest == v_node_j:
                        v_link = link
                        break
                if v_link:
                    break
            
            if v_link:
                # Handle case where path has only one node (both virtual nodes map to same physical node)
                if len(path) == 1:
                    # No physical link resources to release for same-node mapping
                    continue
                
                for idx in range(len(path) - 1):
                    link = self._find_link(path[idx], path[idx + 1])
                    if link:
                        link.used_bandwidth -= v_link.bandwidth
        
        # Remove from active mappings
        del self.active_mappings[mapping_id]
    
    def _find_link(self, node1, node2):
        """
        Find link between two physical nodes.
        Helper method that can be overridden by subclasses.
        """
        if self.evaluator and hasattr(self.evaluator, 'find_link'):
            return self.evaluator.find_link(node1, node2)
        
        # Fallback implementation
        if node1.domain_id == node2.domain_id:
            domain = self.physical_network.domains[node1.domain_id]
            idx1 = None
            idx2 = None
            for i, n in enumerate(domain.nodes):
                if n == node1:
                    idx1 = i
                if n == node2:
                    idx2 = i
                if idx1 is not None and idx2 is not None:
                    break
            
            if idx1 is not None and idx2 is not None:
                link = domain.intra_links[idx1][idx2]
                if link is not None:
                    return link
        
        # Check inter-domain links
        for link in self.physical_network.inter_links:
            if link.src == node1 and link.dest == node2:
                return link
        
        for domain in self.physical_network.domains:
            for link in domain.inter_links:
                if link.src == node1 and link.dest == node2:
                    return link
        
        return None

