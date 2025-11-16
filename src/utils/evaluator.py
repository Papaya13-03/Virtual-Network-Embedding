"""
Physical Network Evaluator

This module provides evaluation and validation functions for Virtual Network
Embedding mappings, including cost calculation and correctness checking.
"""

from typing import Dict, List, Optional, Tuple
from ..types.substrate import (
    PhysicalNode, PhysicalLink, PhysicalDomain, PhysicalNetwork
)
from ..types.virtual import VirtualNode, VirtualLink, VirtualNetwork


class PhysicalNetworkEvaluator:
    """Evaluator for physical network with shortest path computation and cost evaluation."""
    
    def __init__(self, physical_network: PhysicalNetwork):
        """
        Initialize the evaluator with a physical network.
        
        Args:
            physical_network: The physical substrate network
        """
        self.physical_network = physical_network
        self.global_graph = None
        self.node_to_index = {}
        self.index_to_node = {}
        self.dist_matrix = None
        self.next_matrix = None
        self._build_global_graph()
        self._compute_shortest_paths()
    
    def _build_global_graph(self):
        """Build a global graph representation of the physical network."""
        # Create mapping from nodes to indices
        all_nodes = []
        for domain in self.physical_network.domains:
            for node in domain.nodes:
                all_nodes.append(node)
        
        # Build node mapping (use external_id as key since nodes are not hashable)
        for idx, node in enumerate(all_nodes):
            self.node_to_index[node.external_id] = idx
            self.index_to_node[idx] = node
        
        # Initialize graph
        n = len(all_nodes)
        self.global_graph = [[float('inf')] * n for _ in range(n)]
        
        # Set diagonal to 0
        for i in range(n):
            self.global_graph[i][i] = 0
        
        # Add intra-domain links
        for domain in self.physical_network.domains:
            for i, node_i in enumerate(domain.nodes):
                for j, node_j in enumerate(domain.nodes):
                    link = domain.intra_links[i][j]
                    if link is not None:
                        idx_i = self.node_to_index[node_i.external_id]
                        idx_j = self.node_to_index[node_j.external_id]
                        weight = link.delay  # Use delay as weight
                        self.global_graph[idx_i][idx_j] = weight
        
        # Add inter-domain links
        for link in self.physical_network.inter_links:
            idx_src = self.node_to_index[link.src.external_id]
            idx_dest = self.node_to_index[link.dest.external_id]
            weight = link.delay
            self.global_graph[idx_src][idx_dest] = weight
    
    def _compute_shortest_paths(self):
        """Compute shortest paths using Floyd-Warshall algorithm."""
        n = len(self.global_graph)
        
        # Initialize distance and next matrices
        self.dist_matrix = [row[:] for row in self.global_graph]
        self.next_matrix = [[None] * n for _ in range(n)]
        
        # Initialize next matrix
        for i in range(n):
            for j in range(n):
                if i != j and self.dist_matrix[i][j] != float('inf'):
                    self.next_matrix[i][j] = j
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.dist_matrix[i][k] != float('inf') and \
                       self.dist_matrix[k][j] != float('inf'):
                        new_dist = self.dist_matrix[i][k] + self.dist_matrix[k][j]
                        if new_dist < self.dist_matrix[i][j]:
                            self.dist_matrix[i][j] = new_dist
                            self.next_matrix[i][j] = self.next_matrix[i][k]
    
    def get_distance(self, node1: PhysicalNode, node2: PhysicalNode) -> float:
        """
        Get shortest distance between two nodes.
        
        Args:
            node1: Source physical node
            node2: Destination physical node
            
        Returns:
            Shortest distance (delay) between nodes
        """
        idx1 = self.node_to_index.get(node1.external_id)
        idx2 = self.node_to_index.get(node2.external_id)
        
        if idx1 is None or idx2 is None:
            return float('inf')
        
        return self.dist_matrix[idx1][idx2]
    
    def get_path(self, node1: PhysicalNode, node2: PhysicalNode) -> List[PhysicalNode]:
        """
        Get shortest path between two nodes.
        
        Args:
            node1: Source physical node
            node2: Destination physical node
            
        Returns:
            List of nodes forming the shortest path
        """
        idx1 = self.node_to_index.get(node1.external_id)
        idx2 = self.node_to_index.get(node2.external_id)
        
        if idx1 is None or idx2 is None:
            return []
        
        if self.next_matrix[idx1][idx2] is None:
            return []
        
        path = [node1]
        current = idx1
        
        while current != idx2:
            current = self.next_matrix[current][idx2]
            if current is None:
                return []
            path.append(self.index_to_node[current])
        
        return path
    
    def find_link(self, node1: PhysicalNode, node2: PhysicalNode) -> Optional[PhysicalLink]:
        """
        Find link between two physical nodes.
        
        Args:
            node1: Source physical node
            node2: Destination physical node
            
        Returns:
            PhysicalLink if exists, None otherwise
        """
        # Check if nodes are in the same domain first
        if node1.domain_id == node2.domain_id:
            domain = self.physical_network.domains[node1.domain_id]
            # Find indices in domain
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
        
        # Also check domain's inter_links (for reverse direction)
        for domain in self.physical_network.domains:
            for link in domain.inter_links:
                if link.src == node1 and link.dest == node2:
                    return link
        
        return None
    
    def evaluate_mapping_cost(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode],
        link_mapping: Dict[Tuple[VirtualNode, VirtualNode], List[PhysicalNode]]
    ) -> float:
        """
        Evaluate the total cost of a virtual network mapping.
        
        Args:
            virtual_network: The virtual network being mapped
            node_mapping: Mapping from virtual nodes to physical nodes
            link_mapping: Mapping from virtual link pairs (v_node_i, v_node_j) to physical paths
            
        Returns:
            Total mapping cost
        """
        total_cost = 0.0
        
        # Node cost
        for v_node, p_node in node_mapping.items():
            node_cost = v_node.resource * p_node.cost_per_unit
            total_cost += node_cost
        
        # Link cost - need to get VirtualLink from virtual_network
        for (v_node_i, v_node_j), path in link_mapping.items():
            # Find the corresponding VirtualLink in virtual_network
            v_link = None
            for i in range(len(virtual_network.nodes)):
                for j in range(len(virtual_network.nodes)):
                    if virtual_network.links[i][j] is not None:
                        link = virtual_network.links[i][j]
                        if link.src == v_node_i and link.dest == v_node_j:
                            v_link = link
                            break
                    if v_link:
                        break
                if v_link:
                    break
            
            if v_link is None:
                # If link not found, skip this mapping (shouldn't happen in valid mappings)
                continue
            
            bandwidth = v_link.bandwidth
            
            # Handle case where path has only one node (both virtual nodes map to same physical node)
            if len(path) == 1:
                # No physical link cost for same-node mapping
                continue
            
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                # Find the link between node1 and node2
                link = self.find_link(node1, node2)
                if link:
                    link_cost = bandwidth * link.cost_per_unit
                    total_cost += link_cost
        
        return total_cost
    
    def validate_node_mapping(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a node mapping is correct and feasible.
        
        Args:
            virtual_network: The virtual network
            node_mapping: Mapping from virtual nodes to physical nodes
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check all virtual nodes are mapped
        if len(node_mapping) != len(virtual_network.nodes):
            errors.append(
                f"Not all virtual nodes are mapped: {len(node_mapping)}/{len(virtual_network.nodes)}"
            )
        
        # Check each virtual node
        for v_node in virtual_network.nodes:
            if v_node not in node_mapping:
                errors.append(f"Virtual node {v_node.id} is not mapped")
                continue
            
            p_node = node_mapping[v_node]
            
            # Check resource availability
            available_resource = p_node.resource - p_node.used_resource
            if available_resource < v_node.resource:
                errors.append(
                    f"Virtual node {v_node.id} requires {v_node.resource} resources, "
                    f"but physical node {p_node.external_id} only has {available_resource} available"
                )
            
            # Check candidate domain constraint
            if v_node.candidate_domains and p_node.domain_id not in v_node.candidate_domains:
                errors.append(
                    f"Virtual node {v_node.id} must be mapped to domains {v_node.candidate_domains}, "
                    f"but mapped to domain {p_node.domain_id}"
                )
        
        # Check uniqueness (each physical node mapped at most once)
        physical_node_usage = {}
        for v_node, p_node in node_mapping.items():
            if p_node in physical_node_usage:
                errors.append(
                    f"Physical node {p_node.external_id} is mapped to multiple virtual nodes: "
                    f"{physical_node_usage[p_node].id} and {v_node.id}"
                )
            else:
                physical_node_usage[p_node] = v_node
        
        return len(errors) == 0, errors
    
    def validate_link_mapping(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode],
        link_mapping: Dict[Tuple[VirtualNode, VirtualNode], List[PhysicalNode]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a link mapping is correct and feasible.
        
        Args:
            virtual_network: The virtual network
            node_mapping: Mapping from virtual nodes to physical nodes
            link_mapping: Mapping from virtual link pairs to physical paths
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check all virtual links are mapped
        expected_links = 0
        link_pairs = []
        for i in range(len(virtual_network.nodes)):
            for j in range(len(virtual_network.nodes)):
                if virtual_network.links[i][j] is not None:
                    expected_links += 1
                    v_link = virtual_network.links[i][j]
                    link_pairs.append((v_link.src, v_link.dest))
        
        if len(link_mapping) < expected_links:
            errors.append(
                f"Not all virtual links are mapped: {len(link_mapping)}/{expected_links}"
            )
        
        # Check each link mapping
        for (v_node_i, v_node_j), path in link_mapping.items():
            # Verify nodes are mapped
            if v_node_i not in node_mapping:
                errors.append(f"Source virtual node {v_node_i.id} for link mapping is not in node_mapping")
                continue
            if v_node_j not in node_mapping:
                errors.append(f"Dest virtual node {v_node_j.id} for link mapping is not in node_mapping")
                continue
            
            p_node_i = node_mapping[v_node_i]
            p_node_j = node_mapping[v_node_j]
            
            # Handle case where both nodes map to the same physical node
            if p_node_i == p_node_j:
                # Path should be [p_node_i] or [p_node_i, p_node_j] (both valid)
                if len(path) == 1:
                    if path[0] != p_node_i:
                        errors.append(
                            f"Path for link ({v_node_i.id}, {v_node_j.id}) mapped to same node "
                            f"but path node {path[0].external_id} != {p_node_i.external_id}"
                        )
                elif len(path) >= 2:
                    if path[0] != p_node_i or path[-1] != p_node_j:
                        errors.append(
                            f"Path for link ({v_node_i.id}, {v_node_j.id}) does not start/end correctly"
                        )
                # If path is valid for same-node case, skip bandwidth check
                continue
            
            # Verify path starts and ends at correct nodes (for different nodes)
            if len(path) < 2:
                errors.append(
                    f"Path for link ({v_node_i.id}, {v_node_j.id}) is too short: {len(path)} nodes"
                )
                continue
            
            if path[0] != p_node_i:
                errors.append(
                    f"Path for link ({v_node_i.id}, {v_node_j.id}) does not start at "
                    f"mapped physical node {p_node_i.external_id}"
                )
            
            if path[-1] != p_node_j:
                errors.append(
                    f"Path for link ({v_node_i.id}, {v_node_j.id}) does not end at "
                    f"mapped physical node {p_node_j.external_id}"
                )
            
            # Find corresponding VirtualLink
            v_link = None
            for i in range(len(virtual_network.nodes)):
                for j in range(len(virtual_network.nodes)):
                    if virtual_network.links[i][j] is not None:
                        link = virtual_network.links[i][j]
                        if link.src == v_node_i and link.dest == v_node_j:
                            v_link = link
                            break
                    if v_link:
                        break
                if v_link:
                    break
            
            if v_link is None:
                errors.append(
                    f"No VirtualLink found for mapping ({v_node_i.id}, {v_node_j.id})"
                )
                continue
            
            # Check bandwidth availability on each physical link in path
            for idx in range(len(path) - 1):
                node1 = path[idx]
                node2 = path[idx + 1]
                link = self.find_link(node1, node2)
                
                if link is None:
                    errors.append(
                        f"No physical link exists between {node1.external_id} and {node2.external_id} "
                        f"in path for virtual link ({v_node_i.id}, {v_node_j.id})"
                    )
                    continue
                
                available_bandwidth = link.bandwidth - link.used_bandwidth
                if available_bandwidth < v_link.bandwidth:
                    errors.append(
                        f"Insufficient bandwidth on link ({node1.external_id}, {node2.external_id}): "
                        f"required {v_link.bandwidth}, available {available_bandwidth}"
                    )
        
        return len(errors) == 0, errors
    
    def validate_mapping(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode],
        link_mapping: Dict[Tuple[VirtualNode, VirtualNode], List[PhysicalNode]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a complete virtual network mapping (nodes + links).
        
        Args:
            virtual_network: The virtual network
            node_mapping: Mapping from virtual nodes to physical nodes
            link_mapping: Mapping from virtual link pairs to physical paths
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        is_valid = True
        
        # Validate node mapping
        node_valid, node_errors = self.validate_node_mapping(virtual_network, node_mapping)
        if not node_valid:
            is_valid = False
            all_errors.extend(node_errors)
        
        # Validate link mapping
        link_valid, link_errors = self.validate_link_mapping(
            virtual_network, node_mapping, link_mapping
        )
        if not link_valid:
            is_valid = False
            all_errors.extend(link_errors)
        
        return is_valid, all_errors

