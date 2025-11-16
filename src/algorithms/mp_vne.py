"""
Multi-Path Virtual Network Embedding (MP_VNE) Algorithm

This module implements the MP_VNE algorithm for embedding virtual networks
onto physical substrate networks using multi-path routing and optimization.
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import networkx as nx

from ..types.substrate import (
    PhysicalNode, PhysicalLink, PhysicalDomain, PhysicalNetwork
)
from ..types.virtual import VirtualNode, VirtualLink, VirtualNetwork
from ..utils.evaluator import PhysicalNetworkEvaluator
from .base import BaseVNEAlgorithm


class MP_VNE(BaseVNEAlgorithm):
    """Multi-Path Virtual Network Embedding algorithm."""
    
    def __init__(
        self,
        physical_network: PhysicalNetwork,
        evaluator: Optional[PhysicalNetworkEvaluator] = None
    ):
        """
        Initialize MP_VNE algorithm.
        
        Args:
            physical_network: The physical substrate network
            evaluator: Optional evaluator instance (will create one if not provided)
        """
        evaluator = evaluator or PhysicalNetworkEvaluator(physical_network)
        super().__init__(physical_network, evaluator)
    
    def select_candidate_nodes(
        self,
        virtual_node: VirtualNode
    ) -> List[PhysicalNode]:
        """
        Select candidate physical nodes for a virtual node.
        
        Args:
            virtual_node: The virtual node to map
            
        Returns:
            List of candidate physical nodes
        """
        candidates = []
        candidate_domains = virtual_node.candidate_domains
        
        # If no candidate domains specified, allow all domains
        if not candidate_domains or len(candidate_domains) == 0:
            candidate_domains = range(len(self.physical_network.domains))
        
        for domain_id in candidate_domains:
            if 0 <= domain_id < len(self.physical_network.domains):
                domain = self.physical_network.domains[domain_id]
                for node in domain.nodes:
                    # Check resource availability
                    available_resource = node.resource - node.used_resource
                    if available_resource >= virtual_node.resource:
                        candidates.append(node)
        
        return candidates
    
    def _id_mapping_to_node_mapping(
        self,
        id_mapping: Dict[int, PhysicalNode],
        node_id_to_node: Dict[int, VirtualNode]
    ) -> Dict[VirtualNode, PhysicalNode]:
        """Convert id-based mapping to node-based mapping."""
        # Note: VirtualNode is not hashable, so we need to create dict differently
        # Since VirtualNode objects are the same instances in node_id_to_node,
        # we can safely use them as keys by building the dict directly
        result = {}
        for node_id, p_node in id_mapping.items():
            v_node = node_id_to_node[node_id]
            result[v_node] = p_node
        return result
    
    def predict_mapping_cost(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode]
    ) -> float:
        """
        Predict the cost of mapping a virtual network with given node mapping.
        
        Args:
            virtual_network: The virtual network to map
            node_mapping: Proposed node mapping
            
        Returns:
            Predicted mapping cost
        """
        total_cost = 0.0
        
        # Node cost
        for v_node, p_node in node_mapping.items():
            node_cost = v_node.resource * p_node.cost_per_unit
            total_cost += node_cost
        
        # Estimate link cost based on shortest paths
        for i, v_node_i in enumerate(virtual_network.nodes):
            for j, v_node_j in enumerate(virtual_network.nodes):
                if i < j and virtual_network.links[i][j] is not None:
                    v_link = virtual_network.links[i][j]
                    if v_node_i in node_mapping and v_node_j in node_mapping:
                        p_node_i = node_mapping[v_node_i]
                        p_node_j = node_mapping[v_node_j]
                        distance = self.evaluator.get_distance(p_node_i, p_node_j)
                        # Estimate link cost (distance * bandwidth * avg_link_cost)
                        estimated_link_cost = distance * v_link.bandwidth * 5.0  # Average cost
                        total_cost += estimated_link_cost
        
        return total_cost
    
    def optimize_node_mapping(
        self,
        virtual_network: VirtualNetwork,
        max_iterations: int = 100,
        population_size: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
    ) -> Optional[Dict[VirtualNode, PhysicalNode]]:
        """
        Optimize node mapping using PSO-inspired approach.
        
        Args:
            virtual_network: The virtual network to map
            max_iterations: Maximum number of iterations
            population_size: Size of the population
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            
        Returns:
            Best node mapping found, or None if no valid mapping exists
        """
        # Generate candidate nodes for each virtual node (use node.id as key)
        candidates_map = {}
        node_id_to_node = {}
        for v_node in virtual_network.nodes:
            candidates = self.select_candidate_nodes(v_node)
            if not candidates:
                return None  # No valid candidates
            candidates_map[v_node.id] = candidates
            node_id_to_node[v_node.id] = v_node
        
        # Initialize population
        population = []
        for _ in range(population_size):
            mapping = {}
            for v_node in virtual_network.nodes:
                candidate = random.choice(candidates_map[v_node.id])
                mapping[v_node.id] = candidate
            population.append(mapping)
        
        # Initialize velocities (represented as probabilities)
        velocities = []
        for _ in range(population_size):
            velocity = {}
            for v_node in virtual_network.nodes:
                velocity[v_node.id] = [random.random() for _ in candidates_map[v_node.id]]
            velocities.append(velocity)
        
        # Initialize best positions and global best
        best_positions = [mapping.copy() for mapping in population]
        best_costs = [self.predict_mapping_cost(virtual_network, self._id_mapping_to_node_mapping(m, node_id_to_node)) 
                     for m in population]
        global_best_mapping = min(best_positions, key=lambda m: best_costs[best_positions.index(m)])
        global_best_cost = min(best_costs)
        
        # Iterate
        for _ in range(max_iterations):
            for i in range(population_size):
                # Update velocity for each virtual node
                for v_node in virtual_network.nodes:
                    candidates = candidates_map[v_node.id]
                    best_idx = candidates.index(best_positions[i][v_node.id])
                    global_best_idx = candidates.index(global_best_mapping[v_node.id])
                    
                    # Update velocity
                    for j in range(len(candidates)):
                        r1, r2 = random.random(), random.random()
                        velocities[i][v_node.id][j] = (
                            w * velocities[i][v_node.id][j] +
                            c1 * r1 * (1.0 if j == best_idx else 0.0) +
                            c2 * r2 * (1.0 if j == global_best_idx else 0.0)
                        )
                    
                    # Normalize velocities
                    total = sum(velocities[i][v_node.id])
                    if total > 0:
                        velocities[i][v_node.id] = [v / total for v in velocities[i][v_node.id]]
                    
                    # Select new position based on probabilities
                    rand = random.random()
                    cumulative = 0.0
                    for j, prob in enumerate(velocities[i][v_node.id]):
                        cumulative += prob
                        if rand <= cumulative:
                            population[i][v_node.id] = candidates[j]
                            break
                
                # Evaluate new position
                node_mapping = self._id_mapping_to_node_mapping(population[i], node_id_to_node)
                cost = self.predict_mapping_cost(virtual_network, node_mapping)
                if cost < best_costs[i]:
                    best_positions[i] = population[i].copy()
                    best_costs[i] = cost
                    
                    if cost < global_best_cost:
                        global_best_mapping = population[i].copy()
                        global_best_cost = cost
        
        # Convert back to node-based mapping and validate
        final_node_mapping = self._id_mapping_to_node_mapping(global_best_mapping, node_id_to_node)
        if self._validate_node_mapping(final_node_mapping):
            return final_node_mapping
        return None
    
    def _validate_node_mapping(
        self,
        node_mapping: Dict[VirtualNode, PhysicalNode]
    ) -> bool:
        """Validate that a node mapping is feasible."""
        # Check resource availability
        for v_node, p_node in node_mapping.items():
            available_resource = p_node.resource - p_node.used_resource
            if available_resource < v_node.resource:
                return False
        
        # Check uniqueness: each physical node should be mapped at most once
        physical_nodes_used = set()
        for v_node, p_node in node_mapping.items():
            if p_node in physical_nodes_used:
                return False  # Physical node already mapped
            physical_nodes_used.add(p_node)
        
        return True
    
    def construct_link_mapping(
        self,
        virtual_network: VirtualNetwork,
        node_mapping: Dict[VirtualNode, PhysicalNode],
        k: int = 3
    ) -> Dict[Tuple[VirtualNode, VirtualNode], List[PhysicalNode]]:
        """
        Construct link mapping using k-shortest paths.
        
        Args:
            virtual_network: The virtual network
            node_mapping: Node mapping from virtual to physical nodes
            k: Number of paths to consider for each virtual link
            
        Returns:
            Mapping from virtual link pairs to physical paths
        """
        link_mapping = {}
        
        for i, v_node_i in enumerate(virtual_network.nodes):
            for j, v_node_j in enumerate(virtual_network.nodes):
                if i < j and virtual_network.links[i][j] is not None:
                    v_link = virtual_network.links[i][j]
                    p_node_i = node_mapping[v_node_i]
                    p_node_j = node_mapping[v_node_j]
                    
                    # Handle case where both virtual nodes map to the same physical node
                    if p_node_i == p_node_j:
                        # If mapped to same node, path is just that node
                        link_mapping[(v_node_i, v_node_j)] = [p_node_i]
                        link_mapping[(v_node_j, v_node_i)] = [p_node_i]
                        continue
                    
                    # Find k-shortest paths
                    paths = self._find_k_shortest_paths(
                        p_node_i, p_node_j, v_link.bandwidth, k
                    )
                    
                    if paths:
                        # Select the best path based on available bandwidth
                        best_path = None
                        best_score = float('-inf')
                        
                        for path in paths:
                            # Skip paths with only one node (shouldn't happen if src != dest)
                            if len(path) < 2:
                                continue
                                
                            min_bandwidth = float('inf')
                            for idx in range(len(path) - 1):
                                link = self._find_link(path[idx], path[idx + 1])
                                if link:
                                    available = link.bandwidth - link.used_bandwidth
                                    min_bandwidth = min(min_bandwidth, available)
                                else:
                                    # If link not found, this path is invalid
                                    min_bandwidth = 0
                                    break
                            
                            if min_bandwidth >= v_link.bandwidth:
                                distance = self.evaluator.get_distance(p_node_i, p_node_j)
                                # Handle zero distance case (same node or very close nodes)
                                if distance > 0:
                                    score = min_bandwidth / distance
                                else:
                                    # If distance is 0, use bandwidth as score (prefer higher bandwidth)
                                    score = min_bandwidth
                                
                                if score > best_score:
                                    best_score = score
                                    best_path = path
                        
                        if best_path:
                            link_mapping[(v_node_i, v_node_j)] = best_path
                            # Also store reverse mapping
                            link_mapping[(v_node_j, v_node_i)] = list(reversed(best_path))
        
        return link_mapping
    
    def _find_k_shortest_paths(
        self,
        src: PhysicalNode,
        dest: PhysicalNode,
        bandwidth: float,
        k: int
    ) -> List[List[PhysicalNode]]:
        """
        Find k shortest paths between two nodes that satisfy bandwidth constraint.
        
        Args:
            src: Source physical node
            dest: Destination physical node
            bandwidth: Required bandwidth
            k: Number of paths to find
            
        Returns:
            List of paths (each path is a list of nodes)
        """
        # Handle case where src == dest (same node)
        if src == dest:
            return [[src]]
        
        # Use NetworkX for k-shortest paths
        G = nx.DiGraph()
        
        # Add nodes
        for domain in self.physical_network.domains:
            for node in domain.nodes:
                G.add_node(id(node))
        
        # Add intra-domain links
        for domain in self.physical_network.domains:
            for i, node_i in enumerate(domain.nodes):
                for j, node_j in enumerate(domain.nodes):
                    link = domain.intra_links[i][j]
                    if link is not None and (link.bandwidth - link.used_bandwidth) >= bandwidth:
                        G.add_edge(id(node_i), id(node_j), weight=link.delay)
        
        # Add inter-domain links
        for link in self.physical_network.inter_links:
            if (link.bandwidth - link.used_bandwidth) >= bandwidth:
                G.add_edge(id(link.src), id(link.dest), weight=link.delay)
        
        # Check if nodes exist in graph
        if id(src) not in G or id(dest) not in G:
            return []
        
        try:
            # Find k shortest paths
            paths = list(nx.shortest_simple_paths(
                G, id(src), id(dest), weight='weight'
            ))[:k]
            
            # Convert back to PhysicalNode lists
            result = []
            node_map = {}
            for domain in self.physical_network.domains:
                for node in domain.nodes:
                    node_map[id(node)] = node
            
            for path in paths:
                result.append([node_map[node_id] for node_id in path])
            
            return result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        except Exception as e:
            # Handle any other exceptions gracefully
            return []
    
    def embed_virtual_network(
        self,
        virtual_network: VirtualNetwork,
        max_iterations: int = 100,
        population_size: int = 50,
        **kwargs
    ) -> Optional[Dict]:
        """
        Embed a virtual network onto the physical network.
        
        Args:
            virtual_network: The virtual network to embed
            max_iterations: Maximum iterations for optimization
            population_size: Population size for optimization
            
        Returns:
            Mapping result dictionary with node_mapping, link_mapping, and cost,
            or None if embedding fails
        """
        # Optimize node mapping
        node_mapping = self.optimize_node_mapping(
            virtual_network, max_iterations, population_size
        )
        
        if node_mapping is None:
            return None
        
        # Construct link mapping
        link_mapping = self.construct_link_mapping(virtual_network, node_mapping)
        
        # Validate that all required links are mapped
        expected_link_pairs = []
        for i in range(len(virtual_network.nodes)):
            for j in range(i + 1, len(virtual_network.nodes)):
                if virtual_network.links[i][j] is not None:
                    v_node_i = virtual_network.nodes[i]
                    v_node_j = virtual_network.nodes[j]
                    expected_link_pairs.append((v_node_i, v_node_j))
        
        # Check if all expected links are in the mapping
        for link_pair in expected_link_pairs:
            if link_pair not in link_mapping:
                # Try reverse order
                reverse_pair = (link_pair[1], link_pair[0])
                if reverse_pair not in link_mapping:
                    return None  # Missing link mapping
        
        # Evaluate cost
        cost = self.evaluator.evaluate_mapping_cost(
            virtual_network, node_mapping, link_mapping
        )
        
        return {
            'node_mapping': node_mapping,
            'link_mapping': link_mapping,
            'cost': cost
        }
    

