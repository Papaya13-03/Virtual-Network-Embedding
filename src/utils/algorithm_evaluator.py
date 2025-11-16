"""
Algorithm Evaluator

This module provides functionality to evaluate and compare different VNE algorithms.
"""

from typing import List, Dict, Any, Optional
from ..types.substrate import PhysicalNetwork
from ..types.vne_request import VNERequest
from ..algorithms.base import BaseVNEAlgorithm
from .simulation import VNESimulation
from .evaluator import PhysicalNetworkEvaluator


class AlgorithmEvaluator:
    """
    Evaluator for comparing multiple VNE algorithms.
    
    This class runs multiple algorithms on the same test case and
    collects statistics for comparison.
    """
    
    def __init__(self, physical_network: PhysicalNetwork):
        """
        Initialize the algorithm evaluator.
        
        Args:
            physical_network: Physical substrate network
        """
        self.physical_network = physical_network
        self.evaluator = PhysicalNetworkEvaluator(physical_network)
    
    def evaluate_algorithm(
        self,
        algorithm: BaseVNEAlgorithm,
        requests: List[VNERequest],
        algorithm_params: Dict[str, Any] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a single algorithm on a set of requests.
        
        Args:
            algorithm: VNE algorithm instance
            requests: List of VNE requests
            algorithm_params: Algorithm-specific parameters
            verbose: Print progress information
            
        Returns:
            Dictionary with evaluation statistics
        """
        if algorithm_params is None:
            algorithm_params = {}
        
        # Create a fresh copy of the physical network for this algorithm
        # (to avoid resource conflicts between algorithm runs)
        # Note: In a real scenario, you might want to deep copy the network
        # For now, we'll reset resources before each run
        self._reset_network_resources()
        
        # Set evaluator
        algorithm.evaluator = self.evaluator
        
        # Create simulation
        simulation = VNESimulation(self.physical_network, algorithm, self.evaluator)
        
        # Run simulation
        stats = simulation.run(
            requests,
            time_step=1.0,
            max_iterations=algorithm_params.get('max_iterations', 50),
            population_size=algorithm_params.get('population_size', 30),
            verbose=verbose
        )
        
        return stats
    
    def compare_algorithms(
        self,
        algorithms: List[BaseVNEAlgorithm],
        algorithm_names: List[str],
        requests: List[VNERequest],
        algorithm_params: List[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Compare multiple algorithms on the same test case.
        
        Args:
            algorithms: List of algorithm instances
            algorithm_names: List of algorithm names
            requests: List of VNE requests
            algorithm_params: List of parameter dicts for each algorithm
            verbose: Print detailed progress
            
        Returns:
            Dictionary mapping algorithm names to their statistics
        """
        if algorithm_params is None:
            algorithm_params = [{}] * len(algorithms)
        
        if len(algorithms) != len(algorithm_names):
            raise ValueError("Number of algorithms must match number of names")
        
        if len(algorithms) != len(algorithm_params):
            raise ValueError("Number of algorithms must match number of parameter dicts")
        
        results = {}
        
        for alg, name, params in zip(algorithms, algorithm_names, algorithm_params):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating {name}...")
                print(f"{'='*60}")
            
            stats = self.evaluate_algorithm(alg, requests, params, verbose=verbose)
            results[name] = stats
        
        return results
    
    def _reset_network_resources(self):
        """Reset all network resources to initial state."""
        # Reset node resources
        for domain in self.physical_network.domains:
            for node in domain.nodes:
                node.used_resource = 0.0
        
        # Reset link resources
        for domain in self.physical_network.domains:
            # Reset intra-domain links
            for row in domain.intra_links:
                for link in row:
                    if link:
                        link.used_bandwidth = 0.0
            
            # Reset inter-domain links
            for link in domain.inter_links:
                link.used_bandwidth = 0.0
        
        # Reset global inter-domain links
        for link in self.physical_network.inter_links:
            link.used_bandwidth = 0.0
    
    @staticmethod
    def print_comparison_table(results: Dict[str, Dict]):
        """
        Print comparison results in a formatted table.
        
        Args:
            results: Dictionary mapping algorithm names to statistics
        """
        print("\n" + "="*90)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*90)
        
        # Header
        print(f"\n{'Algorithm':<20} {'Acceptance Rate':<18} {'Total Cost':<15} {'Avg Cost':<15} {'Accepted':<10} {'Rejected':<10}")
        print("-" * 90)
        
        # Results for each algorithm
        for name, stats in results.items():
            acceptance_rate = stats['acceptance_rate'] * 100
            total_cost = stats['total_cost']
            accepted = stats['accepted_requests']
            rejected = stats['rejected_requests']
            avg_cost = total_cost / accepted if accepted > 0 else 0.0
            
            print(f"{name:<20} {acceptance_rate:>15.2f}% {total_cost:>15.2f} {avg_cost:>15.2f} {accepted:>10} {rejected:>10}")
        
        print("="*90)

