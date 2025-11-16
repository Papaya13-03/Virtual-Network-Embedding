"""
VNE Simulation with Timeline

This module provides simulation functionality to run VNE algorithms on test cases
with timeline (arrival times and durations).
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..types.substrate import PhysicalNetwork
from ..types.vne_request import VNERequest
from ..algorithms.base import BaseVNEAlgorithm
from ..utils.evaluator import PhysicalNetworkEvaluator
from ..utils.test_generator import VNETestGenerator


class VNESimulation:
    """Simulation runner for VNE algorithms with timeline."""
    
    def __init__(
        self,
        physical_network: PhysicalNetwork,
        algorithm: BaseVNEAlgorithm,
        evaluator: Optional[PhysicalNetworkEvaluator] = None
    ):
        """
        Initialize simulation.
        
        Args:
            physical_network: Physical substrate network
            algorithm: VNE algorithm instance (must be provided)
            evaluator: Optional evaluator instance (will use algorithm's evaluator if None)
        """
        self.physical_network = physical_network
        self.algorithm = algorithm
        self.evaluator = evaluator or algorithm.evaluator
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'total_cost': 0.0,
            'total_delay': 0.0,
            'request_mappings': {}  # request_id -> mapping_result
        }
    
    def run(
        self,
        requests: List[VNERequest],
        time_step: float = 1.0,
        max_iterations: int = 100,
        population_size: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Run simulation on requests with timeline.
        
        Args:
            requests: List of VNE requests with timeline
            time_step: Time step for simulation (smaller = more accurate but slower)
            max_iterations: Maximum iterations for MP_VNE optimization
            population_size: Population size for MP_VNE optimization
            verbose: Print progress information
            
        Returns:
            Dictionary with simulation statistics
        """
        # Sort requests by arrival time
        sorted_requests = sorted(requests, key=lambda r: r.arrival_time)
        self.stats['total_requests'] = len(sorted_requests)
        
        # Create event list: (time, event_type, request)
        # event_type: 'arrival' or 'departure'
        events = []
        for req in sorted_requests:
            events.append((req.arrival_time, 'arrival', req))
            events.append((req.end_time, 'departure', req))
        
        events.sort(key=lambda x: x[0])
        
        if verbose:
            print(f"\n=== Starting VNE Simulation ===")
            print(f"Total requests: {len(sorted_requests)}")
            print(f"Time step: {time_step}")
            print(f"Events: {len(events)}")
            print(f"Simulation time span: {events[0][0]:.2f} - {events[-1][0]:.2f}")
            print()
        
        # Process events
        processed_requests = set()
        
        for event_time, event_type, request in events:
            if event_type == 'arrival':
                if request.request_id in processed_requests:
                    continue
                
                if verbose:
                    print(f"[Time {event_time:.2f}] Processing request {request.request_id}...", end=" ")
                
                # Try to embed the virtual network
                # Pass algorithm-specific parameters via kwargs
                algorithm_kwargs = {}
                if hasattr(self.algorithm, '__class__') and 'MP_VNE' in str(self.algorithm.__class__):
                    algorithm_kwargs = {
                        'max_iterations': max_iterations,
                        'population_size': population_size
                    }
                
                mapping_result = self.algorithm.embed_virtual_network(
                    request.virtual_network,
                    **algorithm_kwargs
                )
                
                if mapping_result is not None:
                    # Validate mapping
                    is_valid, errors = self.evaluator.validate_mapping(
                        request.virtual_network,
                        mapping_result['node_mapping'],
                        mapping_result['link_mapping']
                    )
                    
                    if is_valid:
                        # Update resources
                        mapping_id = self.algorithm.mapping_id_counter
                        self.algorithm.update_resources(
                            request.virtual_network,
                            mapping_result,
                            mapping_id
                        )
                        self.algorithm.mapping_id_counter += 1
                        
                        # Store mapping info
                        self.stats['request_mappings'][request.request_id] = {
                            'mapping_id': mapping_id,
                            'mapping_result': mapping_result,
                            'accepted': True
                        }
                        
                        self.stats['accepted_requests'] += 1
                        self.stats['total_cost'] += mapping_result['cost']
                        
                        if verbose:
                            print(f"✓ ACCEPTED (cost: {mapping_result['cost']:.2f})")
                    else:
                        # Invalid mapping
                        self.stats['request_mappings'][request.request_id] = {
                            'accepted': False,
                            'errors': errors
                        }
                        self.stats['rejected_requests'] += 1
                        if verbose:
                            print(f"✗ REJECTED (validation failed)")
                else:
                    # Embedding failed
                    self.stats['request_mappings'][request.request_id] = {
                        'accepted': False,
                        'errors': ['Embedding failed']
                    }
                    self.stats['rejected_requests'] += 1
                    if verbose:
                        print(f"✗ REJECTED (embedding failed)")
                
                processed_requests.add(request.request_id)
                
            elif event_type == 'departure':
                # Unmap expired request
                if request.request_id in self.stats['request_mappings']:
                    mapping_info = self.stats['request_mappings'][request.request_id]
                    if mapping_info.get('accepted', False):
                        mapping_id = mapping_info['mapping_id']
                        self.algorithm.unmap_virtual_network(mapping_id)
                        
                        if verbose:
                            print(f"[Time {event_time:.2f}] Request {request.request_id} expired and unmapped")
        
        # Calculate acceptance rate
        self.stats['acceptance_rate'] = (
            self.stats['accepted_requests'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        if verbose:
            print(f"\n=== Simulation Complete ===")
            self.print_statistics()
        
        return self.stats
    
    def print_statistics(self):
        """Print simulation statistics."""
        print(f"\n=== Simulation Statistics ===")
        print(f"Total Requests: {self.stats['total_requests']}")
        print(f"Accepted: {self.stats['accepted_requests']}")
        print(f"Rejected: {self.stats['rejected_requests']}")
        print(f"Acceptance Rate: {self.stats['acceptance_rate'] * 100:.2f}%")
        print(f"Total Cost: {self.stats['total_cost']:.2f}")
        if self.stats['accepted_requests'] > 0:
            print(f"Average Cost per Request: {self.stats['total_cost'] / self.stats['accepted_requests']:.2f}")
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        return self.stats.copy()


def run_vne_simulation(
    physical_network: PhysicalNetwork,
    requests: List[VNERequest],
    algorithm: BaseVNEAlgorithm,
    max_iterations: int = 100,
    population_size: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run VNE simulation.
    
    Args:
        physical_network: Physical substrate network
        requests: List of VNE requests with timeline
        algorithm: VNE algorithm instance
        max_iterations: Maximum iterations for optimization (algorithm-specific)
        population_size: Population size for optimization (algorithm-specific)
        verbose: Print progress information
        
    Returns:
        Dictionary with simulation statistics
    """
    simulation = VNESimulation(physical_network, algorithm)
    return simulation.run(
        requests,
        max_iterations=max_iterations,
        population_size=population_size,
        verbose=verbose
    )

