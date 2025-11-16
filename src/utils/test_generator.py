"""
VNE Test Generator with Timeline

This module provides functionality to generate VNE test cases where requests
arrive over time and each has a duration.
"""

import random
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import asdict, is_dataclass

from ..types.substrate import PhysicalNetwork
from ..types.virtual import VirtualNetwork
from ..types.vne_request import VNERequest
from .dataset import DatasetGenerator, dataclass_to_dict


class VNETestGenerator:
    """Generator for VNE test cases with timeline information."""
    
    @staticmethod
    def generate_test_case(
        physical_network: Optional[PhysicalNetwork] = None,
        num_requests: int = 50,
        simulation_time: float = 100.0,
        arrival_distribution: str = "poisson",  # "poisson", "uniform", "exponential"
        arrival_rate: float = 0.5,  # requests per time unit (for poisson)
        min_duration: float = 10.0,
        max_duration: float = 50.0,
        duration_distribution: str = "uniform",  # "uniform", "normal", "exponential"
        num_request_nodes: Optional[int] = None,
        link_connection_rate: Optional[float] = None,
        num_domains: Optional[int] = None
    ) -> Tuple[PhysicalNetwork, List[VNERequest]]:
        """
        Generate a VNE test case with timeline.
        
        Args:
            physical_network: Physical network (will generate if None)
            num_requests: Number of VNE requests to generate
            simulation_time: Total simulation time span
            arrival_distribution: Distribution for request arrivals
                - "poisson": Poisson process
                - "uniform": Uniform distribution
                - "exponential": Exponential intervals
            arrival_rate: Rate parameter for arrival distribution
            min_duration: Minimum request duration
            max_duration: Maximum request duration
            duration_distribution: Distribution for request durations
                - "uniform": Uniform distribution
                - "normal": Normal distribution
                - "exponential": Exponential distribution
            num_request_nodes: Number of nodes per virtual network
            link_connection_rate: Link connection rate for virtual networks
            num_domains: Number of domains (if generating physical network)
            
        Returns:
            Tuple of (PhysicalNetwork, List[VNERequest])
        """
        # Generate physical network if not provided
        if physical_network is None:
            if num_domains is None:
                num_domains = 4
            physical_network, _ = DatasetGenerator.generate_from_config(
                number_of_physical_network_domain=num_domains,
                number_of_nodes=30,
                number_of_boundary_nodes=2,
                link_connection_rate=0.5,
                number_of_request_nodes=num_request_nodes or 6,
                number_of_requests=0  # We'll generate requests separately
            )
        
        # Get domain count from physical network
        if num_domains is None:
            num_domains = len(physical_network.domains)
        
        # Generate arrival times
        arrival_times = VNETestGenerator._generate_arrival_times(
            num_requests, simulation_time, arrival_distribution, arrival_rate
        )
        
        # Generate durations
        durations = VNETestGenerator._generate_durations(
            num_requests, min_duration, max_duration, duration_distribution
        )
        
        # Generate virtual networks
        if num_request_nodes is None:
            num_request_nodes = random.randint(3, 8)
        if link_connection_rate is None:
            link_connection_rate = random.uniform(0.3, 0.7)
        
        # Generate requests
        requests = []
        for i in range(num_requests):
            virtual_network = DatasetGenerator._create_virtual_network(
                num_nodes=num_request_nodes,
                connection_rate=link_connection_rate,
                num_domains=num_domains
            )
            
            request = VNERequest(
                request_id=i,
                virtual_network=virtual_network,
                arrival_time=arrival_times[i],
                duration=durations[i]
            )
            requests.append(request)
        
        return physical_network, requests
    
    @staticmethod
    def _generate_arrival_times(
        num_requests: int,
        simulation_time: float,
        distribution: str,
        rate: float
    ) -> List[float]:
        """Generate arrival times based on specified distribution."""
        arrival_times = []
        
        if distribution == "poisson":
            # Poisson process: exponential inter-arrival times
            current_time = 0.0
            for _ in range(num_requests):
                inter_arrival = random.expovariate(rate)
                current_time += inter_arrival
                if current_time >= simulation_time:
                    break
                arrival_times.append(current_time)
            
            # If we got fewer requests, fill remaining with uniform
            while len(arrival_times) < num_requests:
                arrival_times.append(random.uniform(0, simulation_time))
            
            arrival_times.sort()
            arrival_times = arrival_times[:num_requests]
            
        elif distribution == "uniform":
            # Uniform distribution
            arrival_times = sorted([
                random.uniform(0, simulation_time) 
                for _ in range(num_requests)
            ])
            
        elif distribution == "exponential":
            # Exponential intervals with rate
            current_time = 0.0
            for _ in range(num_requests):
                inter_arrival = random.expovariate(rate)
                current_time += inter_arrival
                arrival_times.append(min(current_time, simulation_time))
            arrival_times.sort()
            
        else:
            # Default: uniform
            arrival_times = sorted([
                random.uniform(0, simulation_time) 
                for _ in range(num_requests)
            ])
        
        return arrival_times
    
    @staticmethod
    def _generate_durations(
        num_requests: int,
        min_duration: float,
        max_duration: float,
        distribution: str
    ) -> List[float]:
        """Generate request durations based on specified distribution."""
        durations = []
        
        if distribution == "uniform" or distribution == "normal" or distribution == "exponential":
            if distribution == "uniform":
                durations = [
                    random.uniform(min_duration, max_duration)
                    for _ in range(num_requests)
                ]
            elif distribution == "normal":
                # Normal distribution centered between min and max
                mean = (min_duration + max_duration) / 2
                std = (max_duration - min_duration) / 4
                for _ in range(num_requests):
                    duration = random.gauss(mean, std)
                    duration = max(min_duration, min(max_duration, duration))
                    durations.append(duration)
            else:  # exponential
                # Exponential distribution scaled to [min, max]
                lambda_param = 1.0 / ((min_duration + max_duration) / 2)
                for _ in range(num_requests):
                    duration = random.expovariate(lambda_param)
                    duration = max(min_duration, min(max_duration, duration))
                    durations.append(duration)
        else:
            # Default: uniform
            durations = [
                random.uniform(min_duration, max_duration)
                for _ in range(num_requests)
            ]
        
        return durations
    
    @staticmethod
    def save_test_case(
        physical_network: PhysicalNetwork,
        requests: List[VNERequest],
        filename: str
    ):
        """
        Save test case to JSON file.
        
        Args:
            physical_network: Physical network
            requests: List of VNE requests
            filename: Output filename
        """
        # Convert requests to dictionaries
        requests_dict = []
        for req in requests:
            req_dict = {
                "request_id": req.request_id,
                "arrival_time": req.arrival_time,
                "duration": req.duration,
                "deadline": req.deadline,
                "virtual_network": dataclass_to_dict(req.virtual_network)
            }
            requests_dict.append(req_dict)
        
        test_case = {
            "physical_network": dataclass_to_dict(physical_network),
            "requests": requests_dict,
            "metadata": {
                "num_requests": len(requests),
                "simulation_time": max(req.end_time for req in requests) if requests else 0.0,
                "total_nodes": sum(len(d.nodes) for d in physical_network.domains),
                "num_domains": len(physical_network.domains)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(test_case, f, indent=2)
        
        print(f"Test case saved to {filename}")
    
    @staticmethod
    def load_test_case(filename: str) -> Tuple[PhysicalNetwork, List[VNERequest]]:
        """
        Load test case from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (PhysicalNetwork, List[VNERequest])
        """
        from .dataset import (
            dict_to_physical_network, dict_to_virtual_network
        )
        
        with open(filename, 'r') as f:
            test_case = json.load(f)
        
        # Load physical network
        physical_network = dict_to_physical_network(
            test_case["physical_network"]
        )
        
        # Load requests
        requests = []
        for req_dict in test_case["requests"]:
            virtual_network = dict_to_virtual_network(
                req_dict["virtual_network"]
            )
            request = VNERequest(
                request_id=req_dict["request_id"],
                virtual_network=virtual_network,
                arrival_time=req_dict["arrival_time"],
                duration=req_dict["duration"],
                deadline=req_dict.get("deadline")
            )
            requests.append(request)
        
        return physical_network, requests
    
    @staticmethod
    def get_requests_at_time(
        requests: List[VNERequest],
        current_time: float
    ) -> Tuple[List[VNERequest], List[VNERequest], List[VNERequest]]:
        """
        Get requests categorized by status at a given time.
        
        Args:
            requests: List of all VNE requests
            current_time: Current time point
            
        Returns:
            Tuple of (pending_requests, active_requests, expired_requests)
        """
        pending = []
        active = []
        expired = []
        
        for req in requests:
            if req.has_expired_at(current_time):
                expired.append(req)
            elif req.is_active_at(current_time):
                active.append(req)
            elif current_time >= req.arrival_time:
                # This shouldn't happen, but handle edge case
                active.append(req)
            else:
                pending.append(req)
        
        return pending, active, expired
    
    @staticmethod
    def print_test_case_summary(
        physical_network: PhysicalNetwork,
        requests: List[VNERequest]
    ):
        """Print summary of test case."""
        print("\n=== VNE Test Case Summary ===")
        print(f"Physical Domains: {len(physical_network.domains)}")
        total_nodes = sum(len(d.nodes) for d in physical_network.domains)
        print(f"Total Physical Nodes: {total_nodes}")
        print(f"Inter-domain Links: {len(physical_network.inter_links)}")
        print(f"\nNumber of Requests: {len(requests)}")
        
        if requests:
            arrival_times = [req.arrival_time for req in requests]
            durations = [req.duration for req in requests]
            end_times = [req.end_time for req in requests]
            
            print(f"Simulation Time Span: 0.0 - {max(end_times):.2f}")
            print(f"First Request Arrives: {min(arrival_times):.2f}")
            print(f"Last Request Arrives: {max(arrival_times):.2f}")
            print(f"Last Request Ends: {max(end_times):.2f}")
            print("\nDuration Statistics:")
            print(f"  Min: {min(durations):.2f}")
            print(f"  Max: {max(durations):.2f}")
            avg_duration = sum(durations) / len(durations)
            print(f"  Avg: {avg_duration:.2f}")
            
            # Count overlapping requests at different times
            max_concurrent = 0
            time_points = sorted(set(arrival_times + end_times))
            for t in time_points:
                _, active, _ = VNETestGenerator.get_requests_at_time(requests, t)
                max_concurrent = max(max_concurrent, len(active))
            
            print(f"Max Concurrent Requests: {max_concurrent}")


def generate_vne_test_case(
    num_requests: int = 50,
    simulation_time: float = 100.0,
    **kwargs
) -> Tuple[PhysicalNetwork, List[VNERequest]]:
    """
    Convenience function to generate a VNE test case.
    
    Args:
        num_requests: Number of VNE requests
        simulation_time: Total simulation time span
        **kwargs: Additional arguments passed to generate_test_case
        
    Returns:
        Tuple of (PhysicalNetwork, List[VNERequest])
    """
    return VNETestGenerator.generate_test_case(
        num_requests=num_requests,
        simulation_time=simulation_time,
        **kwargs
    )


def handle_generate_test_case():
    """Generate and save VNE test case with timeline."""
    from datetime import datetime
    
    # Configuration parameters
    num_requests = 50
    simulation_time = 100.0
    arrival_distribution = "poisson"  # "poisson", "uniform", "exponential"
    arrival_rate = 0.5  # requests per time unit (for poisson)
    min_duration = 10.0
    max_duration = 50.0
    duration_distribution = "uniform"  # "uniform", "normal", "exponential"
    num_request_nodes = 6
    link_connection_rate = 0.5
    num_domains = 4
    
    # Generate test case
    print("Generating VNE test case with timeline...")
    physical_network, requests = VNETestGenerator.generate_test_case(
        physical_network=None,
        num_requests=num_requests,
        simulation_time=simulation_time,
        arrival_distribution=arrival_distribution,
        arrival_rate=arrival_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        duration_distribution=duration_distribution,
        num_request_nodes=num_request_nodes,
        link_connection_rate=link_connection_rate,
        num_domains=num_domains
    )
    
    # Print summary
    VNETestGenerator.print_test_case_summary(physical_network, requests)
    
    # Save to JSON
    os.makedirs("datasets", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"./datasets/testcase-{now}.json"
    VNETestGenerator.save_test_case(physical_network, requests, filename)
    
    print(f"\nTest case saved to: {filename}")
    return filename

