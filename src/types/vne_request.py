"""
VNE Request with Timeline Information

This module defines data structures for Virtual Network Embedding requests
that include timing information (arrival time, duration).
"""

from dataclasses import dataclass
from typing import Optional
from .virtual import VirtualNetwork


@dataclass
class VNERequest:
    """
    Virtual Network Embedding Request with timeline information.
    
    Attributes:
        request_id: Unique identifier for the request
        virtual_network: The virtual network to be embedded
        arrival_time: Time when the request arrives (in time units)
        duration: Duration for which the request will be active (in time units)
        deadline: Optional deadline for embedding (None if no deadline)
    """
    request_id: int
    virtual_network: VirtualNetwork
    arrival_time: float
    duration: float
    deadline: Optional[float] = None
    
    @property
    def end_time(self) -> float:
        """Calculate the end time of the request."""
        return self.arrival_time + self.duration
    
    def is_active_at(self, current_time: float) -> bool:
        """
        Check if the request is active at a given time.
        
        Args:
            current_time: Current time point
            
        Returns:
            True if request is active at current_time, False otherwise
        """
        return self.arrival_time <= current_time < self.end_time
    
    def has_expired_at(self, current_time: float) -> bool:
        """
        Check if the request has expired at a given time.
        
        Args:
            current_time: Current time point
            
        Returns:
            True if request has expired at current_time, False otherwise
        """
        return current_time >= self.end_time

