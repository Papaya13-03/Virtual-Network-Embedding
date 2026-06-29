from dataclasses import dataclass
from problem.virtual_network import VirtualNetwork

@dataclass
class VirtualNetworkRequest:
    """
    Virtual Network Request (VNR)
    """
    id: str
    virtual_network: VirtualNetwork
    arrival_time: float  # Time when request arrives
    lifetime: float      # Duration the request remains active
