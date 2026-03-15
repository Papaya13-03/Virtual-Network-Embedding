from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class SubstrateNode:
    """
    Physical Node (n_i^s)
    """
    id: str
    cpu_capacity: float      # C_i^s: CPU processing capacity
    cpu_price: float         # P_i^s: Unit price of node resource
    processing_delay: float  # D_i^s: Processing delay

@dataclass
class SubstrateLink:
    """
    Physical Link (l_ij^s)
    """
    source: str
    target: str
    bandwidth_capacity: float  # B_ij^s: Bandwidth capacity
    bandwidth_price: float     # P_ij^s: Resource unit price
    transmission_delay: float  # D_ij^s: Transmission delay

@dataclass
class SubstrateNetwork:
    """
    Substrate Network (G^s)
    """
    nodes: Dict[str, SubstrateNode] = field(default_factory=dict)
    links: Dict[Tuple[str, str], SubstrateLink] = field(default_factory=dict)
