from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class VirtualNode:
    """
    Virtual Node (n_i^v)
    """
    id: str
    cpu_demand: float  # C_i^v: CPU demand

@dataclass
class VirtualLink:
    """
    Virtual Link (l_ij^v)
    """
    source: str
    target: str
    bandwidth_demand: float  # B_ij^v: Bandwidth demand

@dataclass
class VirtualNetwork:
    """
    Virtual Network (G^v)
    """
    id: str
    nodes: Dict[str, VirtualNode] = field(default_factory=dict)
    links: Dict[Tuple[str, str], VirtualLink] = field(default_factory=dict)
