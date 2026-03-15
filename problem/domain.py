from dataclasses import dataclass, field
from typing import Dict, Tuple, Set

from problem.substrate_network import SubstrateNetwork, SubstrateLink

@dataclass
class PhysicalDomain:
    """
    Physical Domain (G_i^s)
    """
    id: str
    network: SubstrateNetwork
    boundary_nodes: Set[str] = field(default_factory=set)

@dataclass
class MultiDomainNetwork:
    """
    Multi-Domain Physical Network
    """
    domains: Dict[str, PhysicalDomain] = field(default_factory=dict)
    inter_domain_links: Dict[Tuple[str, str], SubstrateLink] = field(default_factory=dict)
