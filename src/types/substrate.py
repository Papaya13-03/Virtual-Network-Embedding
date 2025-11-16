from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PhysicalNode:
    internal_id: int
    external_id: int
    domain_id: int

    resource: float
    used_resource: float
    cost_per_unit: float
    delay: float
    
    def __hash__(self):
        return hash(self.external_id)
    
    def __eq__(self, other):
        if not isinstance(other, PhysicalNode):
            return False
        return self.external_id == other.external_id


@dataclass
class PhysicalLink:
    src: PhysicalNode
    dest: PhysicalNode

    bandwidth: float
    used_bandwidth: float
    cost_per_unit: float
    delay: float


@dataclass
class PhysicalDomain:
    id: int
    nodes: List[PhysicalNode]
    boundary_nodes: List[PhysicalNode]
    intra_links: List[List[Optional[PhysicalLink]]]
    inter_links: List[PhysicalLink]

    dist: List[List[float]] = field(default_factory=list)
    next_node: List[List[Optional[int]]] = field(default_factory=list)


@dataclass
class PhysicalNetwork:
    domains: List[PhysicalDomain]
    inter_links: List[PhysicalLink]
