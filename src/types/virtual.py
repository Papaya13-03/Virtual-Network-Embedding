import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class VirtualNode:
    id: int
    resource: float
    candidate_domains: tuple  # Changed to tuple for hashability
    
    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.candidate_domains, list):
            object.__setattr__(self, 'candidate_domains', tuple(self.candidate_domains))

@dataclass
class VirtualLink:
    src: VirtualNode
    dest: VirtualNode
    bandwidth: float

@dataclass
class VirtualNetwork:
    nodes: List[VirtualNode]
    links: List[List[Optional[VirtualLink]]]

