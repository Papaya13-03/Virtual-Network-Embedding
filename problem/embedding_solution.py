from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class EmbeddingSolution:
    """
    VNE Mapping Solution
    """
    vnr_id: str
    is_successful: bool
    
    # Node Mapping (M_N: N^v -> N^s)
    # Maps virtual node ID to substrate node ID
    node_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Link Mapping (M_L: L^v -> Path(L^s))
    # Maps virtual link (src, dst) to a path in substrate network
    # A path is represented as a list of substrate links (src, dst)
    link_mapping: Dict[Tuple[str, str], List[Tuple[str, str]]] = field(default_factory=dict)
    
    # Objective metrics
    embedding_cost: float = 0.0
