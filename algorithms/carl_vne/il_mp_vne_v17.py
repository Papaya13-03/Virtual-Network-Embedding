"""V17 — V16 architecture with top-1 candidate per allowed domain.

Strictest paper-aligned candidate selection: each virtual node picks ONE
candidate per allowed domain (paper Algorithm 1 phrasing: "Each virtual node
selects only one of its candidate nodes in one of its mapped domains").

PSO then operates on these N candidates per vnode (N = num_allowed_domains,
typically 1-3 in our datasets).

Paired with MPVNEV3 (PreCost top-1 per domain) → only difference between them
is NN vs PreCost ranking criterion. Apples-to-apples test of model value.
"""
from algorithms.carl_vne.il_mp_vne_v16 import (
    ILMPVNEV16 as _V16Base,
    ILMPVNEV16Direct as _V16Direct,
    ILMPVNEV16PSO as _V16PSO,
)


class ILMPVNEV17(_V16Base):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V17"


class ILMPVNEV17Direct(_V16Direct):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V17-Direct"


class ILMPVNEV17PSO(_V16PSO):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "IL-MP-VNE-V17-PSO"
