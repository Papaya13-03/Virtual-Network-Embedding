"""V17 — V16 architecture with top-1 candidate per allowed domain.

Strictest paper-aligned candidate selection: each virtual node picks ONE
candidate per allowed domain (paper Algorithm 1 phrasing: "Each virtual node
selects only one of its candidate nodes in one of its mapped domains").

PSO then operates on these N candidates per vnode (N = num_allowed_domains,
typically 1-3 in our datasets).

Paired with MPVNEV3 (PreCost top-1 per domain) → only difference between them
is NN vs PreCost ranking criterion. Apples-to-apples test of model value.
"""
from algorithms.carl_vne.topk_inference import (
    TopKVNE as _V16Base,
    TopKVNEDirect as _V16Direct,
    TopKVNEPSO as _V16PSO,
)


class R2Pretrain(_V16Base):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "R2"


class R2PretrainDirect(_V16Direct):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "R2-Direct"


class R2PretrainPSO(_V16PSO):
    PER_DOMAIN_K = 1
    def __init__(self):
        super().__init__()
        self.name = "R2-PSO"
