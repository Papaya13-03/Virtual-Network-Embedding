"""MP-VNE V3 — top-1 candidate per allowed domain (strict paper interpretation).

Per paper Algorithm 1 wording: "Each virtual node selects only one of its
candidate nodes in one of its mapped domains."

V3 = mp_vne_mr (multi-restart) with top_k forced to 1.
Paired with ILMPVNEV17 (NN top-1 per domain) → ONLY difference = PreCost vs NN ranking.
"""
from algorithms.mp_vne_mr.mp_vne_mr import MPVNEMR


class MPVNEV3(MPVNEMR):
    def __init__(self):
        super().__init__()
        self.name = "MP-VNE-V3"
        # Force top_k = 1 per domain.
        self.config.setdefault("candidate_selection", {})["top_k"] = 1
