from typing import Dict, Type

from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.mp_vne_v4.mp_vne_v4 import MPVNEV4
from algorithms.carl_vne.il_mp_vne_v17 import (
    ILMPVNEV17,
    ILMPVNEV17Direct,
    ILMPVNEV17PSO,
)
from algorithms.carl_vne.carl_vne import (
    CARLVNE,
    CARLVNEDirect,
    CARLVNEPSO,
)

ALGORITHMS: Dict[str, Type] = {
    # Heuristic baselines
    "mp_vne": MPVNE,        # original paper heuristic (100n: 32.83%); also IL expert
    "mp_vne_v4": MPVNEV4,   # paper-faithful PSO baseline (100n: 23.30%)
    # R2 — IL-pretrain stage of CARL-VNE (imitation only, ablation baseline)
    "il_mp_vne_v17": ILMPVNEV17,
    "il_mp_vne_v17_direct": ILMPVNEV17Direct,
    "il_mp_vne_v17_pso": ILMPVNEV17PSO,
    # CARL-VNE — proposed method (V17 arch + cand_head RL-fine-tuned via
    # direct-decoding PPO). Deploy via PSO.
    "carl_vne": CARLVNE,
    "carl_vne_direct": CARLVNEDirect,
    "carl_vne_pso": CARLVNEPSO,
    # Backwards-compatible aliases (pre-rename il_mp_vne_v19 keys)
    "il_mp_vne_v19": CARLVNE,
    "il_mp_vne_v19_direct": CARLVNEDirect,
    "il_mp_vne_v19_pso": CARLVNEPSO,
}


def get_algorithm(name: str):
    algo_class = ALGORITHMS.get(name.lower())
    if not algo_class:
        raise ValueError(
            f"Algorithm '{name}' not found. Available algorithms: {list(ALGORITHMS.keys())}"
        )
    return algo_class()
