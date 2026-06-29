from typing import Dict, Type

from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.mc_vnm.mc_vnm import MCVNM
from algorithms.vne_pso.vne_pso import VNEPSO
from algorithms.carl_vne.r2_pretrain import (
    R2Pretrain,
    R2PretrainDirect,
    R2PretrainPSO,
)
from algorithms.carl_vne.carl_vne import (
    CARLVNE,
    CARLVNEDirect,
    CARLVNEPSO,
)

ALGORITHMS: Dict[str, Type] = {
    # Heuristic baseline — paper-faithful PSO (former mp_vne_v4; 100n: 23.30%).
    # The pre-rename "mp_vne" (32.83%) lives on internally as
    # algorithms.mp_vne.legacy.MPVNELegacy (IL expert + parent class only).
    "mp_vne": MPVNE,
    "mp_vne_v4": MPVNE,     # backwards-compatible alias
    # MC-VNM — multi-domain heuristic baseline (greedy max-CPU node + Kruskal/MST link).
    "mc_vnm": MCVNM,
    # VNE-PSO — vanilla PSO node placement (all-node search) + Dijkstra link mapping.
    "vne_pso": VNEPSO,
    # R2 — IL-pretrain stage of CARL-VNE (imitation only, ablation baseline)
    "r2": R2Pretrain,
    "r2_direct": R2PretrainDirect,
    "r2_pso": R2PretrainPSO,
    # CARL-VNE — proposed method (V17 arch + cand_head RL-fine-tuned via
    # direct-decoding PPO). Deploy via PSO.
    "carl_vne": CARLVNE,
    "carl_vne_direct": CARLVNEDirect,
    "carl_vne_pso": CARLVNEPSO,
    # Backwards-compatible aliases (pre-rename keys)
    "il_mp_vne_v17": R2Pretrain,
    "il_mp_vne_v17_direct": R2PretrainDirect,
    "il_mp_vne_v17_pso": R2PretrainPSO,
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
