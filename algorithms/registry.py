from typing import Dict, Type

from algorithms.mc_vnm.mc_vnm import MCVNM
from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE
from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
from algorithms.rl_oa_mp_vne_v2.rl_oa_mp_vne import (
    RLOAMPVNE as RLOAMPVNEV2,
    RLOAMPVNEV2Direct,
    RLOAMPVNEV2PSO,
)

ALGORITHMS: Dict[str, Type] = {
    "mc_vnm": MCVNM,
    "mp_vne": MPVNE,
    "oa_mp_vne": OAMPVNE,
    "rl_cand_vne": RLCandVNE,
    "rl_oa_mp_vne": RLOAMPVNE,
    "rl_oa_mp_vne_v2": RLOAMPVNEV2,                # default mode from config
    "rl_oa_mp_vne_v2_direct": RLOAMPVNEV2Direct,   # autoregressive, no PSO
    "rl_oa_mp_vne_v2_pso": RLOAMPVNEV2PSO,         # PSO + top-K (A/B reference)
}


def get_algorithm(name: str):
    algo_class = ALGORITHMS.get(name.lower())
    if not algo_class:
        raise ValueError(
            f"Algorithm '{name}' not found. Available algorithms: {list(ALGORITHMS.keys())}"
        )
    return algo_class()
