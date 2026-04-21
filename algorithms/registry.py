from typing import Dict, Type
from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.mc_vnm.mc_vnm import MCVNM
from algorithms.mpq_vne.mpq_vne import MPQVNE
from algorithms.srl_vne.srl_vne import SRLVNE
from algorithms.mp_dqn_vne.mp_dqn_vne import MPDQNVNE
from algorithms.srl_mp_vne.srl_mp_vne import SRLMPVNE
from algorithms.tarp_vne.tarp_vne import TARPVNE
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from algorithms.rl_oa_mp_vne.rl_oa_mp_vne import RLOAMPVNE
from algorithms.rl_cand_vne.rl_cand_vne import RLCandVNE

# Dictionary mapping algorithm names to their class implementation
ALGORITHMS: Dict[str, Type] = {
    "mp_vne": MPVNE,
    "mc_vnm": MCVNM,
    "mpq_vne": MPQVNE,
    "srl_vne": SRLVNE,
    "mp_dqn_vne": MPDQNVNE,
    "srl_mp_vne": SRLMPVNE,
    "tarp_vne": TARPVNE,
    "oa_mp_vne": OAMPVNE,
    "rl_oa_mp_vne": RLOAMPVNE,
    "rl_cand_vne": RLCandVNE,
}

def get_algorithm(name: str):
    """
    Returns an instance of the requested algorithm.
    """
    algo_class = ALGORITHMS.get(name.lower())
    if not algo_class:
        raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {list(ALGORITHMS.keys())}")
    return algo_class()
