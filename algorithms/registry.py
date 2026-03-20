from typing import Dict, Type
from algorithms.mp_vne.mp_vne import MPVNE
from algorithms.mc_vnm.mc_vnm import MCVNM
from algorithms.mpq_vne.mpq_vne import MPQVNE

# Dictionary mapping algorithm names to their class implementation
ALGORITHMS: Dict[str, Type] = {
    "mp_vne": MPVNE,
    "mc_vnm": MCVNM,
    "mpq_vne": MPQVNE
}

def get_algorithm(name: str):
    """
    Returns an instance of the requested algorithm.
    """
    algo_class = ALGORITHMS.get(name.lower())
    if not algo_class:
        raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {list(ALGORITHMS.keys())}")
    return algo_class()
