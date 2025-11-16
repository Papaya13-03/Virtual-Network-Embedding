from typing import Callable, List, TypedDict
import numpy as np
from src.types.substrate import SubstrateNetwork
from src.types.virtual import VirtualNetwork
from src.types.dataset import Dataset
from src.types.request import VirtualRequest


def generate_dataset(
    substrate_generator: Callable[[], SubstrateNetwork],
    virtual_generator: Callable[[], VirtualNetwork],
    total_time_units: float = 100,
    avg_requests: float = 10,
    avg_lifetime: float = 1000,
    seed: int | None = None
) -> Dataset:
    """
    Generate dataset using existing substrate_generator() and virtual_generator().
    """

    rng = np.random.default_rng(seed)

    substrate_network: SubstrateNetwork = substrate_generator()

    num_requests: int = rng.poisson(lam=avg_requests)
    virtual_requests: List[VirtualRequest] = []

    for _ in range(num_requests):
        vnetwork: VirtualNetwork = virtual_generator()
        arrival_time: float = float(rng.uniform(0, total_time_units))
        lifetime: float = float(rng.exponential(scale=avg_lifetime))
        virtual_requests.append(VirtualRequest(
            vnetwork=vnetwork,
            arrival_time=arrival_time,
            lifetime=lifetime
        ))

    return Dataset(
        substrate_network=substrate_network,
        virtual_requests=virtual_requests
    )
