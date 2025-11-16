import numpy as np

def generate_dataset(substrate_generator, virtual_generator,
                     total_time_units=100, avg_requests=10, avg_lifetime=1000,
                     seed=None):
    """
    Generate dataset using existing substrate_generator() and virtual_generator().
    Uses a NumPy Generator (default_rng) for reproducible random numbers.
    
    substrate_generator() -> returns a SubstrateNetwork object
    virtual_generator()   -> returns a VirtualNetwork object
    
    Returns:
        {
            "substrate_network": SubstrateNetwork,
            "virtual_requests": List of dicts {"vnetwork": VirtualNetwork, "arrival_time": float, "lifetime": float}
        }
    """

    rng = np.random.default_rng(seed)

    substrate_network = substrate_generator()

    num_requests = rng.poisson(lam=avg_requests)
    virtual_requests = []

    for _ in range(num_requests):
        vnetwork = virtual_generator()
        arrival_time = rng.uniform(0, total_time_units)
        lifetime = rng.exponential(scale=avg_lifetime)
        virtual_requests.append({
            "vnetwork": vnetwork,
            "arrival_time": arrival_time,
            "lifetime": lifetime
        })

    dataset = {
        "substrate_network": substrate_network,
        "virtual_requests": virtual_requests
    }

    return dataset
