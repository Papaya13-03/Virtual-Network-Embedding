import random
from typing import List

from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink


def _sample_allowed_domains(
    domain_ids: List[str],
    p_all: float, p_single: float, p_subset: float,
    subset_min: int, subset_max: int,
) -> List[str]:
    r = random.random()
    if r < p_all:
        return []
    if r < p_all + p_single:
        return [random.choice(domain_ids)]
    k_max = max(1, min(subset_max, len(domain_ids)))
    k_min = max(1, min(subset_min, k_max))
    k = random.randint(k_min, k_max)
    return random.sample(domain_ids, k)


def generate_random_vn_with_domains(
    min_nodes: int,
    max_nodes: int,
    min_cpu: float,
    max_cpu: float,
    min_bw: float,
    max_bw: float,
    link_prob: float,
    domain_ids: List[str],
    p_all: float,
    p_single: float,
    p_subset: float,
    subset_min: int,
    subset_max: int,
) -> VirtualNetwork:
    """
    Random connected VN with per-vnode allowed_domains sampled from one of three modes:
    all (empty list), single domain, or random subset.
    """
    num_nodes = random.randint(min_nodes, max_nodes)
    vn = VirtualNetwork(id=f"syn_{random.randint(0, 999999)}")

    node_ids = [f"v{i}" for i in range(num_nodes)]
    for nid in node_ids:
        vn.nodes[nid] = VirtualNode(
            id=nid,
            cpu_demand=round(random.uniform(min_cpu, max_cpu), 2),
            allowed_domains=_sample_allowed_domains(
                domain_ids, p_all, p_single, p_subset, subset_min, subset_max,
            ),
        )

    shuffled = node_ids[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        parent = shuffled[random.randint(0, i - 1)]
        child = shuffled[i]
        src, dst = (parent, child) if parent < child else (child, parent)
        bw = round(random.uniform(min_bw, max_bw), 2)
        vn.links[(src, dst)] = VirtualLink(source=src, target=dst, bandwidth_demand=bw)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            key = (node_ids[i], node_ids[j])
            if key not in vn.links and random.random() < link_prob:
                bw = round(random.uniform(min_bw, max_bw), 2)
                vn.links[key] = VirtualLink(
                    source=node_ids[i], target=node_ids[j], bandwidth_demand=bw,
                )

    return vn
