import random
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink


def generate_random_vn(
    min_nodes: int = 2,
    max_nodes: int = 8,
    min_cpu: float = 1.0,
    max_cpu: float = 30.0,
    min_bw: float = 5.0,
    max_bw: float = 80.0,
    link_prob: float = 0.5,
) -> VirtualNetwork:
    """Generate a random connected virtual network."""
    num_nodes = random.randint(min_nodes, max_nodes)
    vn = VirtualNetwork(id=f"syn_{random.randint(0, 999999)}")

    node_ids = [f"v{i}" for i in range(num_nodes)]
    for nid in node_ids:
        vn.nodes[nid] = VirtualNode(
            id=nid,
            cpu_demand=round(random.uniform(min_cpu, max_cpu), 2),
        )

    # Create spanning tree first to guarantee connectivity
    shuffled = node_ids[:]
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        parent = shuffled[random.randint(0, i - 1)]
        child = shuffled[i]
        src, dst = (parent, child) if parent < child else (child, parent)
        bw = round(random.uniform(min_bw, max_bw), 2)
        vn.links[(src, dst)] = VirtualLink(source=src, target=dst, bandwidth_demand=bw)

    # Add extra random links
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            key = (node_ids[i], node_ids[j])
            if key not in vn.links and random.random() < link_prob:
                bw = round(random.uniform(min_bw, max_bw), 2)
                vn.links[key] = VirtualLink(
                    source=node_ids[i], target=node_ids[j], bandwidth_demand=bw
                )

    return vn
