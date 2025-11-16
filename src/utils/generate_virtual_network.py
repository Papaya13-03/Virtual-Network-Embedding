import random
from typing import List
from src.types.virtual import VirtualLink, VirtualNetwork, VirtualNode

def generate_virtual_network_test(
    num_nodes: int = 6,
    cpu_range: tuple = (1, 10),
    bandwidth_range: tuple = (1, 10),
    num_domains: int = 4,
    link_connection_rate: float = 50
) -> VirtualNetwork:
    """
    Generate a random VirtualNetwork for testing.
    Each node can be mapped to 1..num_domains candidate domains.
    Nodes are connected randomly according to link_connection_rate (%).
    """
    # ---- Create virtual nodes ----
    vnodes: List[VirtualNode] = []
    for i in range(num_nodes):
        domains = random.sample(range(num_domains), random.randint(1, num_domains))
        node = VirtualNode(node_id=i, cpu_demand=random.uniform(*cpu_range), domains=domains)
        vnodes.append(node)

    # ---- Create virtual links ----
    vlinks: List[VirtualLink] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() <= link_connection_rate / 100.0:
                link = VirtualLink(
                    src=vnodes[i],
                    dst=vnodes[j],
                    bandwidth=random.uniform(*bandwidth_range)
                )
                vlinks.append(link)

    # ---- Create virtual network ----
    vnetwork = VirtualNetwork(nodes=vnodes, links=vlinks)
    return vnetwork


# Example usage:
# vnet = generate_virtual_network_test()
# print(f"Generated Virtual Network with {len(vnet.nodes)} nodes and {len(vnet.links)} links")
# for n in vnet.nodes:
#     print(f"Node {n.id}: CPU demand = {n.cpu_demand:.2f}, candidate domains = {n.domains}")
# for l in vnet.links:
#     print(f"Link {l.src.id} -> {l.dst.id}: bandwidth = {l.bandwidth:.2f}")
