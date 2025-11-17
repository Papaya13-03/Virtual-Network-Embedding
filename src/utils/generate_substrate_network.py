import random
from typing import List

from src.types.substrate import InterLink, SubstrateDomain, SubstrateLink, SubstrateNetwork, SubstrateNode

def generate_substrate_network(
    num_domains: int = 4,
    num_nodes: int = 30,                    
    node_resource_range: tuple = (100, 300),
    node_cost_range: tuple = (1, 10),
    node_delay_range: tuple = (1, 10),
    link_resource_range: tuple = (1000, 3000),
    link_cost_range: tuple = (1, 10),
    link_delay_range: tuple = (1, 10),
    inter_link_cost_range: tuple = (5, 15),
    inter_link_delay_range: tuple = (10, 30),
    num_boundary_nodes: int = 2,
    link_connection_rate: float = 50,
    seed: int = None
):
    """
    Generate a SubstrateNetwork using your classes:
    - num_nodes is TOTAL nodes across all domains (will be split evenly)
    - domains store nodes as list (SubstrateDomain.nodes: List[SubstrateNode])
    - inter-domain links use inter_link_cost_range / inter_link_delay_range
    """

    if seed is not None:
        random.seed(seed)

    network = SubstrateNetwork()

    base = num_nodes // num_domains
    extra = num_nodes % num_domains
    nodes_per_domain: List[int] = [base + (1 if i < extra else 0) for i in range(num_domains)]

    domain_boundary_nodes: List[List[SubstrateNode]] = []
    global_node_id = 0

    for d in range(num_domains):
        domain = SubstrateDomain(domain_id=d)
        count = nodes_per_domain[d]

        for _ in range(count):
            cpu = random.uniform(*node_resource_range)
            cost = random.uniform(*node_cost_range)
            delay = random.uniform(*node_delay_range)

            node = SubstrateNode(
                node_id=global_node_id,
                cpu_capacity=cpu,
                cost_per_unit=cost,
                delay=delay
            )
            global_node_id += 1
            domain.add_node(node)

        if len(domain.nodes) < num_boundary_nodes:
            boundary_nodes = list(domain.nodes)
        else:
            boundary_nodes = random.sample(domain.nodes, num_boundary_nodes)

        domain.set_boundary_nodes(boundary_nodes)
        domain_boundary_nodes.append(boundary_nodes)

        nodes = domain.nodes
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() <= link_connection_rate / 100.0:
                    bw = random.uniform(*link_resource_range)
                    link_cost = random.uniform(*link_cost_range)
                    link_delay = random.uniform(*link_delay_range)
                    link = SubstrateLink(
                        src=nodes[i],
                        dst=nodes[j],
                        bandwidth=bw,
                        cost_per_unit=link_cost,
                        delay=link_delay
                    )
                    domain.add_link(link)

        network.add_domain(domain)


    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            src_nodes = domain_boundary_nodes[i]
            dst_nodes = domain_boundary_nodes[j]
            for src in src_nodes:
                for dst in dst_nodes:
                    if random.random() <= link_connection_rate / 100.0:
                        bw = random.uniform(*link_resource_range)
                        cost = random.uniform(*inter_link_cost_range)
                        delay = random.uniform(*inter_link_delay_range)
                        inter_link = InterLink(
                            src_domain=network.domains[i],
                            dst_domain=network.domains[j],
                            src=src,
                            dst=dst,
                            bandwidth=bw,
                            cost_per_unit=cost,
                            delay=delay
                        )
                        network.add_link(inter_link)

    return network
