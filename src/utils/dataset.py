from datetime import datetime
import random
from typing import Tuple

from ..types.substrate import (
    PhysicalNode, PhysicalLink, PhysicalDomain, PhysicalNetwork
)
from ..types.virtual import VirtualNode, VirtualLink, VirtualNetwork

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ..types.substrate import PhysicalNode, PhysicalLink, PhysicalDomain, PhysicalNetwork
from ..types.virtual import VirtualNode, VirtualLink, VirtualNetwork



class DatasetGenerator:
    """Generate synthetic datasets for Virtual Network Embedding"""

    @staticmethod
    def generate_from_config(
        number_of_physical_network_domain: int,
        number_of_nodes: int,
        number_of_boundary_nodes: int,
        link_connection_rate: float,
        number_of_request_nodes: int,
        number_of_requests: int = 10
    ) -> Tuple[PhysicalNetwork, list]:
        """
        Generate dataset from configuration parameters.
        
        Args:
            number_of_physical_network_domain: Number of physical domains
            number_of_nodes: Total number of substrate nodes
            number_of_boundary_nodes: Boundary nodes per domain
            link_connection_rate: Connection probability for links (0.0 to 1.0)
            number_of_request_nodes: Nodes per virtual network request
            number_of_requests: Number of virtual network requests to generate
            
        Returns:
            Tuple of (PhysicalNetwork, list of VirtualNetworks)
        """
        # Distribute nodes randomly across domains
        node_distribution = DatasetGenerator._distribute_nodes_randomly(
            number_of_nodes,
            number_of_physical_network_domain
        )
        
        # Generate physical network
        domains = []
        global_node_id = 0

        for domain_id in range(number_of_physical_network_domain):
            domain = DatasetGenerator._create_physical_domain(
                num_nodes=node_distribution[domain_id],
                boundary_nodes_count=number_of_boundary_nodes,
                domain_id=domain_id,
                connection_rate=link_connection_rate,
                start_global_id=global_node_id
            )
            global_node_id += len(domain.nodes)
            domains.append(domain)

        # Generate inter-domain links
        inter_links = DatasetGenerator._create_inter_domain_links(
            domains,
            number_of_physical_network_domain
        )

        physical_network = PhysicalNetwork(
            domains=domains,
            inter_links=inter_links
        )

        # Generate virtual network requests
        virtual_networks = [
            DatasetGenerator._create_virtual_network(
                number_of_request_nodes,
                link_connection_rate,
                number_of_physical_network_domain
            )
            for _ in range(number_of_requests)
        ]

        return physical_network, virtual_networks

    @staticmethod
    def _distribute_nodes_randomly(total_nodes: int, num_domains: int) -> list:
        """Distribute nodes randomly across domains."""
        if num_domains <= 0:
            raise ValueError("num_domains must be positive")
        if total_nodes < num_domains:
            raise ValueError("total_nodes must be at least equal to num_domains")
        
        distribution = [1] * num_domains
        remaining_nodes = total_nodes - num_domains
        
        for _ in range(remaining_nodes):
            domain_idx = random.randint(0, num_domains - 1)
            distribution[domain_idx] += 1
        
        return distribution

    @staticmethod
    def _create_physical_domain(
        num_nodes: int,
        boundary_nodes_count: int,
        domain_id: int,
        connection_rate: float,
        start_global_id: int
    ) -> PhysicalDomain:
        """Create a physical domain with nodes and intra-domain links."""
        # Create nodes
        nodes = []
        for i in range(num_nodes):
            node = PhysicalNode(
                external_id=start_global_id + i,
                internal_id=i,
                domain_id=domain_id,
                resource=random.uniform(100, 300),
                used_resource=0,
                cost_per_unit=random.uniform(1, 10),
                delay=random.uniform(1, 10)
            )
            nodes.append(node)

        # Create intra-domain links
        intra_links = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() <= connection_rate:
                    link = PhysicalLink(
                        src=nodes[i],
                        dest=nodes[j],
                        bandwidth=random.uniform(1000, 3000),
                        used_bandwidth=0,
                        cost_per_unit=random.uniform(1, 10),
                        delay=random.uniform(1, 10)
                    )
                    intra_links[i][j] = intra_links[j][i] = link

        # Select boundary nodes
        boundary_nodes = random.sample(
            nodes,
            k=min(boundary_nodes_count, num_nodes)
        )

        return PhysicalDomain(
            id=domain_id,
            nodes=nodes,
            boundary_nodes=boundary_nodes,
            intra_links=intra_links,
            inter_links=[]
        )

    @staticmethod
    def _create_inter_domain_links(domains: list, num_domains: int) -> list:
        """Create links between domains."""
        inter_links = []
        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                src = random.choice(domains[i].boundary_nodes)
                dest = random.choice(domains[j].boundary_nodes)

                # Link from domain i to domain j
                link_ij = PhysicalLink(
                    src=src,
                    dest=dest,
                    bandwidth=random.uniform(1000, 3000),
                    used_bandwidth=0,
                    cost_per_unit=random.uniform(1, 10),
                    delay=random.uniform(1, 10)
                )
                inter_links.append(link_ij)
                domains[i].inter_links.append(link_ij)

                # Reverse link from domain j to domain i
                link_ji = PhysicalLink(
                    src=link_ij.dest,
                    dest=link_ij.src,
                    bandwidth=link_ij.bandwidth,
                    used_bandwidth=0,
                    cost_per_unit=link_ij.cost_per_unit,
                    delay=link_ij.delay
                )
                domains[j].inter_links.append(link_ji)

        return inter_links

    @staticmethod
    def _create_virtual_network(
        num_nodes: int,
        connection_rate: float,
        num_domains: int
    ) -> VirtualNetwork:
        """Create a virtual network."""
        # Create nodes
        nodes = []
        for i in range(num_nodes):
            node = VirtualNode(
                id=i,
                resource=random.uniform(1, 10),
                candidate_domains=random.sample(
                    range(num_domains),
                    k=random.randint(1, min(3, num_domains))
                )
            )
            nodes.append(node)

        # Create links
        links = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() <= connection_rate:
                    link_ij = VirtualLink(
                        src=nodes[i],
                        dest=nodes[j],
                        bandwidth=random.uniform(1, 10)
                    )
                    links[i][j] = link_ij

                    link_ji = VirtualLink(
                        src=nodes[j],
                        dest=nodes[i],
                        bandwidth=link_ij.bandwidth
                    )
                    links[j][i] = link_ji

        return VirtualNetwork(nodes=nodes, links=links)

def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass instances to dictionaries, handling nested structures."""
    if is_dataclass(obj):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def save_dataset_to_json(
    physical_network: PhysicalNetwork,
    virtual_networks: list,
    filename: str = "dataset.json"
) -> None:
    """
    Save physical network and virtual networks to JSON file.
    
    Args:
        physical_network: The physical network object
        virtual_networks: List of virtual network objects
        filename: Output JSON filename
    """
    dataset = {
        "physical_network": dataclass_to_dict(physical_network),
        "virtual_networks": [dataclass_to_dict(vn) for vn in virtual_networks]
    }
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {filename}")


def dict_to_physical_node(node_dict: dict) -> PhysicalNode:
    """Convert dictionary to PhysicalNode."""
    return PhysicalNode(
        internal_id=node_dict["internal_id"],
        external_id=node_dict["external_id"],
        domain_id=node_dict["domain_id"],
        resource=node_dict["resource"],
        used_resource=node_dict["used_resource"],
        cost_per_unit=node_dict["cost_per_unit"],
        delay=node_dict["delay"]
    )


def dict_to_physical_link(link_dict: dict, node_map: dict) -> PhysicalLink:
    """Convert dictionary to PhysicalLink using node_map to resolve node references."""
    src_id = link_dict["src"]["external_id"]
    dest_id = link_dict["dest"]["external_id"]
    
    return PhysicalLink(
        src=node_map[src_id],
        dest=node_map[dest_id],
        bandwidth=link_dict["bandwidth"],
        used_bandwidth=link_dict["used_bandwidth"],
        cost_per_unit=link_dict["cost_per_unit"],
        delay=link_dict["delay"]
    )


def dict_to_physical_domain(
    domain_dict: dict, 
    node_map: dict, 
    create_links: bool = True
) -> PhysicalDomain:
    """Convert dictionary to PhysicalDomain.
    
    Args:
        domain_dict: Dictionary representation of domain
        node_map: Map of external_id to PhysicalNode
        create_links: If False, only create nodes (for two-pass loading)
    """
    # Create nodes for this domain
    nodes = [dict_to_physical_node(node_dict) for node_dict in domain_dict["nodes"]]
    
    # Update node_map with these nodes
    for node in nodes:
        node_map[node.external_id] = node
    
    # Create boundary nodes (they should reference the same node objects)
    boundary_nodes = []
    for boundary_dict in domain_dict["boundary_nodes"]:
        boundary_id = boundary_dict["external_id"]
        # Find the corresponding node in nodes list
        boundary_node = next(n for n in nodes if n.external_id == boundary_id)
        boundary_nodes.append(boundary_node)
    
    # Create intra_links matrix (only if create_links is True)
    num_nodes = len(nodes)
    intra_links = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    if create_links:
        intra_links_dict = domain_dict.get("intra_links", [])
        for i, row in enumerate(intra_links_dict):
            for j, link_dict in enumerate(row):
                if link_dict is not None:
                    link = dict_to_physical_link(link_dict, node_map)
                    intra_links[i][j] = link
    
    # Create inter_links list (only if create_links is True)
    inter_links = []
    if create_links:
        inter_links_dict = domain_dict.get("inter_links", [])
        for link_dict in inter_links_dict:
            link = dict_to_physical_link(link_dict, node_map)
            inter_links.append(link)
    
    # Handle optional fields
    dist = domain_dict.get("dist", [])
    next_node = domain_dict.get("next_node", [])
    
    return PhysicalDomain(
        id=domain_dict["id"],
        nodes=nodes,
        boundary_nodes=boundary_nodes,
        intra_links=intra_links,
        inter_links=inter_links,
        dist=dist,
        next_node=next_node
    )


def dict_to_physical_network(physical_dict: dict) -> PhysicalNetwork:
    """Convert dictionary to PhysicalNetwork."""
    node_map = {}  # Maps external_id to PhysicalNode
    
    # First pass: Create all nodes (without links)
    domains = []
    domain_dicts = []
    for domain_dict in physical_dict["domains"]:
        domain = dict_to_physical_domain(domain_dict, node_map, create_links=False)
        domains.append(domain)
        domain_dicts.append(domain_dict)
    
    # Second pass: Create all links (now all nodes are in node_map)
    for i, domain_dict in enumerate(domain_dicts):
        # Create intra_links
        num_nodes = len(domains[i].nodes)
        intra_links = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]
        intra_links_dict = domain_dict.get("intra_links", [])
        for row_idx, row in enumerate(intra_links_dict):
            for col_idx, link_dict in enumerate(row):
                if link_dict is not None:
                    link = dict_to_physical_link(link_dict, node_map)
                    intra_links[row_idx][col_idx] = link
        domains[i].intra_links = intra_links
        
        # Create inter_links for domain
        inter_links = []
        inter_links_dict = domain_dict.get("inter_links", [])
        for link_dict in inter_links_dict:
            link = dict_to_physical_link(link_dict, node_map)
            inter_links.append(link)
        domains[i].inter_links = inter_links
    
    # Create inter_links for physical network
    inter_links = []
    for link_dict in physical_dict.get("inter_links", []):
        link = dict_to_physical_link(link_dict, node_map)
        inter_links.append(link)
    
    return PhysicalNetwork(
        domains=domains,
        inter_links=inter_links
    )


def dict_to_virtual_node(node_dict: dict) -> VirtualNode:
    """Convert dictionary to VirtualNode."""
    return VirtualNode(
        id=node_dict["id"],
        resource=node_dict["resource"],
        candidate_domains=node_dict["candidate_domains"]
    )


def dict_to_virtual_link(link_dict: dict, node_map: dict) -> VirtualLink:
    """Convert dictionary to VirtualLink using node_map to resolve node references."""
    src_id = link_dict["src"]["id"]
    dest_id = link_dict["dest"]["id"]
    
    return VirtualLink(
        src=node_map[src_id],
        dest=node_map[dest_id],
        bandwidth=link_dict["bandwidth"]
    )


def dict_to_virtual_network(virtual_dict: dict) -> VirtualNetwork:
    """Convert dictionary to VirtualNetwork."""
    # Create all nodes first
    nodes = [dict_to_virtual_node(node_dict) for node_dict in virtual_dict["nodes"]]
    node_map = {node.id: node for node in nodes}
    
    # Create links matrix
    num_nodes = len(nodes)
    links = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    links_dict = virtual_dict.get("links", [])
    for i, row in enumerate(links_dict):
        for j, link_dict in enumerate(row):
            if link_dict is not None:
                link = dict_to_virtual_link(link_dict, node_map)
                links[i][j] = link
    
    return VirtualNetwork(
        nodes=nodes,
        links=links
    )


def load_dataset_from_json(filename: str) -> Tuple[PhysicalNetwork, list]:
    """
    Load physical network and virtual networks from JSON file.
    
    Args:
        filename: Input JSON filename
        
    Returns:
        Tuple of (PhysicalNetwork, list of VirtualNetworks)
    """
    with open(filename, 'r') as f:
        dataset = json.load(f)
    
    # Load physical network
    physical_network = dict_to_physical_network(dataset["physical_network"])
    
    # Load virtual networks
    virtual_networks = [
        dict_to_virtual_network(vn_dict)
        for vn_dict in dataset["virtual_networks"]
    ]
    
    print(f"Dataset loaded from {filename}")
    print(f"Physical Domains: {len(physical_network.domains)}")
    total_nodes = sum(len(d.nodes) for d in physical_network.domains)
    print(f"Total Physical Nodes: {total_nodes}")
    print(f"Inter-domain Links: {len(physical_network.inter_links)}")
    print(f"Virtual Network Requests: {len(virtual_networks)}")
    
    return physical_network, virtual_networks


def handle_generate_dataset():
    """Generate and save dataset."""
    # Configuration parameters
    number_of_physical_network_domain = 4
    number_of_nodes = 30
    number_of_boundary_nodes = 2
    link_connection_rate = 0.5
    number_of_request_nodes = 6
    number_of_requests = 10

    # Generate dataset
    print("Generating dataset...")
    physical_network, virtual_networks = DatasetGenerator.generate_from_config(
        number_of_physical_network_domain=number_of_physical_network_domain,
        number_of_nodes=number_of_nodes,
        number_of_boundary_nodes=number_of_boundary_nodes,
        link_connection_rate=link_connection_rate,
        number_of_request_nodes=number_of_request_nodes,
        number_of_requests=number_of_requests
    )

    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Physical Domains: {len(physical_network.domains)}")
    total_nodes = sum(len(d.nodes) for d in physical_network.domains)
    print(f"Total Physical Nodes: {total_nodes}")
    print(f"Inter-domain Links: {len(physical_network.inter_links)}")
    print(f"Virtual Network Requests: {len(virtual_networks)}")
    print(f"Nodes per Virtual Network: {number_of_request_nodes}")

    # Save to JSON
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"./datasets/dataset-{now}.json"
    save_dataset_to_json(physical_network, virtual_networks, filename)

