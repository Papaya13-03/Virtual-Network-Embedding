import json
import os
from typing import List

from problem.substrate_network import SubstrateNode, SubstrateLink, SubstrateNetwork
from problem.domain import PhysicalDomain, MultiDomainNetwork
from problem.virtual_network import VirtualNode, VirtualLink, VirtualNetwork
from problem.request import VirtualNetworkRequest

def read_substrate(filepath: str) -> MultiDomainNetwork:
    """Reads a multi-domain substrate network from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Substrate file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    md_network = MultiDomainNetwork()

    # Parse Domains
    for domain_data in data.get("domains", []):
        domain_id = domain_data["id"]
        
        nodes = {}
        for n in domain_data.get("nodes", []):
            nodes[n["id"]] = SubstrateNode(
                id=n["id"],
                cpu_capacity=n["cpu_capacity"],
                cpu_price=n["cpu_price"],
                processing_delay=n["processing_delay"]
            )
            
        links = {}
        for l in domain_data.get("links", []):
            links[(l["source"], l["target"])] = SubstrateLink(
                source=l["source"],
                target=l["target"],
                bandwidth_capacity=l["bandwidth_capacity"],
                bandwidth_price=l["bandwidth_price"],
                transmission_delay=l["transmission_delay"]
            )
            
        s_network = SubstrateNetwork(nodes=nodes, links=links)
        md_network.domains[domain_id] = PhysicalDomain(id=domain_id, network=s_network)

    # Parse Inter-Domain Links
    for l in data.get("inter_domain_links", []):
        md_network.inter_domain_links[(l["source"], l["target"])] = SubstrateLink(
            source=l["source"],
            target=l["target"],
            bandwidth_capacity=l["bandwidth_capacity"],
            bandwidth_price=l["bandwidth_price"],
            transmission_delay=l["transmission_delay"]
        )

    return md_network


def read_virtual_requests(filepath: str) -> List[VirtualNetworkRequest]:
    """Reads a list of virtual network requests from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Requests file not found: {filepath}")

    with open(filepath, 'r') as f:
        data_list = json.load(f)

    requests = []
    for data in data_list:
        v_net_data = data["virtual_network"]
        
        nodes = {}
        for n in v_net_data.get("nodes", []):
            nodes[n["id"]] = VirtualNode(
                id=n["id"],
                cpu_demand=n["cpu_demand"]
            )
            
        links = {}
        for l in v_net_data.get("links", []):
            links[(l["source"], l["target"])] = VirtualLink(
                source=l["source"],
                target=l["target"],
                bandwidth_demand=l["bandwidth_demand"]
            )
            
        v_network = VirtualNetwork(
            id=v_net_data["id"],
            nodes=nodes,
            links=links
        )

        requests.append(VirtualNetworkRequest(
            id=data["id"],
            virtual_network=v_network,
            arrival_time=data["arrival_time"],
            lifetime=data["lifetime"]
        ))

    return requests
