import json
from typing import Dict, Any

from src.types.substrate import (
    SubstrateNetwork,
    SubstrateDomain,
    SubstrateNode,
    SubstrateLink,
    InterLink
)
from src.types.virtual import VirtualNetwork, VirtualNode, VirtualLink


def dataset_to_json(dataset: Dict[str, Any], filename: str = "dataset.json"):
    """
    Convert dataset to JSON-serializable dict and save to file.
    Handles both intra-domain links, inter-domain links, and boundary nodes.
    """

    # ---- Helpers ----
    def substrate_node_to_dict(node: SubstrateNode):
        return {
            "node_id": node.node_id,
            "cpu_capacity": node.cpu_capacity,
            "cost_per_unit": node.cost_per_unit,
            "delay": node.delay,
            "available_cpu": node.available_cpu
        }

    def substrate_link_to_dict(link: SubstrateLink):
        return {
            "src": link.src.node_id,
            "dst": link.dst.node_id,
            "bandwidth": link.bandwidth,
            "cost_per_unit": link.cost_per_unit,
            "delay": link.delay,
            "available_bw": link.available_bw
        }

    def substrate_domain_to_dict(domain: SubstrateDomain):
        return {
            "domain_id": domain.domain_id,
            "nodes": [substrate_node_to_dict(n) for n in domain.nodes],
            "links": [substrate_link_to_dict(l) for l in domain.links],
            "boundary_nodes": [n.node_id for n in getattr(domain, "boundary_nodes", [])]
        }

    def interlink_to_dict(link: InterLink):
        return {
            "src_domain": link.src_domain.domain_id,
            "dst_domain": link.dst_domain.domain_id,
            "src": link.src.node_id,
            "dst": link.dst.node_id,
            "bandwidth": link.bandwidth,
            "cost_per_unit": link.cost_per_unit,
            "delay": link.delay,
            "available_bw": link.available_bw
        }

    def virtual_node_to_dict(node: VirtualNode):
        return {
            "id": node.id,
            "cpu_demand": node.cpu_demand,
            "domains": node.domains
        }

    def virtual_link_to_dict(link: VirtualLink):
        return {
            "src": link.src.id,
            "dst": link.dst.id,
            "bandwidth": link.bandwidth
        }

    def virtual_network_to_dict(vnetwork: VirtualNetwork):
        return {
            "nodes": [virtual_node_to_dict(n) for n in vnetwork.nodes],
            "links": [virtual_link_to_dict(l) for l in vnetwork.links]
        }

    # ---- Build JSON dict ----
    json_data = {
        "substrate_network": {
            "domains": [substrate_domain_to_dict(d) for d in dataset["substrate_network"].domains],
            "inter_domain_links": [
                interlink_to_dict(l) for l in getattr(dataset["substrate_network"], "links", [])
            ]
        },
        "virtual_requests": [
            {
                "vnetwork": virtual_network_to_dict(req["vnetwork"]),
                "arrival_time": req["arrival_time"],
                "lifetime": req["lifetime"]
            } for req in dataset["virtual_requests"]
        ]
    }

    # ---- Save to file ----
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Dataset saved to {filename}")
