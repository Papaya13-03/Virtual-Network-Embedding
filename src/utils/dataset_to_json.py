import json
from typing import Dict, Any

from src.types.substrate import (
    SubstrateDomain,
    SubstrateNode,
    SubstrateLink,
)

from src.types.virtual import (
    VirtualNetwork,
    VirtualNode,
    VirtualLink
)

def dataset_to_json(dataset: Dict[str, Any], filename: str = "dataset.json") -> None:
    """
    Convert dataset to JSON-serializable dict and save to file.
    """

    # ---- Helper functions with type hints ----
    def substrate_node_to_dict(node: SubstrateNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "cpu_capacity": node.cpu_capacity,
            "cost_per_unit": node.cost_per_unit,
            "delay": node.delay,
            "available_cpu": node.available_cpu
        }

    def substrate_link_to_dict(link: SubstrateLink) -> Dict[str, Any]:
        return {
            "src": link.src.node_id,
            "dst": link.dst.node_id,
            "bandwidth": link.bandwidth,
            "cost_per_unit": link.cost_per_unit,
            "delay": link.delay,
            "available_bw": link.available_bw
        }

    def substrate_domain_to_dict(domain: SubstrateDomain) -> Dict[str, Any]:
        return {
            "domain_id": domain.domain_id,
            "nodes": [substrate_node_to_dict(n) for n in domain.nodes],
            "links": [substrate_link_to_dict(l) for l in domain.links]
        }

    def virtual_node_to_dict(node: VirtualNode) -> Dict[str, Any]:
        return {
            "id": node.id,
            "cpu_demand": node.cpu_demand,
            "domains": node.domains
        }

    def virtual_link_to_dict(link: VirtualLink) -> Dict[str, Any]:
        return {
            "src": link.src.id,
            "dst": link.dst.id,
            "bandwidth": link.bandwidth
        }

    def virtual_network_to_dict(vnetwork: VirtualNetwork) -> Dict[str, Any]:
        return {
            "nodes": [virtual_node_to_dict(n) for n in vnetwork.nodes],
            "links": [virtual_link_to_dict(l) for l in vnetwork.links]
        }

    # ---- Build JSON-serializable dict ----
    json_data: Dict[str, Any] = {
        "substrate_network": {
            "domains": [substrate_domain_to_dict(d) for d in dataset["substrate_network"].domains],
            "inter_domain_links": [
                substrate_link_to_dict(l) for l in getattr(dataset["substrate_network"], "links", [])
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

    with open(filename, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Dataset saved to {filename}")
