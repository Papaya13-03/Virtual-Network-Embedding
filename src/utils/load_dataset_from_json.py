import json
from typing import Dict, Any, List
from src.types.substrate import (
    SubstrateNetwork,
    SubstrateDomain,
    SubstrateNode,
    SubstrateLink,
    InterLink
)
from src.types.virtual import (
    VirtualNetwork,
    VirtualNode,
    VirtualLink
)

def load_dataset_from_json(filename: str) -> Dict[str, Any]:
    """
    Load dataset from JSON file and reconstruct SubstrateNetwork and VirtualNetwork objects.

    Returns:
        {
            "substrate_network": SubstrateNetwork,
            "virtual_requests": List[dict]  # each with keys "vnetwork", "arrival_time", "lifetime"
        }
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # ---- Reconstruct SubstrateNetwork ----
    substrate_data = data["substrate_network"]
    substrate_network = SubstrateNetwork()

    # Domains
    for d in substrate_data["domains"]:
        domain = SubstrateDomain(domain_id=d["domain_id"])
        # Nodes
        for n in d["nodes"]:
            node = SubstrateNode(
                node_id=n["node_id"],
                cpu_capacity=n["cpu_capacity"],
                cost_per_unit=n["cost_per_unit"],
                delay=n["delay"]
            )
            node.available_cpu = n.get("available_cpu", node.cpu_capacity)
            domain.add_node(node)
        # Links
        for l in d["links"]:
            src_node = next(node for node in domain.nodes if node.node_id == l["src"])
            dst_node = next(node for node in domain.nodes if node.node_id == l["dst"])
            link = SubstrateLink(
                src=src_node,
                dst=dst_node,
                bandwidth=l["bandwidth"],
                cost_per_unit=l["cost_per_unit"],
                delay=l["delay"]
            )
            link.available_bw = l.get("available_bw", link.bandwidth)
            domain.add_link(link)
        substrate_network.add_domain(domain)

    # Inter-domain links
    for l in substrate_data.get("inter_domain_links", []):
        src_domain = substrate_network.domains[l["src_domain"]]
        dst_domain = substrate_network.domains[l["dst_domain"]]
        src_node = next(node for node in src_domain.nodes if node.node_id == l["src"])
        dst_node = next(node for node in dst_domain.nodes if node.node_id == l["dst"])
        inter_link = InterLink(
            src_domain=src_domain,
            dst_domain=dst_domain,
            src=src_node,
            dst=dst_node,
            bandwidth=l["bandwidth"],
            cost_per_unit=l["cost_per_unit"],
            delay=l["delay"]
        )
        inter_link.available_bw = l.get("available_bw", inter_link.bandwidth)
        substrate_network.add_link(inter_link)

    # ---- Reconstruct Virtual Requests ----
    virtual_requests: List[Dict[str, Any]] = []
    for req in data["virtual_requests"]:
        vnodes = [VirtualNode(n["id"], n["cpu_demand"], n["domains"]) for n in req["vnetwork"]["nodes"]]
        vlinks = []
        for l in req["vnetwork"]["links"]:
            src_node = next(n for n in vnodes if n.id == l["src"])
            dst_node = next(n for n in vnodes if n.id == l["dst"])
            vlink = VirtualLink(src_node, dst_node, l["bandwidth"])
            vlinks.append(vlink)
        vnetwork = VirtualNetwork(nodes=vnodes, links=vlinks)
        virtual_requests.append({
            "vnetwork": vnetwork,
            "arrival_time": req["arrival_time"],
            "lifetime": req["lifetime"]
        })

    return {
        "substrate_network": substrate_network,
        "virtual_requests": virtual_requests
    }
