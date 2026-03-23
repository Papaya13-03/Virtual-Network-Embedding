import json
import random
import os
import argparse
import yaml
from typing import Dict, Any, List

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_domain(domain_id: str, cfg: Dict[str, Any], num_nodes: int = 50) -> Dict[str, Any]:
    nodes = []
    links = []

    domain_cfg = cfg['substrate']['intra_domain']
    edge_prob = domain_cfg.get('edge_prob', 0.5)

    for i in range(num_nodes):
        node_id = f"{domain_id}_node_{i}"
        nodes.append({
            "id": node_id,
            "cpu_capacity": round(random.uniform(domain_cfg['cpu_capacity']['min'], domain_cfg['cpu_capacity']['max']), 2),
            "cpu_price": round(random.uniform(domain_cfg['cpu_price']['min'], domain_cfg['cpu_price']['max']), 2),
            "processing_delay": round(random.uniform(domain_cfg['processing_delay']['min'], domain_cfg['processing_delay']['max']), 2)
        })

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                links.append({
                    "source": f"{domain_id}_node_{i}",
                    "target": f"{domain_id}_node_{j}",
                    "bandwidth_capacity": round(random.uniform(domain_cfg['bandwidth_capacity']['min'], domain_cfg['bandwidth_capacity']['max']), 2),
                    "bandwidth_price": round(random.uniform(domain_cfg['bandwidth_price']['min'], domain_cfg['bandwidth_price']['max']), 2),
                    "transmission_delay": round(random.uniform(domain_cfg['transmission_delay']['min'], domain_cfg['transmission_delay']['max']), 2)
                })

    return {
        "id": domain_id,
        "nodes": nodes,
        "links": links
    }

def generate_multi_domain_substrate(cfg: Dict[str, Any], num_domains: int = 3, nodes_per_domain: int = 50) -> Dict[str, Any]:
    domains = []
    inter_domain_links = []
    
    # Generate individual domains
    for d in range(num_domains):
        domain = generate_domain(f"domain_{d}", cfg, num_nodes=nodes_per_domain)
        domains.append(domain)
        
    inter_cfg = cfg['substrate']['inter_domain']
    inter_domain_prob = inter_cfg.get('edge_prob', 0.1)

    # Generate inter-domain links
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            # Select random boundary nodes
            nodes_i = domains[i]["nodes"]
            nodes_j = domains[j]["nodes"]
            
            # Create a few links between domains
            num_inter_links = random.randint(1, max(1, int(nodes_per_domain * inter_domain_prob)))
            for _ in range(num_inter_links):
                src_node = random.choice(nodes_i)["id"]
                dst_node = random.choice(nodes_j)["id"]
                
                inter_domain_links.append({
                    "source": src_node,
                    "target": dst_node,
                    "bandwidth_capacity": round(random.uniform(inter_cfg['bandwidth_capacity']['min'], inter_cfg['bandwidth_capacity']['max']), 2), 
                    "bandwidth_price": round(random.uniform(inter_cfg['bandwidth_price']['min'], inter_cfg['bandwidth_price']['max']), 2),
                    "transmission_delay": round(random.uniform(inter_cfg['transmission_delay']['min'], inter_cfg['transmission_delay']['max']), 2)
                })

    return {
        "domains": domains,
        "inter_domain_links": inter_domain_links
    }

def generate_virtual_request(vnr_id: str, arrival_time: float, cfg: Dict[str, Any], num_domains: int = 3) -> Dict[str, Any]:
    vnr_cfg = cfg['virtual_request']
    
    num_nodes = random.randint(vnr_cfg['num_nodes']['min'], vnr_cfg['num_nodes']['max'])
    lifetime = round(random.expovariate(1.0 / vnr_cfg['lifetime_mean']), 2)
    edge_prob = vnr_cfg.get('edge_prob', 0.5)
    
    nodes = []
    links = []

    for i in range(num_nodes):
        allowed = []
        if random.random() < 0.6: # 60% chance to have regional constraint
            num_allowed = random.randint(1, min(2, num_domains))
            allowed = random.sample([f"domain_{d}" for d in range(num_domains)], num_allowed)

        nodes.append({
            "id": f"{vnr_id}_node_{i}",
            "cpu_demand": round(random.uniform(vnr_cfg['cpu_demand']['min'], vnr_cfg['cpu_demand']['max']), 2),
            "allowed_domains": allowed
        })
        
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                 links.append({
                    "source": f"{vnr_id}_node_{i}",
                    "target": f"{vnr_id}_node_{j}",
                    "bandwidth_demand": round(random.uniform(vnr_cfg['bandwidth_demand']['min'], vnr_cfg['bandwidth_demand']['max']), 2)
                })
                 
    # Ensure graph is connected (simple line connection mapping)
    for i in range(num_nodes - 1):
        found = False
        for link in links:
            if (link["source"] == f"{vnr_id}_node_{i}" and link["target"] == f"{vnr_id}_node_{i+1}") or \
               (link["target"] == f"{vnr_id}_node_{i}" and link["source"] == f"{vnr_id}_node_{i+1}"):
                found = True
                break
        if not found:
             links.append({
                "source": f"{vnr_id}_node_{i}",
                "target": f"{vnr_id}_node_{i+1}",
                "bandwidth_demand": round(random.uniform(vnr_cfg['bandwidth_demand']['min'], vnr_cfg['bandwidth_demand']['max']), 2)
            })

    return {
        "id": vnr_id,
        "arrival_time": round(arrival_time, 2),
        "lifetime": lifetime,
        "virtual_network": {
            "id": f"vn_{vnr_id}",
            "nodes": nodes,
            "links": links
        }
    }

def generate_dataset(scenario_name: str, config_path: str, num_domains: int = 3, nodes_per_domain: int = 50, num_requests: int = 100):
    cfg = load_config(config_path)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scenario_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Multi-Domain Substrate Network
    substrate = generate_multi_domain_substrate(cfg, num_domains=num_domains, nodes_per_domain=nodes_per_domain)
    with open(os.path.join(output_dir, "substrate.json"), "w") as f:
        json.dump(substrate, f, indent=2)

    # 2. Generate Virtual Requests
    requests = []
    current_time = 0.0
    
    arrival_rate = cfg['virtual_request']['arrival_rate']
    
    for i in range(num_requests):
        inter_arrival_time = random.expovariate(arrival_rate)
        current_time += inter_arrival_time
        vnr = generate_virtual_request(f"vnr_{i}", current_time, cfg, num_domains=num_domains)
        requests.append(vnr)

    with open(os.path.join(output_dir, "virtual_requests.json"), "w") as f:
        json.dump(requests, f, indent=2)

    print(f"Dataset generated successfully at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate VNE Benchmark Dataset (Multi-Domain)")
    parser.add_argument("--scenario", type=str, required=True, help="Name of the scenario directory (e.g., scenario_1)")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--domains", type=int, default=3, help="Number of substrate domains")
    parser.add_argument("--nodes_per_domain", type=int, default=50, help="Number of nodes per domain")
    parser.add_argument("--requests", type=int, default=100, help="Number of virtual network requests")
    
    args = parser.parse_args()
    
    generate_dataset(args.scenario, args.config, args.domains, args.nodes_per_domain, args.requests)

if __name__ == "__main__":
    main()
