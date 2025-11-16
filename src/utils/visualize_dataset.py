import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import os
from typing import Dict, Any

def visualize_dataset(dataset: Dict[str, Any], output_dir="./assets/visualization"):
    os.makedirs(output_dir, exist_ok=True)

    substrate = dataset["substrate_network"]
    virtual_requests = dataset["virtual_requests"]

    # -------------------
    # 1. Substrate network visualization
    # -------------------
    G = nx.Graph()
    domain_colors_map = {}
    cmap = plt.cm.tab10

    # Add nodes with label: ID + CPU
    for idx, domain in enumerate(substrate.domains):
        domain_colors_map[domain.domain_id] = cmap(idx % 10)
        for node in domain.nodes:
            node_label = f"{node.node_id}\nCPU:{node.cpu_capacity:.0f}"
            G.add_node(node_label, domain=domain.domain_id, node_obj=node)

    # Intra-domain links (solid)
    for domain in substrate.domains:
        for link in domain.links:
            src_label = f"{link.src.node_id}\nCPU:{link.src.cpu_capacity:.0f}"
            dst_label = f"{link.dst.node_id}\nCPU:{link.dst.cpu_capacity:.0f}"
            G.add_edge(src_label, dst_label, style='solid',
                    label=f"BW:{link.bandwidth:.0f},C:{link.cost_per_unit:.1f},D:{link.delay:.1f}")

    # Inter-domain links (dashed)
    for link in substrate.links:
        src_label = f"{link.src.node_id}\nCPU:{link.src.cpu_capacity:.0f}"
        dst_label = f"{link.dst.node_id}\nCPU:{link.dst.cpu_capacity:.0f}"
        G.add_edge(src_label, dst_label, style='dashed',
                label=f"BW:{link.bandwidth:.0f},C:{link.cost_per_unit:.1f},D:{link.delay:.1f}")

    # Layout rộng, ít chồng lấn
    pos = nx.kamada_kawai_layout(G)

    node_colors = [domain_colors_map[G.nodes[n]["domain"]] for n in G.nodes()]

    plt.figure(figsize=(28, 20))  # ảnh rộng hơn
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)  # node to hơn
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')  # font to

    # Draw edges
    solid_edges = [e for e in G.edges(data=True) if e[2].get('style') == 'solid']
    dashed_edges = [e for e in G.edges(data=True) if e[2].get('style') == 'dashed']

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, style='solid', alpha=0.7, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style='dashed', alpha=0.7, width=3)

    # Edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Legend
    for domain_id, color in domain_colors_map.items():
        plt.scatter([], [], c=[color], label=f"Domain {domain_id}")
    plt.legend(loc="upper right", markerscale=2)

    plt.title("Substrate Network (Node ID + CPU inside, dashed = inter-domain)", fontsize=18)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "substrate_network.png"))
    plt.close()

    # -------------------
    # 2. Node CPU and Link Bandwidth Distribution
    # -------------------
    all_cpu = [node.cpu_capacity for domain in substrate.domains for node in domain.nodes]
    all_bw = [link.bandwidth for domain in substrate.domains for link in domain.links] + \
             [link.bandwidth for link in substrate.links]

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(all_cpu, bins=15, ax=ax[0], kde=True, color='skyblue')
    ax[0].set_title("Substrate Node CPU Distribution")
    ax[0].set_xlabel("CPU Capacity")
    ax[0].set_ylabel("Count")

    sns.histplot(all_bw, bins=15, ax=ax[1], kde=True, color='salmon')
    ax[1].set_title("Substrate Link Bandwidth Distribution")
    ax[1].set_xlabel("Bandwidth")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "substrate_distributions.png"))
    plt.close()

    # -------------------
    # 3. Virtual requests active timeline
    # -------------------
    if virtual_requests:
        max_time = max([req["arrival_time"] + req["lifetime"] for req in virtual_requests])
        times = np.linspace(0, max_time, num=500)
        active_counts = [sum(1 for req in virtual_requests if req["arrival_time"] <= t <= req["arrival_time"] + req["lifetime"]) for t in times]

        plt.figure(figsize=(10,5))
        plt.plot(times, active_counts, color='blue')
        plt.xlabel("Time")
        plt.ylabel("Number of active virtual requests")
        plt.title("Virtual Requests Active Over Time")
        plt.savefig(os.path.join(output_dir, "virtual_requests_timeline.png"))
        plt.close()

    print(f"Visualization saved in '{output_dir}'")
