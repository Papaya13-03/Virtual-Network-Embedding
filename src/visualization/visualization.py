"""Visualization utilities for VNE project."""
import os
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from ..types.substrate import PhysicalNetwork
from ..types.virtual import VirtualNetwork


def visualize_physical_network(network: PhysicalNetwork):
    """Visualize a physical network with domains, nodes, and links."""
    G = nx.Graph()

    for domain in network.domains:
        for node in domain.nodes:
            G.add_node(
                node.external_id,
                domain=domain.id,
                resource=node.resource,
                cost=node.cost_per_unit,
                delay=node.delay,
            )

    for domain in network.domains:
        for row in domain.intra_links:
            for link in row:
                if link:
                    G.add_edge(
                        link.src.external_id,
                        link.dest.external_id,
                        bandwidth=link.bandwidth,
                        type="intra"
                    )

    for link in network.inter_links:
        G.add_edge(
            link.src.external_id,
            link.dest.external_id,
            bandwidth=link.bandwidth,
            type="inter"
        )

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=200)
    plt.figure(figsize=(12, 9))

    domain_colors = {d.id: f"C{d.id % 10}" for d in network.domains}

    for domain in network.domains:
        nodes = [n.external_id for n in domain.nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=domain_colors[domain.id],
            label=f"Domain {domain.id}",
            node_size=700
        )

    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "intra"]
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "inter"]

    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, style="solid", width=1.5, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=inter_edges, style="dotted", width=2.2, edge_color="black")

    node_labels = {
        n: f"{n}\n({d['resource']:.1f})"
        for n, d in G.nodes(data=True)
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    edge_labels = {
        (u, v): f"{d['bandwidth']:.1f}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Physical Network Visualization")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    
    # Save figure to file with unique name
    os.makedirs("visualization", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"visualization/physical_network_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Physical network visualization saved to: {output_file}")
    
    plt.show()


def visualize_virtual_network(virtual_network: VirtualNetwork):
    """Visualize a virtual network with nodes and links."""
    G = nx.Graph()
    for node in virtual_network.nodes:
        G.add_node(node.id, resource=node.resource)

    for i in range(len(virtual_network.nodes)):
        for j in range(i + 1, len(virtual_network.nodes)):
            link = virtual_network.links[i][j]
            if link:
                G.add_edge(link.src.id, link.dest.id, bandwidth=link.bandwidth)

    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=150)

    node_labels = {
        n: f"{n}\n({d['resource']:.1f})"
        for n, d in G.nodes(data=True)
    }

    edge_labels = {
        (u, v): f"{d['bandwidth']:.1f}"
        for u, v, d in G.edges(data=True)
    }

    plt.figure(figsize=(7, 6))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color='skyblue',
        node_size=800,
        edge_color='gray',
        width=1.5
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Virtual Network Visualization")
    plt.axis("off")
    plt.tight_layout()
    
    # Save figure to file with unique name
    os.makedirs("visualization", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"visualization/virtual_network_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Virtual network visualization saved to: {output_file}")
    
    plt.show()


def visualize_array(arr, title, x, y):
    """Visualize an array as a line plot."""
    plt.plot(arr, marker='o')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()

