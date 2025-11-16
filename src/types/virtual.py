from typing import List

class VirtualNode:
    def __init__(self, node_id: int, cpu_demand: float, domains: List[int]):
        self.id = node_id
        self.cpu_demand = cpu_demand
        self.domains = domains

class VirtualLink:
    def __init__(self, src: VirtualNode, dst: VirtualNode, bandwidth: float):
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth

class VirtualNetwork:
    def __init__(self, nodes: List[VirtualNode] = None, links: List[VirtualLink] = None):
        self.nodes = nodes if nodes is not None else []
        self.links = links if links is not None else []
