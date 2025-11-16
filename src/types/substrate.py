import random
from typing import List, Dict

class SubstrateNode:
    def __init__(self, node_id: int, cpu_capacity: float, cost_per_unit: float, delay: float = 0.0):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.cost_per_unit = cost_per_unit
        self.delay = delay
        self.available_cpu = cpu_capacity

class SubstrateLink:
    def __init__(self, src: SubstrateNode, dst: SubstrateNode, bandwidth: float, cost_per_unit: float, delay: float = 0.0):
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth
        self.cost_per_unit = cost_per_unit
        self.delay = delay
        self.available_bw = bandwidth

class SubstrateDomain:
    def __init__(self, domain_id: int):
        self.domain_id = domain_id
        self.nodes: List[SubstrateNode] = []
        self.links: List[SubstrateLink] = []

    def add_node(self, snode: SubstrateNode):
        self.nodes.append(snode)

    def add_link(self, slink: SubstrateLink):
        self.links.append(slink)

class InterLink:
    def __init__(self, src_domain: SubstrateDomain, dst_domain: SubstrateDomain, src: SubstrateNode, dst: SubstrateNode, bandwidth: float, cost_per_unit: float, delay: float = 0.0):
        self.src_domain = src_domain
        self.dst_domain = dst_domain
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth
        self.cost_per_unit = cost_per_unit
        self.delay = delay
        self.available_bw = bandwidth

class SubstrateNetwork:
    def __init__(self):
        self.domains: List[SubstrateDomain] = []
        self.links: List[InterLink] = []

    def add_domain(self, domain: SubstrateDomain):
        self.domains.append(domain)

    def add_link(self, inter_link: InterLink):
        self.links.append(inter_link)
