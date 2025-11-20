from typing import List, Dict
from src.types.substrate import SubstrateNetwork, SubstrateDomain, SubstrateNode, SubstrateLink, InterLink
from src.types.virtual import VirtualNetwork, VirtualNode, VirtualLink

class MC_VNE_TimeSeries:
    def __init__(self, substrate_network: SubstrateNetwork):
        self.substrate = substrate_network
        # Maintain allocated resources over time
        # Could extend to more sophisticated time-aware resource tracking
        self.allocations = []

    def embed_request(self, vnetwork: VirtualNetwork, current_time: float, lifetime: float) -> bool:
        """
        Attempt to embed a virtual network request at current_time
        Returns True if embedding successful, False otherwise
        """
        # Step 1: Node Mapping
        node_mapping = self.node_mapping(vnetwork)

        if not node_mapping:
            return False  # cannot map nodes

        # Step 2: Link Mapping
        link_mapping = self.link_mapping(vnetwork, node_mapping)

        if not link_mapping:
            return False  # cannot map links

        # Step 3: Reserve resources
        self.reserve_resources(node_mapping, link_mapping, current_time, lifetime)

        return True

    def node_mapping(self, vnetwork: VirtualNetwork) -> Dict[VirtualNode, SubstrateNode]:
        """
        Map virtual nodes to substrate nodes with sufficient CPU and domain constraints
        """
        mapping = {}
        for vnode in vnetwork.nodes:
            candidates = []
            for domain in self.substrate.domains:
                if vnode.domains and domain.domain_id not in vnode.domains:
                    continue  # domain constraint
                for snode in domain.nodes:
                    if snode.available_cpu >= vnode.cpu_demand:
                        candidates.append(snode)
            if not candidates:
                return {}  # failed mapping
            # Simple heuristic: choose node with max available CPU
            chosen = max(candidates, key=lambda n: n.available_cpu)
            mapping[vnode] = chosen
        return mapping

    def link_mapping(self, vnetwork: VirtualNetwork, node_mapping: Dict[VirtualNode, SubstrateNode]) -> Dict[VirtualLink, List]:
        """
        Map virtual links to substrate paths. Return dict {vlink: path_links}
        Here we use a simple shortest-path on available bandwidth.
        """
        link_mapping = {}
        for vlink in vnetwork.links:
            src_snode = node_mapping[vlink.src]
            dst_snode = node_mapping[vlink.dst]
            path = self.find_path(src_snode, dst_snode, vlink.bandwidth)
            if not path:
                return {}  # failed mapping
            link_mapping[vlink] = path
        return link_mapping

    def find_path(self, src: SubstrateNode, dst: SubstrateNode, bandwidth: float) -> List[SubstrateLink]:
        """
        Simple BFS/DFS for path with enough available bandwidth
        Can be replaced by Dijkstra or k-shortest path considering cost/delay
        """
        visited = set()
        stack = [(src, [])]
        while stack:
            current, path = stack.pop()
            if current == dst:
                return path
            visited.add(current)
            for link in current.links:  # assuming each node knows its incident links
                neighbor = link.dst if link.src == current else link.src
                if neighbor in visited:
                    continue
                if link.available_bw >= bandwidth:
                    stack.append((neighbor, path + [link]))
        return []

    def reserve_resources(self, node_mapping, link_mapping, current_time: float, lifetime: float):
        """
        Deduct resources from substrate network and record allocation with expiry
        """
        for vnode, snode in node_mapping.items():
            snode.available_cpu -= vnode.cpu_demand
        for vlink, path_links in link_mapping.items():
            for link in path_links:
                link.available_bw -= vlink.bandwidth
        # Record allocation for later release
        self.allocations.append({
            "node_mapping": node_mapping,
            "link_mapping": link_mapping,
            "expiry": current_time + lifetime
        })

    def release_expired(self, current_time: float):
        """
        Release resources whose lifetime has expired
        """
        new_allocations = []
        for alloc in self.allocations:
            if alloc["expiry"] <= current_time:
                # release nodes
                for vnode, snode in alloc["node_mapping"].items():
                    snode.available_cpu += vnode.cpu_demand
                # release links
                for vlink, path_links in alloc["link_mapping"].items():
                    for link in path_links:
                        link.available_bw += vlink.bandwidth
            else:
                new_allocations.append(alloc)
        self.allocations = new_allocations
