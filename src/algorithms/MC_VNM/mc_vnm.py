import uuid
from typing import List, Dict
from collections import OrderedDict, deque
from copy import deepcopy

from src.types.virtual import VirtualNetwork, VirtualNode, VirtualLink
from src.types.substrate import SubstrateNetwork, SubstrateDomain, SubstrateNode, SubstrateLink, InterLink


class MC_VNM:
    def __init__(self, substrate_network: SubstrateNetwork):
        self.substrate: SubstrateNetwork = substrate_network
        # request_id -> {"node_mapping", "link_mapping", "expire_time"}
        self._active_mappings: Dict[str, Dict] = OrderedDict()

    # ---------------- MAIN ENTRY ----------------
    def handle_mapping_request(self, vnetwork: VirtualNetwork, current_time: float, lifetime: float = 1000):
        request_id = str(uuid.uuid4())

        # Node mapping
        node_mapping = self.node_mapping(vnetwork)
        if not node_mapping:
            raise ValueError("Node mapping failed")

        # Link mapping (Kruskal)
        link_mapping = self.link_mapping(vnetwork, node_mapping)
        if link_mapping is None:
            raise ValueError("Link mapping failed")

        # Reserve resources
        self.reserve_resources(node_mapping, link_mapping)

        # Compute cost
        cost = self.compute_cost(node_mapping, link_mapping)

        # Save snapshot
        mapping_info = {
            "node_mapping": deepcopy(node_mapping),
            "link_mapping": deepcopy(link_mapping),
            "expire_time": current_time + lifetime
        }
        self._active_mappings[request_id] = mapping_info

        return request_id, cost, mapping_info

    # ---------------- NODE MAPPING ----------------
    def node_mapping(self, vnetwork: VirtualNetwork) -> Dict[VirtualNode, SubstrateNode]:
        mapping: Dict[VirtualNode, SubstrateNode] = {}
        for vnode in vnetwork.nodes:
            candidates: List[SubstrateNode] = []
            for domain in self.substrate.domains:
                if vnode.domains and domain.domain_id not in vnode.domains:
                    continue
                for snode in domain.nodes:
                    if snode.available_cpu >= vnode.cpu_demand:
                        candidates.append(snode)
            if not candidates:
                return {}
            chosen = max(candidates, key=lambda n: n.available_cpu)
            mapping[vnode] = chosen
        return mapping
    
    # ---------------- LINK MAPPING ----------------
    def link_mapping(self, vnetwork: VirtualNetwork, node_mapping: Dict[VirtualNode, SubstrateNode]) -> Dict[VirtualLink, List]:
        result: Dict[VirtualLink, List] = {}
        used_bw_links: Dict[SubstrateLink, float] = {}  # để tạm trừ khi thử

        try:
            for vlink in vnetwork.links:
                src_snode = node_mapping[vlink.src]
                dst_snode = node_mapping[vlink.dst]

                # tìm path với bandwidth >= vlink.bandwidth
                path = self.kruskal_path(src_snode, dst_snode, vlink.bandwidth, used_bw_links)
                if path is None:
                    raise ValueError(f"Cannot map virtual link {vlink.src.id}->{vlink.dst.id}")

                # tạm trừ băng thông trên path
                for link in path:
                    link.available_bw -= vlink.bandwidth
                    used_bw_links[link] = used_bw_links.get(link, 0) + vlink.bandwidth

                result[vlink] = path

        except Exception as e:
            # rollback tạm trừ nếu mapping thất bại
            for link, bw in used_bw_links.items():
                link.available_bw += bw
            return None

        return result

    # ---------------- KRUSKAL PATH (thêm param used_bw_links) ----------------
    def kruskal_path(self, src: SubstrateNode, dst: SubstrateNode, bandwidth: float, used_bw_links: Dict = None) -> List[SubstrateLink]:
        """
        Kruskal + BFS tìm path, xét available_bw - used_bw_links >= bandwidth
        """
        if src == dst:
            return []

        if used_bw_links is None:
            used_bw_links = {}

        # 1. Gather valid links
        valid_links: List[SubstrateLink] = []
        for domain in self.substrate.domains:
            for link in domain.links:
                avail = link.available_bw - used_bw_links.get(link, 0)
                if avail >= bandwidth:
                    valid_links.append(link)
        for ilink in self.substrate.links:
            avail = ilink.available_bw - used_bw_links.get(ilink, 0)
            if avail >= bandwidth:
                valid_links.append(ilink)

        if not valid_links:
            return None

        # 2. Kruskal MST
        parent = {}
        def find(u):
            parent.setdefault(u, u)
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            parent[pu] = pv
            return True

        valid_links.sort(key=lambda l: getattr(l, "cost_per_unit", 1.0))
        tree_links: List[SubstrateLink] = []
        for link in valid_links:
            if union(link.src, link.dst):
                tree_links.append(link)
            if find(src) == find(dst):
                break

        # 3. BFS tìm path src->dst trong tree_links
        adj: Dict[SubstrateNode, List[SubstrateLink]] = {}
        for link in tree_links:
            adj.setdefault(link.src, []).append(link)
            adj.setdefault(link.dst, []).append(link)

        visited = set()
        queue = deque([(src, [])])
        while queue:
            node, path = queue.popleft()
            if node == dst:
                return path
            visited.add(node)
            for link in adj.get(node, []):
                neighbor = link.dst if link.src == node else link.src
                if neighbor not in visited:
                    queue.append((neighbor, path + [link]))

        return None


    # ---------------- RESOURCES ----------------
    def reserve_resources(self, node_mapping, link_mapping):
        for vnode, snode in node_mapping.items():
            snode.available_cpu -= vnode.cpu_demand
        for [vlink, path] in link_mapping.items():
            for link in path:
                link.available_bw -= getattr(vlink, "bandwidth", 0)

    # ---------------- COST FUNCTION ----------------
    def compute_cost(self, node_mapping, link_mapping) -> float:
        node_cost = sum(snode.cost_per_unit * vnode.cpu_demand for vnode, snode in node_mapping.items())
        link_cost = sum(getattr(link, "cost_per_unit", 1.0) * getattr(vlink, "bandwidth", 1.0)
                        for [vlink, path] in link_mapping.items() for link in path)
        return node_cost + link_cost

    # ---------------- RELEASE EXPIRED ----------------
    def release_expired_requests(self, current_time: float):
        expired_ids = [rid for rid, info in self._active_mappings.items() if info["expire_time"] <= current_time]
        for rid in expired_ids:
            info = self._active_mappings.pop(rid)
            # release CPU
            for vnode, snode in info["node_mapping"].items():
                snode.available_cpu += vnode.cpu_demand
            # release bandwidth
            for path in info["link_mapping"].values():
                for link in path:
                    link.available_bw += getattr(link, "bandwidth", 0)
