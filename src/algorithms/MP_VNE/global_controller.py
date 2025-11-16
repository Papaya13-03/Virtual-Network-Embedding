from algorithms.MP_VNE.local_controller import LocalController
from types.substrate import SubstrateNetwork

class GlobalController:
    def __init__(self, snetwork: SubstrateNetwork):
        self.snetwork = snetwork
        self.local_controllers = [LocalController(domain) for domain in snetwork.domains]

    def process_request(self, request):
        candidates = []
        for vnode in request.nodes:
            node_candidates = []
            for lc in self.local_controllers:
                node_candidates.extend(lc.get_candidates(vnode))
            candidates.append(node_candidates)
        return candidates

    def link_cost(self, node_a, node_b):
        """
        Tính chi phí ánh xạ liên kết giữa hai nút vật lý:
        - Dùng Floyd hoặc Dijkstra để tìm đường đi ngắn nhất.
        - Chi phí = tổng độ trễ + băng thông sử dụng.
        """
        # Giả sử snetwork có hàm shortest_path(node_a, node_b)
        path = self.snetwork.shortest_path(node_a, node_b)
        cost = sum(link.delay + link.cost for link in path)
        return cost

    def commit_mapping(self, mapping):
        """
        Thực hiện ánh xạ thực tế:
        - Gán nút ảo -> nút vật lý.
        - Gán liên kết ảo -> đường đi vật lý.
        - Cập nhật tài nguyên.
        """
        for vnode, snode in enumerate(mapping):
            snode.allocate_cpu(vnode.cpu_demand)
        # Ánh xạ liên kết
        for vlink in self.snetwork.virtual_links:
            path = self.snetwork.shortest_path(mapping[vlink.src], mapping[vlink.dst])
            for link in path:
                link.allocate_bw(vlink.bw_demand)

    def release_resources(self):
        """
        Giải phóng tài nguyên khi mạng ảo hết vòng đời hoặc ánh xạ thất bại.
        """
        self.snetwork.reset_allocations()

    def calculate_pre_cost(self, vnode, candidate):
        """
        Tính chi phí dự đoán trước (PreCost) cho mỗi ứng viên:
        PreCost = CPU(n_v) * C(n_s) + Σ(BW(l) * C_link)/NoL
        """
        cpu_cost = vnode.cpu_demand * candidate.cpu_cost
        link_cost = 0
        for vlink in vnode.links:
            link_cost += vlink.bw_demand * candidate.link_cost / max(1, len(vnode.links))
        return cpu_cost + link_cost