from typing import Dict, Tuple
import torch

from problem.domain import PhysicalDomain
from problem.virtual_network import VirtualNetwork


def extract_domain_features(domain: PhysicalDomain) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-snode feature matrix X and normalized adjacency A for one domain.

    X columns: [avail_cpu_ratio, cpu_price_norm, proc_delay_norm, degree_norm, avg_neighbor_bw_norm]
    A: D^{-1/2} (A_bw + I) D^{-1/2}, where A_bw edge weight = available_bw / capacity.
    """
    net = domain.network
    node_ids = list(net.nodes.keys())
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    degrees = [0] * n
    neighbor_bw_sum = [0.0] * n
    for (u, v), link in net.links.items():
        bw = getattr(link, "available_bw", link.bandwidth_capacity)
        if u in idx:
            degrees[idx[u]] += 1
            neighbor_bw_sum[idx[u]] += bw
        if v in idx:
            degrees[idx[v]] += 1
            neighbor_bw_sum[idx[v]] += bw

    max_degree = max(max(degrees), 1)
    max_cap = max((nd.cpu_capacity for nd in net.nodes.values()), default=1.0) or 1.0
    max_avg_nbr_bw = max(
        (neighbor_bw_sum[i] / degrees[i] for i in range(n) if degrees[i] > 0),
        default=1.0,
    ) or 1.0

    X = torch.zeros(n, 5)
    for i, nid in enumerate(node_ids):
        node = net.nodes[nid]
        avail = getattr(node, "available_cpu", node.cpu_capacity)
        X[i, 0] = avail / max_cap
        X[i, 1] = node.cpu_price / 10.0
        X[i, 2] = node.processing_delay / 10.0
        X[i, 3] = degrees[i] / max_degree
        if degrees[i] > 0:
            X[i, 4] = (neighbor_bw_sum[i] / degrees[i]) / max_avg_nbr_bw

    A = torch.zeros(n, n)
    for (u, v), link in net.links.items():
        if u in idx and v in idx:
            w = getattr(link, "available_bw", link.bandwidth_capacity) / link.bandwidth_capacity
            A[idx[u], idx[v]] = w
            A[idx[v], idx[u]] = w

    A = A + torch.eye(n)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return X, A_norm


def extract_vnode_features(vn: VirtualNetwork) -> torch.Tensor:
    """
    Per-vnode features: [cpu_demand_norm, degree_norm, adj_bw_norm, req_size_norm, req_links_norm]
    """
    vnodes = list(vn.nodes.values())
    n = len(vnodes)
    feats = torch.zeros(n, 5)

    degrees: Dict[str, int] = {nd.id: 0 for nd in vnodes}
    adj_bw: Dict[str, float] = {nd.id: 0.0 for nd in vnodes}
    for vlink in vn.links.values():
        degrees[vlink.source] = degrees.get(vlink.source, 0) + 1
        degrees[vlink.target] = degrees.get(vlink.target, 0) + 1
        adj_bw[vlink.source] = adj_bw.get(vlink.source, 0.0) + vlink.bandwidth_demand
        adj_bw[vlink.target] = adj_bw.get(vlink.target, 0.0) + vlink.bandwidth_demand

    max_cpu = max((nd.cpu_demand for nd in vnodes), default=1.0) or 1.0
    max_deg = max(degrees.values(), default=1) or 1
    max_bw = max(adj_bw.values(), default=1.0) or 1.0

    for i, nd in enumerate(vnodes):
        feats[i, 0] = nd.cpu_demand / max_cpu
        feats[i, 1] = degrees[nd.id] / max_deg
        feats[i, 2] = adj_bw[nd.id] / max_bw
        feats[i, 3] = len(vn.nodes) / 20.0
        feats[i, 4] = len(vn.links) / 40.0
    return feats


def build_vn_adjacency(vn: VirtualNetwork) -> torch.Tensor:
    """
    Normalized VN adjacency with self-loops, BW-weighted edges.
    """
    node_ids = list(vn.nodes.keys())
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    max_bw = max((vl.bandwidth_demand for vl in vn.links.values()), default=1.0) or 1.0

    A = torch.zeros(n, n)
    for vl in vn.links.values():
        if vl.source in idx and vl.target in idx:
            w = vl.bandwidth_demand / max_bw
            A[idx[vl.source], idx[vl.target]] = w
            A[idx[vl.target], idx[vl.source]] = w

    A = A + torch.eye(n)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
    return D_inv_sqrt @ A @ D_inv_sqrt
