import random
from typing import Dict

from algorithms.oa_mp_vne.global_controller import GlobalController
from algorithms.rl_cand_vne.vn_generator import generate_random_vn_with_domains
from algorithms.oa_mp_vne.oa_mp_vne import OAMPVNE
from problem.domain import MultiDomainNetwork
from problem.request import VirtualNetworkRequest


def fractional_drop(gc: GlobalController, u_max_cpu: float, u_max_bw: float) -> None:
    """
    Reset allocations then randomly reduce available resources per snode/slink.
    """
    gc.reset_allocations()
    gc.clear_caches()
    for lc in gc.local_controllers:
        for node in lc.domain.network.nodes.values():
            u = random.uniform(0.0, u_max_cpu)
            node.available_cpu = node.cpu_capacity * (1.0 - u)
        for link in lc.domain.network.links.values():
            u = random.uniform(0.0, u_max_bw)
            link.available_bw = link.bandwidth_capacity * (1.0 - u)


def warmup_embed(
    gc: GlobalController,
    md_network: MultiDomainNetwork,
    M_max: int,
    vn_kwargs: Dict,
) -> None:
    """
    Reset allocations then embed up to M random VNs greedily via OA-MP-VNE
    to create a realistic loaded state. Failed embeddings are discarded.
    Allocations remain in place after this call (caller rolls them back later).
    """
    gc.reset_allocations()
    gc.clear_caches()
    M = random.randint(0, M_max)
    if M == 0:
        return
    helper = OAMPVNE()
    domain_ids = [lc.domain.id for lc in gc.local_controllers]
    for i in range(M):
        vn = generate_random_vn_with_domains(
            min_nodes=vn_kwargs["min_nodes"], max_nodes=vn_kwargs["max_nodes"],
            min_cpu=vn_kwargs["min_cpu"], max_cpu=vn_kwargs["max_cpu"],
            min_bw=vn_kwargs["min_bw"], max_bw=vn_kwargs["max_bw"],
            link_prob=vn_kwargs["link_prob"],
            domain_ids=domain_ids,
            p_all=1.0, p_single=0.0, p_subset=0.0,
            subset_min=2, subset_max=3,
        )
        req = VirtualNetworkRequest(
            id=f"warmup_{i}",
            virtual_network=vn,
            arrival_time=0.0,
            lifetime=float("inf"),
        )
        try:
            helper.solve(md_network, req)
        except Exception:
            continue


def sample_substrate_state(
    gc: GlobalController,
    md_network: MultiDomainNetwork,
    warmup_fraction: float,
    u_max_cpu: float,
    u_max_bw: float,
    M_max: int,
    vn_kwargs: Dict,
) -> str:
    """
    Dispatcher: return the mode used ('fractional_drop' or 'warmup_embed').
    """
    if random.random() < warmup_fraction:
        warmup_embed(gc, md_network, M_max=M_max, vn_kwargs=vn_kwargs)
        return "warmup_embed"
    fractional_drop(gc, u_max_cpu=u_max_cpu, u_max_bw=u_max_bw)
    return "fractional_drop"
