def print_mapping(mapping_info, request_id):
    """
    Print mapping info in a readable format.
    mapping_info: {
        "mapping": {vnode: snode},
        "vlink_paths": {vlink: [links]},
        "expire_time": float
    }
    """
    print(f"\n=== Mapping for Request {request_id} (expires at t={mapping_info['expire_time']}) ===")
    
    print("\n-- Node Mapping --")
    for vnode, snode in mapping_info["mapping"].items():
        print(f"VNode {vnode.id:>5} -> SNode {snode.node_id:>5} (CPU demand: {vnode.cpu_demand}, CPU available: {snode.available_cpu})")
    
    print("\n-- Link Mapping --")
    for vlink, path in mapping_info.get("vlink_paths", {}).items():
        if not path:
            print(f"VLink {vlink.src.id} -> {vlink.dst.id}: no path")
            continue
        path_str = " -> ".join(f"{link.src.node_id}-{link.dst.node_id}" for link in path)
        print(f"VLink {vlink.src.id} -> {vlink.dst.id}: {path_str} (BW: {vlink.bandwidth})")
    
    print("="*60)
