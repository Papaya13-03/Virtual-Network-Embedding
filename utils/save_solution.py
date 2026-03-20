import json
import os
from typing import List, Dict, Any
from problem.embedding_solution import EmbeddingSolution

def _serialize_solution(solution: EmbeddingSolution) -> Dict[str, Any]:
    """Helper to convert an EmbeddingSolution dataclass to a JSON-serializable dict."""
    
    # Serialize link mapping to standard strings/lists
    # For MP-VNE: links might be mapped to multiple split paths with corresponding bandwidth capacities
    # E.g. Dict[Tuple[str, str], List[Tuple[List[Tuple[str, str]], float]]]
    serialized_link_mapping = {}
    for (v_src, v_dst), paths in solution.link_mapping.items():
        key = f"{v_src}->{v_dst}"
        serialized_paths = []
        for item in paths:
            # Check if this is a MultiPath split tuple: (path_links, allocated_bandwidth)
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list):
                path_links, bw = item
                formatted_path = [f"{s}->{d}" for s, d in path_links]
                serialized_paths.append({
                    "path": formatted_path,
                    "allocated_bandwidth": bw
                })
            else:
                # Single path tuple compatibility
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    s, d = item
                    serialized_paths.append(f"{s}->{d}")
                else:
                    serialized_paths.append(str(item))
                
        serialized_link_mapping[key] = serialized_paths

    return {
        "vnr_id": solution.vnr_id,
        "is_successful": solution.is_successful,
        "node_mapping": solution.node_mapping,
        "link_mapping": serialized_link_mapping,
        "embedding_cost": round(solution.embedding_cost, 4) if solution.embedding_cost else 0.0
    }

def write_solution(solution: EmbeddingSolution, filepath: str):
    """Writes a single EmbeddingSolution to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(_serialize_solution(solution), f, indent=2)

def write_solutions(solutions: List[EmbeddingSolution], filepath: str):
    """Writes a list of EmbeddingSolutions to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    serialized_list = [_serialize_solution(sol) for sol in solutions]
    with open(filepath, 'w') as f:
        json.dump(serialized_list, f, indent=2)
