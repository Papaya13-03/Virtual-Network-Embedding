import json
import os
from typing import List, Dict, Any, Tuple
from problem.embedding_solution import EmbeddingSolution

def _deserialize_solution(data: Dict[str, Any]) -> EmbeddingSolution:
    """Helper to convert a JSON-serialized dictionary back to an EmbeddingSolution dataclass."""
    
    solution = EmbeddingSolution(
        vnr_id=data.get("vnr_id", ""),
        is_successful=data.get("is_successful", False),
        node_mapping=data.get("node_mapping", {}),
        embedding_cost=data.get("embedding_cost", 0.0)
    )
    
    deserialized_link_mapping = {}
    
    for link_key, paths in data.get("link_mapping", {}).items():
        # Reconstruct the string key 'v1->v2' back into a Tuple[str, str]
        if "->" in link_key:
            src, dst = link_key.split("->", 1)
            vlink_tuple = (src, dst)
        else:
            vlink_tuple = eval(link_key) if "(" in link_key else (link_key, "") # Fallback
            
        deserialized_paths = []
        for path_obj in paths:
            if isinstance(path_obj, dict) and "path" in path_obj and "allocated_bandwidth" in path_obj:
                # Reconstruct Multi-Path format: List[Tuple[List[Tuple[str, str]], float]]
                path_links = []
                for link_str in path_obj["path"]:
                    if "->" in link_str:
                        s, d = link_str.split("->", 1)
                        path_links.append((s, d))
                deserialized_paths.append((path_links, path_obj["allocated_bandwidth"]))
            elif isinstance(path_obj, str) and "->" in path_obj:
                # Reconstruct Single-Path format into List[Tuple[str, str]]
                s, d = path_obj.split("->", 1)
                deserialized_paths.append((s, d))
            else:
                deserialized_paths.append(path_obj)
                
        deserialized_link_mapping[vlink_tuple] = deserialized_paths
        
    solution.link_mapping = deserialized_link_mapping
    return solution


def read_solution(filepath: str) -> EmbeddingSolution:
    """Reads a single EmbeddingSolution from a JSON file. Useful for visualization."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Solution file not found: {filepath}")
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    return _deserialize_solution(data)


def read_solutions(filepath: str) -> List[EmbeddingSolution]:
    """Reads a list of EmbeddingSolutions from a JSON file. Useful for batch visualization."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Solutions file not found: {filepath}")
        
    with open(filepath, 'r') as f:
        data_list = json.load(f)
        
    return [_deserialize_solution(data) for data in data_list]
