import argparse
import os

from utils.load_dataset import read_substrate, read_virtual_requests
from utils.save_solution import write_solutions
from algorithms.registry import get_algorithm

def run_experiment(algorithm_name, substrate_path, requests_path, output_path, limit=None):
    print(f"Loading datasets...")
    print(f"  Substrate: {substrate_path}")
    print(f"  Requests:  {requests_path}")
    
    substrate_network = read_substrate(substrate_path)
    virtual_requests = read_virtual_requests(requests_path)
    
    if limit is not None and limit > 0:
        virtual_requests = virtual_requests[:limit]
        print(f"Limited to {limit} virtual network requests. (Total available: {len(read_virtual_requests(requests_path))})")
    else:
        print(f"Loaded {len(virtual_requests)} virtual network requests.")
    
    print(f"Initializing algorithm: {algorithm_name}")
    algorithm = get_algorithm(algorithm_name)
    
    solutions = []
    
    print("Running algorithm...")
    for i, request in enumerate(virtual_requests):
        print(f"  Processing request {i+1}/{len(virtual_requests)} (ID: {request.id})...")
        try:
            solution = algorithm.solve(substrate_network, request)
            solutions.append(solution)
            status = "Success" if solution.is_successful else "Failed"
            print(f"    -> Result: {status}, Cost: {solution.embedding_cost:.4f}")
        except Exception as e:
            print(f"    -> Error processing request {request.id}: {e}")
            
    print(f"Completed processing. Successfully calculated {sum(1 for s in solutions if s.is_successful)} out of {len(solutions)} requests.")
    
    print(f"Saving solutions to {output_path}...")
    write_solutions(solutions, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Virtual Network Embedding (VNE) Runner")
    parser.add_argument("--algorithm", type=str, default="mp_vne", help="Algorithm to run (default: mp_vne)")
    parser.add_argument("--substrate", type=str, default="datasets/scenario_default/substrate.json", help="Path to substrate network JSON file")
    parser.add_argument("--requests", type=str, default="datasets/scenario_default/virtual_requests.json", help="Path to virtual requests JSON file")
    parser.add_argument("--output", type=str, default="results/scenario_default/solutions.json", help="Path to output solutions JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of requests to process")
    
    args = parser.parse_args()
    
    run_experiment(
        algorithm_name=args.algorithm,
        substrate_path=args.substrate,
        requests_path=args.requests,
        output_path=args.output,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
