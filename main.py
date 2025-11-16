"""
Main entry point for VNE project.

This script runs and compares different VNE algorithms on test cases.
"""

import os
import glob
from typing import List, Dict, Any
from src.utils.test_generator import VNETestGenerator, handle_generate_test_case
from src.utils.algorithm_evaluator import AlgorithmEvaluator
from src.algorithms.mp_vne import MP_VNE
from src.algorithms.base import BaseVNEAlgorithm


def main():
    """Main function to run algorithm comparison."""
    import os
    import glob
    
    # Find the latest test case file
    testcase_files = glob.glob("./datasets/testcase-*.json")
    if testcase_files:
        # Get the latest file
        latest_file = max(testcase_files, key=os.path.getctime)
        print(f"Loading test case from {latest_file}...")
        physical_network, requests = VNETestGenerator.load_test_case(latest_file)
        
        print("\n=== Test Case Info ===")
        VNETestGenerator.print_test_case_summary(physical_network, requests)
        
        # Create algorithm evaluator
        evaluator = AlgorithmEvaluator(physical_network)
        
        # Define algorithms to compare
        algorithms = [
            MP_VNE(physical_network),
            # Add more algorithms here in the future
            # Example: GreedyVNE(physical_network),
            # Example: GeneticVNE(physical_network),
        ]
        
        algorithm_names = [
            "MP_VNE",
            # Add more algorithm names here
        ]
        
        algorithm_params = [
            {'max_iterations': 50, 'population_size': 30},
            # Add more algorithm parameters here
        ]
        
        # Run comparison
        results = evaluator.compare_algorithms(
            algorithms,
            algorithm_names,
            requests,
            algorithm_params,
            verbose=True
        )
        
        # Print comparison
        AlgorithmEvaluator.print_comparison_table(results)
        
    else:
        print("No test case files found. Generating a new one...")
        handle_generate_test_case()
        print("\nPlease run the script again to compare algorithms on the generated test case.")


if __name__ == "__main__":
    main()

