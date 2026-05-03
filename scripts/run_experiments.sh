#!/bin/bash

# Utility script to run multiple VNE algorithms for comparison against a specific scenario multiple times.
# Usage: ./scripts/run_experiments.sh [scenario_name] [number_of_runs] [limit]

SCENARIO=${1:-"scenario_stress"}
RUNS=${2:-1}
LIMIT=${3:-""}

PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# Array of algorithms to run (add more as they are implemented)
ALGORITHMS=("rl_oa_mp_vne")

SUBSTRATE_PATH="$PROJECT_ROOT/datasets/$SCENARIO/substrate.json"
REQUESTS_PATH="$PROJECT_ROOT/datasets/$SCENARIO/virtual_requests.json"
OUTPUT_DIR_BASE="$PROJECT_ROOT/results/$SCENARIO"

echo "================================================================"
echo "Running VNE Experiments for scenario: $SCENARIO"
echo "Total Runs per Algorithm: $RUNS"
if [ -n "$LIMIT" ]; then
    echo "Limit set to $LIMIT requests per algorithm inside each run."
fi
echo "================================================================"

for ALGO in "${ALGORITHMS[@]}"; do
    echo ""
    echo ">>> Starting evaluation for algorithm: $ALGO <<<"
    
    for (( i=1; i<=RUNS; i++ )); do
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/run_${i}"
        mkdir -p "$OUTPUT_DIR"
        
        OUTPUT_PATH="${OUTPUT_DIR}/solutions_${ALGO}.json"
        
        # Force python3 to ensure torch compatibility
        PYTHON_CMD="python3"
        
        CMD="$PYTHON_CMD $PROJECT_ROOT/main.py --algorithm $ALGO --substrate $SUBSTRATE_PATH --requests $REQUESTS_PATH --output $OUTPUT_PATH"
        
        if [ -n "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi
        
        echo "  --> [Run $i/$RUNS] Executing: $CMD"
        eval "$CMD"
        
        if [ $? -eq 0 ]; then
            echo "  --> [Run $i/$RUNS] Successfully completed $ALGO."
            echo "  --> [Run $i/$RUNS] Results saved to: $OUTPUT_PATH"
        else
            echo "  --> [Run $i/$RUNS] Error: $ALGO failed to execute."
        fi
    done
done

echo ""
echo "================================================================"
echo "All experiments finished."
echo "Results are divided into separate 'run_X' directories inside results/$SCENARIO/"
echo "You can average the outputs using these generated files."
echo "================================================================"
