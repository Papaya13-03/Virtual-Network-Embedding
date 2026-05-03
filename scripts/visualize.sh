#!/bin/bash

# Utility script to visualize VNE evaluation results.
# Usage: ./evaluation/visualize.sh [scenario_name] [project_root]

SCENARIO=${1:-"scenario_large"}
PROJECT_ROOT=${2:-"."}

PYTHON_SCRIPT="$PROJECT_ROOT/evaluation/visualize_results.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Visualization script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "Visualizing results for scenario: $SCENARIO..."
python3 "$PYTHON_SCRIPT" --scenario "$SCENARIO" --project_root "$PROJECT_ROOT"

if [ $? -eq 0 ]; then
    echo "Visualization completed successfully."
    echo "Output: results/$SCENARIO/average_evaluation_plots.png"
else
    echo "Error: Visualization failed."
    exit 1
fi
