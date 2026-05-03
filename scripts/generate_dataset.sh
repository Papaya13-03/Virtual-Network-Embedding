#!/bin/bash

# Utility script to quickly generate a dataset
# Usage: ./scripts/generate_dataset.sh <scenario_name> [domains] [nodes_per_domain] [requests] [config_path]

SCENARIO=${1:-"test_1"}
DOMAINS=${2:-5}
NODES=${3:-6}
REQUESTS=${4:-1000}

PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
CONFIG_PATH=${5:-"$PROJECT_ROOT/configs/dataset.yaml"}
GENERATOR_SCRIPT="$PROJECT_ROOT/scripts/generate_dataset.py"

echo "Running multi-domain VNE dataset generator for $SCENARIO..."
python3 "$GENERATOR_SCRIPT" --scenario "$SCENARIO" --config "$CONFIG_PATH" --domains "$DOMAINS" --nodes_per_domain "$NODES" --requests "$REQUESTS"
