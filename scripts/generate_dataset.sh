#!/bin/bash

# Utility script to quickly generate a dataset
# Usage: ./scripts/generate_dataset.sh <scenario_name> [domains] [nodes_per_domain] [requests] [config_path]

SCENARIO=${1:-"scenario_default"}
DOMAINS=${2:-3}
NODES=${3:-50}
REQUESTS=${4:-100}

PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
CONFIG_PATH=${5:-"$PROJECT_ROOT/configs/dataset.yaml"}
GENERATOR_SCRIPT="$PROJECT_ROOT/datasets/generator/generate_dataset.py"

echo "Running multi-domain VNE dataset generator for $SCENARIO..."
python3 "$GENERATOR_SCRIPT" --scenario "$SCENARIO" --config "$CONFIG_PATH" --domains "$DOMAINS" --nodes_per_domain "$NODES" --requests "$REQUESTS"
