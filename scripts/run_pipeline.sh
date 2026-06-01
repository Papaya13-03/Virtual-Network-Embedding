#!/usr/bin/env bash
# Master pipeline: pretrain + eval + visualize for all 4 dataset sizes.
#
# Sequential because compute-bound. Total ETA ~10-15h depending on machine.
# Each step writes its own log. Resumable: skips if checkpoint/results already exist.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="uv run python -u"

SIZES=(100 200 500 1000)

log() { echo "[$(date +%H:%M:%S)] $*"; }

for size in "${SIZES[@]}"; do
    SCENARIO="scenario_${size}nodes"
    TRAIN_SCENARIO="${SCENARIO}_train"
    CKPT="checkpoints/il_mp_vne_${size}nodes.pt"

    # ---- Pretrain (skip if checkpoint exists) ----
    if [ -f "$CKPT" ]; then
        log "skip pretrain $size (ckpt exists)"
    else
        log "pretrain $size nodes — start"
        $PYTHON scripts/imitation_pretrain.py \
            --substrate "datasets/${TRAIN_SCENARIO}/substrate.json" \
            --requests "datasets/${TRAIN_SCENARIO}/virtual_requests.json" \
            --episodes 10000 \
            --batch-size 16 \
            --checkpoint "$CKPT" \
            --log-file "logs/imitation_${size}nodes.csv" \
            --print-every 20 \
            --seed 42 \
            > "logs/imitation_${size}nodes_run.log" 2>&1
        log "pretrain $size — done"
    fi

    # ---- Eval mp_vne ----
    OUT_MP="results/${SCENARIO}/mp_vne"
    if [ -f "$OUT_MP/metrics.json" ]; then
        log "skip eval mp_vne $size"
    else
        log "eval mp_vne $size — start"
        $PYTHON scripts/run_eval.py \
            --algorithm mp_vne \
            --substrate "datasets/${SCENARIO}/substrate.json" \
            --requests "datasets/${SCENARIO}/virtual_requests.json" \
            --output "$OUT_MP" \
            > "logs/eval_mp_vne_${size}nodes.log" 2>&1
        log "eval mp_vne $size — done"
    fi

    # ---- Eval il_mp_vne_pso (proposed) ----
    OUT_IL="results/${SCENARIO}/il_mp_vne_pso"
    if [ -f "$OUT_IL/metrics.json" ]; then
        log "skip eval il_mp_vne_pso $size"
    else
        log "eval il_mp_vne_pso $size — start"
        $PYTHON scripts/run_eval.py \
            --algorithm il_mp_vne_pso \
            --substrate "datasets/${SCENARIO}/substrate.json" \
            --requests "datasets/${SCENARIO}/virtual_requests.json" \
            --checkpoint "$CKPT" \
            --output "$OUT_IL" \
            > "logs/eval_il_mp_vne_pso_${size}nodes.log" 2>&1
        log "eval il_mp_vne_pso $size — done"
    fi
done

# ---- Visualize ----
log "visualize"
$PYTHON scripts/visualize_metrics.py \
    --output results/comparison.png

log "ALL DONE"
