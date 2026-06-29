#!/bin/bash
# Robustness sweep eval: 11 test sets (center + 5 axes x 2) x {MP-VNE, CARL-VNE CF e90}.
# Max 3 concurrent. Skips a job whose metrics.json already exists (resumable).
set -u
cd "$(dirname "$0")/.."

CKPT=experiments/carl_vne_100nodes/costfocused/checkpoints/ckpt_e90.pt
SETS="center life_short life_250 life_long life_1000 size_small size_3_5 size_6_9 size_large dens_sparse dens_015 dens_055 dens_dense res_low res_075 res_150 res_high region_loose region_04 region_08 region_strict"
LOGDIR=experiments/carl_vne_100nodes/kb_eval_logs
mkdir -p "$LOGDIR"
MAXJOBS=3

run_eval() {
  local set=$1 algo=$2 name=$3 ckptflag=$4
  local sub=datasets/scenario_100nodes_${set}/substrate.json
  local req=datasets/scenario_100nodes_${set}/virtual_requests.json
  local out=results/scenario_100nodes_${set}/${name}
  if [ -f "${out}/metrics.json" ]; then echo "skip ${set}/${name} (already done)"; return; fi
  echo "start ${set}/${name}"
  uv run python scripts/run_eval.py --algorithm "$algo" \
    --substrate "$sub" --requests "$req" $ckptflag \
    --output "$out" --seed 42 > "${LOGDIR}/${set}_${name}.log" 2>&1
  echo "done ${set}/${name}"
}

for set in $SETS; do
  run_eval "$set" mp_vne mp_vne "" &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
  run_eval "$set" carl_vne_pso carl_vne_cf_e90 "--checkpoint $CKPT" &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
done
wait
echo "ALL KB EVALS DONE"
