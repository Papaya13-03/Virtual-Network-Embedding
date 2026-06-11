# Virtual Network Embedding (VNE)

Research project for the thesis method **CARL-VNE** (Candidate-RL VNE: IL
pretrain + PPO fine-tuned candidate head) against the heuristic baselines
**MP-VNE** and **MP-VNE-V4**.

## Layout

```
algorithms/       carl_vne (proposed method, self-contained), mp_vne, mp_vne_v4
problem/          Problem model: substrate, virtual network, requests, solutions
configs/          YAML configs (il_mp_vne_v6.yaml = CARL-VNE base net, mp_vne.yaml)
datasets/         Scenario substrate + VNR JSON files (50/100/200 nodes, + generator)
experiments/      Training artifacts, one dir per experiment:
  carl_vne_50nodes/    normal/, costfocused/, pporeal/ (logs, checkpoints, eval logs)
  carl_vne_100nodes/   normal/, costfocused/ — GLOBAL epoch numbering:
                       training_epoch_summary.csv + checkpoints/ckpt_e{N}.pt
  carl_vne_200nodes/   200-node scaling experiment
  baselines/           mp_vne, mp_vne_v4 eval logs
  pretrain/            IL-pretrain checkpoints (incl. thesis R2: il_mp_vne_v6_100nodes_r2.pt)
  figures/             Generated charts
results/          Eval outputs per scenario (solutions.json + metrics.json)
scripts/          Entry points: ppo_finetune.py, run_eval.py, generate_dataset.py, viz_*
tests/            pytest suite
thesis/           LaTeX thesis
docs/             Research notes, experiment scenarios (kich_ban_thuc_nghiem.md)
utils/            Dataset / solution I/O helpers
```

## Train (PPO fine-tune, global epoch numbering)

Continuation runs append to the same CSV and auto-continue the epoch count
(e.g. 77 epochs trained → next run starts at epoch 78; or pass `--start-epoch`):

```bash
python scripts/ppo_finetune.py --algorithm carl_vne \
  --substrate datasets/scenario_100nodes_train/substrate.json \
  --requests datasets/scenario_100nodes_train/virtual_requests.json \
  --ref-checkpoint experiments/carl_vne_100nodes/normal/checkpoints/ckpt_e77.pt \
  --rollout direct --target cand --epochs 20 --episodes 5000 \
  --success-bonus 1.0 --cost-lambda 0.3 --fail-reward -1.0 \
  --checkpoint experiments/carl_vne_100nodes/normal/checkpoints/ckpt.pt \
  --log-file experiments/carl_vne_100nodes/normal/training.csv
```

## Evaluate

```bash
python scripts/run_eval.py --algorithm carl_vne_pso \
  --substrate datasets/scenario_100nodes/substrate.json \
  --requests datasets/scenario_100nodes/virtual_requests.json \
  --checkpoint experiments/carl_vne_100nodes/costfocused/checkpoints/ckpt_e79.pt \
  --output results/scenario_100nodes/carl_vne_eval --seed 42
```

## Tests

```bash
pytest tests/
```
