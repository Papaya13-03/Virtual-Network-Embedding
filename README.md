# Virtual Network Embedding (VNE)

Research project comparing VNE algorithms: heuristic (MP-VNE, OA-MP-VNE, MC-VNM) and reinforcement learning (RL-OA-MP-VNE, RL-Cand-VNE).

## Layout

```
algorithms/     Algorithm implementations (one subdir per algorithm)
problem/        Problem model: substrate, virtual network, requests, solutions
configs/        YAML configs per algorithm
datasets/       Scenario substrate + virtual request JSON files (+ generator)
evaluation/     Result comparison and plotting utilities
scripts/        Runnable entry points (training, dataset generation, plots)
tests/          pytest suite
docs/           Research notes, summaries, diagrams, design specs
utils/          Dataset / solution I/O and visualization helpers
```

## Run

```bash
python main.py --algorithm mp_vne \
  --substrate datasets/scenario_1/substrate.json \
  --requests datasets/scenario_1/virtual_requests.json \
  --output results/scenario_1/solutions.json
```

## Tests

```bash
pytest tests/
```
