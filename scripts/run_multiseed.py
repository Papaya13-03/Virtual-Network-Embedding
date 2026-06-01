"""Run an algorithm with N seeds, save per-seed outputs, and write a
metrics_avg.json summarizing mean ± std across seeds.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True)
    p.add_argument("--substrate", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-base", required=True,
                   help="Base output dir; per-seed runs go into <base>/seed_<n>/.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    base = Path(args.output_base)
    base.mkdir(parents=True, exist_ok=True)

    per_seed_metrics = []
    for seed in args.seeds:
        seed_dir = base / f"seed_{seed}"
        cmd = [
            "uv", "run", "python", "-u", "scripts/run_eval.py",
            "--algorithm", args.algorithm,
            "--substrate", args.substrate,
            "--requests", args.requests,
            "--output", str(seed_dir),
            "--seed", str(seed),
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        print(f"\n=== running seed={seed} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        with open(seed_dir / "metrics.json") as f:
            per_seed_metrics.append(json.load(f))

    # Aggregate
    keys = ["acceptance_rate", "avg_cost", "revenue_rate", "revenue_cost_ratio",
            "avg_delay", "n_success", "elapsed_seconds"]
    summary = {"algorithm": args.algorithm, "seeds": args.seeds, "per_seed": per_seed_metrics}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics]
        summary[f"{k}_mean"] = statistics.mean(vals)
        summary[f"{k}_std"]  = statistics.stdev(vals) if len(vals) > 1 else 0.0

    with open(base / "metrics_avg.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== aggregate ({len(args.seeds)} seeds) → {base}/metrics_avg.json ===")
    for k in keys:
        m = summary[f"{k}_mean"]; s = summary[f"{k}_std"]
        print(f"  {k:<22}  {m:>12.4f}  ± {s:>8.4f}")


if __name__ == "__main__":
    main()
