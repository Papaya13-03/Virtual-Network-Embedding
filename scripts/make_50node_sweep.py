#!/usr/bin/env python3
"""Build the 50-node robustness-sweep datasets from the existing 100-node ones.

VNR distribution is independent of substrate size, and both substrates use the
same 10 domains, so the 100-node VNR sets are reused verbatim against the
50-node substrate. Only arrival_time is doubled (Poisson 0.1 -> 0.05) so the
load-per-capacity matches: the 50-node substrate has half the capacity, so
halving the arrival rate keeps utilization identical to the 100-node sweep.

Result: each 50-node axis point has the EXACT same VNR-property distribution
(size / cpu / bw / lifetime / region / links) as its 100-node counterpart.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSTRATE_SRC = ROOT / "datasets" / "scenario_50nodes" / "substrate.json"
ARRIVAL_SCALE = 2.0  # arrival_rate 0.1 -> 0.05

SETS = [
    "center",
    "life_short", "life_250", "life_1000", "life_long",
    "size_small", "size_3_5", "size_6_9", "size_large",
    "dens_sparse", "dens_015", "dens_055", "dens_dense",
    "res_low", "res_075", "res_150", "res_high",
    "region_loose", "region_04", "region_08", "region_strict",
]


def build(set_name: str):
    src_req = ROOT / f"datasets/scenario_100nodes_{set_name}/virtual_requests.json"
    if not src_req.exists():
        print(f"  SKIP {set_name}: no 100-node source")
        return
    dst = ROOT / f"datasets/scenario_50nodes_{set_name}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(SUBSTRATE_SRC, dst / "substrate.json")

    reqs = json.loads(src_req.read_text())
    for r in reqs:
        r["arrival_time"] = round(r["arrival_time"] * ARRIVAL_SCALE, 2)
    (dst / "virtual_requests.json").write_text(json.dumps(reqs, indent=2))
    print(f"  OK   {set_name}: {len(reqs)} VNRs -> {dst.name}")


def main():
    print(f"Building 50-node sweep (substrate={SUBSTRATE_SRC.parent.name}, arrival x{ARRIVAL_SCALE})")
    for s in SETS:
        build(s)
    print("done")


if __name__ == "__main__":
    main()
