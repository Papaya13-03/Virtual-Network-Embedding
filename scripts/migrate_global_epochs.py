#!/usr/bin/env python3
"""One-off migration: 100-node PPO cont* history -> global epoch numbering.

Merges the per-phase CSVs (base, cont..cont5) of the two reward recipes into
single files with GLOBAL epoch numbers, and renames per-epoch checkpoints to
ckpt_e{global}.pt under experiments/carl_vne_100nodes/<recipe>/.

  normal:      77 epochs (base 1-10, cont 11-20, cont2 21-30, cont3 31-50,
                          cont4 51-70, cont5 71-77)
  costfocused: 79 epochs (same phases, cont5 71-79)

Phase-final checkpoints (e.g. ..._cont4.pt) are duplicates of the phase's last
per-epoch checkpoint; they are verified tensor-equal and dropped (kept with a
_phase_final suffix if they ever differ).

Usage:
  uv run python scripts/migrate_global_epochs.py --dry-run   # print actions
  uv run python scripts/migrate_global_epochs.py             # do it
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
CKPTS = ROOT / "checkpoints"
DEST = ROOT / "experiments" / "carl_vne_100nodes"

# suffix -> (offset, expected epoch count per recipe)
PHASES = [
    ("",       0,  {"normal": 10, "costfocused": 10}),
    ("_cont",  10, {"normal": 10, "costfocused": 10}),
    ("_cont2", 20, {"normal": 10, "costfocused": 10}),
    ("_cont3", 30, {"normal": 20, "costfocused": 20}),
    ("_cont4", 50, {"normal": 20, "costfocused": 20}),
    ("_cont5", 70, {"normal": 7,  "costfocused": 9}),
]
TOTALS = {"normal": 77, "costfocused": 79}


def read_data_rows(path):
    """Return (header, rows) skipping blank lines and repeated headers."""
    header, rows = None, []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("epoch,"):
            header = line
            continue
        rows.append(line)
    return header, rows


def shift_epoch(row, offset):
    parts = row.split(",")
    parts[0] = str(int(float(parts[0])) + offset)
    return ",".join(parts)


def tensors_equal(p1, p2):
    a = torch.load(p1, map_location="cpu", weights_only=False)
    b = torch.load(p2, map_location="cpu", weights_only=False)
    sa, sb = a["policy_state_dict"], b["policy_state_dict"]
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def migrate_recipe(recipe, dry):
    stem = f"ppo_v19_100nodes_{recipe}"
    ck_stem = f"il_mp_vne_v19_100nodes_{recipe}"
    out = DEST / recipe
    actions = []

    # ---- 1. merge epoch summaries -> training_epoch_summary.csv ----
    merged_epochs = []
    header = None
    for suffix, offset, expected in PHASES:
        src = LOGS / f"{stem}{suffix}_epoch_summary.csv"
        h, rows = read_data_rows(src)
        header = header or h
        assert len(rows) == expected[recipe], \
            f"{src.name}: {len(rows)} rows, expected {expected[recipe]}"
        merged_epochs += [shift_epoch(r, offset) for r in rows]
    epochs_seen = [int(r.split(",")[0]) for r in merged_epochs]
    assert epochs_seen == list(range(1, TOTALS[recipe] + 1)), \
        f"{recipe}: epochs not contiguous 1..{TOTALS[recipe]}"
    actions.append(f"write {out}/training_epoch_summary.csv "
                   f"({len(merged_epochs)} epochs)")

    # ---- 2. merge per-episode CSVs -> training.csv ----
    merged_eps = []
    ep_header = None
    for suffix, offset, _ in PHASES:
        src = LOGS / f"{stem}{suffix}.csv"
        h, rows = read_data_rows(src)
        ep_header = ep_header or h
        merged_eps += [shift_epoch(r, offset) for r in rows]
    actions.append(f"write {out}/training.csv ({len(merged_eps)} rows)")

    # ---- 3. per-epoch checkpoints -> checkpoints/ckpt_e{global}.pt ----
    ck_moves = {}
    for suffix, offset, expected in PHASES:
        for k in range(1, expected[recipe] + 1):
            src = CKPTS / f"{ck_stem}{suffix}_e{k}.pt"
            assert src.exists(), f"missing {src.name}"
            dst = out / "checkpoints" / f"ckpt_e{offset + k}.pt"
            assert dst.name not in {d.name for d in ck_moves.values()}
            ck_moves[src] = dst
    assert len(ck_moves) == TOTALS[recipe]
    actions.append(f"move {len(ck_moves)} checkpoints -> "
                   f"{out}/checkpoints/ckpt_e1..e{TOTALS[recipe]}.pt")

    # ---- 4. phase-final dedup (no cont5 finals exist: runs interrupted) ----
    finals = []
    for suffix, offset, expected in PHASES:
        final = CKPTS / f"{ck_stem}{suffix}.pt"
        if final.exists():
            last_ep = CKPTS / f"{ck_stem}{suffix}_e{expected[recipe]}.pt"
            finals.append((final, last_ep))
    actions.append(f"dedup-check {len(finals)} phase-final checkpoints")

    # ---- 5. run logs -> run_logs/ ----
    run_logs = [LOGS / f"{stem}{suffix}_run.log" for suffix, _, _ in PHASES
                if (LOGS / f"{stem}{suffix}_run.log").exists()]
    actions.append(f"move {len(run_logs)} run logs -> {out}/run_logs/")

    print(f"\n=== {recipe} ===")
    for a in actions:
        print(f"  {a}")
    if dry:
        return

    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_logs").mkdir(parents=True, exist_ok=True)

    (out / "training_epoch_summary.csv").write_text(
        header + "\n" + "\n".join(merged_epochs) + "\n")
    (out / "training.csv").write_text(
        ep_header + "\n" + "\n".join(merged_eps) + "\n")

    for src, dst in ck_moves.items():
        shutil.copy2(src, dst)
        assert filecmp.cmp(src, dst, shallow=False), f"copy mismatch {dst}"
        src.unlink()

    for final, last_ep_old in finals:
        # last_ep was just moved — find its new location
        new_last = ck_moves.get(last_ep_old)
        if new_last is None:  # already moved this run; reconstruct path
            raise AssertionError(f"last-epoch ckpt for {final.name} not found")
        if tensors_equal(final, new_last):
            final.unlink()
            print(f"  dropped {final.name} (== {new_last.name})")
        else:
            keep = out / "checkpoints" / f"{final.stem}_phase_final.pt"
            shutil.move(str(final), keep)
            print(f"  WARNING: {final.name} differs from {new_last.name}; "
                  f"kept as {keep.name}")

    for rl in run_logs:
        shutil.move(str(rl), out / "run_logs" / rl.name)

    # remove now-merged CSV sources
    for suffix, _, _ in PHASES:
        for f in (LOGS / f"{stem}{suffix}.csv",
                  LOGS / f"{stem}{suffix}_epoch_summary.csv"):
            if f.exists():
                f.unlink()
    print(f"  done: {TOTALS[recipe]} epochs migrated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for recipe in ("normal", "costfocused"):
        migrate_recipe(recipe, args.dry_run)
    if args.dry_run:
        print("\n(dry run — nothing changed)")


if __name__ == "__main__":
    sys.exit(main())
