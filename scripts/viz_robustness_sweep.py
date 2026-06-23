"""Robustness sweep on the 100-node test set: CARL-VNE (CF e90) vs MP-VNE
across 5 VNR-distribution axes, each at 5 points.

For EACH axis, produce a 2x2 figure with 4 metrics (acceptance / avg cost /
rev-cost / delay), MP-VNE vs CARL-VNE. Also a single acceptance overview.

Reads results/scenario_100nodes_<set>/{mp_vne,carl_vne_cf_e90}/metrics.json.
Missing points (eval not done yet) are skipped, not errored.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# Scale-specific config (filled in from CLI in main()).
NODES = 100
CARL_NAME = "carl_vne_cf_e90"
CARL_LABEL = "CARL-VNE CF e90"

# axis -> ([(tick_label, set_suffix, x_value), ...], xlabel, log_x)
AXES = {
    "Lifetime (VNR)": (
        [("125", "life_short", 125), ("250", "life_250", 250), ("500", "center", 500),
         ("1000", "life_1000", 1000), ("2000", "life_long", 2000)],
        "Lifetime trung bình", True),
    "VNR size (nodes)": (
        [("2-3", "size_small", 2.5), ("3-5", "size_3_5", 4.0), ("3-7", "center", 5.0),
         ("6-9", "size_6_9", 7.5), ("8-12", "size_large", 10.0)],
        "Số node / VNR (trung bình)", False),
    "VNR density": (
        [("0.0", "dens_sparse", 0.0), ("0.15", "dens_015", 0.15), ("0.3", "center", 0.3),
         ("0.55", "dens_055", 0.55), ("0.8", "dens_dense", 0.8)],
        "edge_prob (mật độ link)", False),
    "Resource demand": (
        [("0.5x", "res_low", 0.5), ("0.75x", "res_075", 0.75), ("1x", "center", 1.0),
         ("1.5x", "res_150", 1.5), ("2x", "res_high", 2.0)],
        "Hệ số tài nguyên yêu cầu", False),
    "Region constraint": (
        [("0.2", "region_loose", 0.2), ("0.4", "region_04", 0.4), ("0.6", "center", 0.6),
         ("0.8", "region_08", 0.8), ("1.0", "region_strict", 1.0)],
        "Tỉ lệ vnode ràng buộc miền", False),
}

# (metrics.json key, panel title + arrow, scale)
METRICS = [
    ("acceptance_rate", "Acceptance rate (%)  (↑ tốt hơn)", 100.0),
    ("avg_cost", "Avg cost  (↓ tốt hơn)", 1.0),
    ("revenue_cost_ratio", "Revenue / cost  (↑ tốt hơn)", 1.0),
    ("avg_delay", "Avg delay  (↓ tốt hơn)", 1.0),
]

def algos():
    return [("MP-VNE", "mp_vne", "tab:red", "o--"),
            ("CARL-VNE", CARL_NAME, "tab:purple", "s-")]


def metric(set_suffix, algo, key, scale):
    p = ROOT / f"results/scenario_{NODES}nodes_{set_suffix}/{algo}/metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())[key] * scale


SLUGS = {
    "Lifetime (VNR)": "lifetime",
    "VNR size (nodes)": "size",
    "VNR density": "density",
    "Resource demand": "resource",
    "Region constraint": "region",
}


def slug(axis_title):
    return SLUGS[axis_title]


def main():
    plt.rcParams.update({"axes.titleweight": "bold", "axes.labelweight": "bold",
                         "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
    saved = []

    # --- Per-axis 4-metric figures ---
    for title, (points, xlabel, logx) in AXES.items():
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (key, ptitle, scale) in zip(axes.flat, METRICS):
            for alabel, aname, color, style in algos():
                xs, ys = [], []
                for _, suf, xv in points:
                    v = metric(suf, aname, key, scale)
                    if v is not None:
                        xs.append(xv); ys.append(v)
                if xs:
                    ax.plot(xs, ys, style, color=color, linewidth=2.2,
                            markersize=9, label=alabel)
            if logx:
                ax.set_xscale("log")
                ax.set_xticks([p[2] for p in points])
                ax.set_xticklabels([p[0] for p in points])
            ax.set_title(ptitle, fontsize=16)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=13)
        fig.suptitle(f"Robustness {NODES}-node — trục: {title}  ({CARL_LABEL} vs MP-VNE)",
                     fontsize=18, fontweight="bold")
        fig.tight_layout()
        out = OUT / f"{NODES}nodes_robust_{slug(title)}_{DATE_TAG}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)

    # --- Acceptance overview (all 5 axes in one 2x3) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, (points, xlabel, logx)) in zip(axes.flat, AXES.items()):
        for alabel, aname, color, style in algos():
            xs, ys = [], []
            for _, suf, xv in points:
                v = metric(suf, aname, "acceptance_rate", 100.0)
                if v is not None:
                    xs.append(xv); ys.append(v)
            if xs:
                ax.plot(xs, ys, style, color=color, linewidth=2.2, markersize=9, label=alabel)
        if logx:
            ax.set_xscale("log"); ax.set_xticks([p[2] for p in points])
            ax.set_xticklabels([p[0] for p in points])
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel); ax.set_ylabel("Acceptance (%)")
        ax.grid(alpha=0.3); ax.legend(fontsize=13)
    axes.flat[5].axis("off")
    fig.suptitle(f"Robustness {NODES}-node — Acceptance overview (5 trục)",
                 fontsize=18, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"{NODES}nodes_robust_overview_{DATE_TAG}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)

    for s in saved:
        print(f"Saved: {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=100, choices=[50, 100])
    ap.add_argument("--carl-name", default=None,
                    help="results subdir for CARL (default: carl_vne_cf_e90 for 100n, carl_vne_cf_e195 for 50n)")
    ap.add_argument("--carl-label", default=None, help="legend/title label for CARL")
    a = ap.parse_args()
    NODES = a.nodes
    CARL_NAME = a.carl_name or ("carl_vne_cf_e195" if NODES == 50 else "carl_vne_cf_e90")
    CARL_LABEL = a.carl_label or (f"CARL-VNE CF e{'195' if NODES == 50 else '90'}")
    main()
