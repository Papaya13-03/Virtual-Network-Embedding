"""Robustness sweep on the 100-node test set: CARL-VNE (CF e90) vs MP-VNE
across 5 VNR-distribution axes, each at low / center / high.

Reads results/scenario_100nodes_<set>/{mp_vne,carl_vne_cf_e90}/metrics.json.
Figure: 2x3 grid — 5 axis panels (acceptance vs axis value) + 1 delta summary.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# axis -> [(label, set_suffix, x_value), ...] low/center/high, + axis title & xlabel
AXES = {
    "Lifetime (VNR)": (
        [("short", "life_short", 128), ("center", "center", 512), ("long", "life_long", 2050)],
        "Lifetime trung bình", True),
    "VNR size (nodes)": (
        [("small", "size_small", 2.5), ("center", "center", 5.0), ("large", "size_large", 10.0)],
        "Số node / VNR", False),
    "VNR density": (
        [("sparse", "dens_sparse", 0.78), ("center", "center", 1.15), ("dense", "dens_dense", 1.76)],
        "Số link / node", False),
    "Resource demand": (
        [("low", "res_low", 2.2), ("center", "center", 4.5), ("high", "res_high", 8.9)],
        "CPU demand trung bình", False),
    "Region constraint": (
        [("loose", "region_loose", 0.2), ("center", "center", 0.6), ("strict", "region_strict", 1.0)],
        "Tỉ lệ vnode ràng buộc miền", False),
}


def acc(set_suffix, algo):
    p = ROOT / f"results/scenario_100nodes_{set_suffix}/{algo}/metrics.json"
    return json.loads(p.read_text())["acceptance_rate"] * 100


def main():
    plt.rcParams.update({"axes.titleweight": "bold", "axes.labelweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.flat

    deltas = []  # (axis_label, point_label, delta) for summary
    for ax, (title, (points, xlabel, logx)) in zip(flat, AXES.items()):
        xs = [p[2] for p in points]
        mp = [acc(p[1], "mp_vne") for p in points]
        cf = [acc(p[1], "carl_vne_cf_e90") for p in points]
        ax.plot(xs, mp, "o--", color="tab:red", linewidth=2, markersize=8, label="MP-VNE")
        ax.plot(xs, cf, "s-", color="tab:purple", linewidth=2.4, markersize=8, label="CARL-VNE")
        for x, a, b in zip(xs, mp, cf):
            ax.annotate(f"{b:.1f}", (x, b), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="purple")
        for p, a, b in zip(points, mp, cf):
            deltas.append((title, p[0], b - a))
        if logx:
            ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel); ax.set_ylabel("Acceptance rate (%)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 6th panel: delta (CARL − MP) at every point.
    ax = flat[5]
    labels = [f"{t.split()[0][:4]}:{pl}" for t, pl, _ in deltas]
    vals = [d for _, _, d in deltas]
    colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title("CARL-VNE − MP-VNE (điểm %)", fontsize=12)
    ax.set_xlabel("Chênh lệch acceptance (điểm)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Robustness 100-node: CARL-VNE (CF e90) vs MP-VNE qua 5 trục đặc tính VNR",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"100nodes_robustness_sweep_{DATE_TAG}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
