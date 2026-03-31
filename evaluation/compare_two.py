"""Compare mp_vne vs oa_mp_vne across N runs, averaging metrics."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_network_info(substrate_path):
    with open(substrate_path, 'r') as f:
        substrate = json.load(f)
    node_prices = {}
    link_delays = {}
    for domain in substrate["domains"]:
        for node in domain["nodes"]:
            node_prices[node["id"]] = node.get("cpu_price", 1.0)
        for link in domain.get("links", []):
            u, v = link["source"], link["target"]
            link_delays[(u, v)] = link.get("transmission_delay", 1.0)
            link_delays[(v, u)] = link.get("transmission_delay", 1.0)
    for link in substrate.get("inter_domain_links", []):
        u, v = link["source"], link["target"]
        link_delays[(u, v)] = link.get("transmission_delay", 5.0)
        link_delays[(v, u)] = link.get("transmission_delay", 5.0)
    return node_prices, link_delays

def calc_metrics(requests, solutions, node_prices, link_delays):
    sol_dict = {s["vnr_id"]: s for s in solutions}
    requests = sorted(requests, key=lambda x: x["arrival_time"])

    ok = 0
    cum_rev = cum_cost = tot_cost = tot_delay = 0.0
    metrics = {"rac": [], "lar": [], "r2c": [], "avg_cost": [], "avg_delay": [], "success_count": [], "time": []}

    for i, req in enumerate(requests):
        sol = sol_dict.get(req["id"])
        is_ok = sol and sol.get("is_successful", False)
        if is_ok:
            ok += 1
            rev = sum(n["cpu_demand"] for n in req["virtual_network"]["nodes"])
            rev += sum(l["bandwidth_demand"] for l in req["virtual_network"]["links"])
            ic = 0.0
            for vid, sid in sol["node_mapping"].items():
                vn = next(n for n in req["virtual_network"]["nodes"] if n["id"] == vid)
                ic += vn["cpu_demand"] * node_prices.get(sid, 1.0)
            vl_delays = []
            for _, paths in sol["link_mapping"].items():
                for pi in paths:
                    bw = pi["allocated_bandwidth"]
                    ic += bw * len(pi["path"])
                    pd = sum(link_delays.get(tuple(l.split("->")), 1.0) for l in pi["path"])
                    vl_delays.append(pd)
            inst_delay = np.mean(vl_delays) if vl_delays else 0
            dur = req["lifetime"]
            cum_rev += rev * dur
            cum_cost += ic * dur
            tot_cost += ic
            tot_delay += inst_delay

        t = req["arrival_time"]
        metrics["rac"].append(ok / (i + 1))
        metrics["lar"].append(cum_rev / t if t > 0 else 0)
        metrics["r2c"].append(cum_rev / cum_cost if cum_cost > 0 else 0)
        metrics["avg_cost"].append(tot_cost / ok if ok > 0 else 0)
        metrics["avg_delay"].append(tot_delay / ok if ok > 0 else 0)
        metrics["success_count"].append(ok)
        metrics["time"].append(t)
    return metrics

def bin_metrics(time_pts, mdict, bin_size=1000):
    if not len(time_pts): return [], {k: [] for k in mdict}
    bins = np.arange(0, time_pts[-1] + bin_size, bin_size)
    bt = bins[1:]
    bm = {k: [] for k in mdict}
    for i in range(len(bins) - 1):
        mask = (time_pts >= bins[i]) & (time_pts < bins[i + 1])
        for k, v in mdict.items():
            bv = np.array(v)[mask]
            if len(bv): bm[k].append(np.mean(bv))
            else:
                prev = np.array(v)[time_pts < bins[i]]
                bm[k].append(prev[-1] if len(prev) else 0)
    return bt, bm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenario_1")
    p.add_argument("--project_root", default=".")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--algos", nargs="+", default=["mp_vne", "oa_mp_vne"])
    args = p.parse_args()

    sub_path = os.path.join(args.project_root, "datasets", args.scenario, "substrate.json")
    req_path = os.path.join(args.project_root, "datasets", args.scenario, "virtual_requests.json")
    res_dir = os.path.join(args.project_root, "results", args.scenario)

    with open(req_path) as f:
        requests = json.load(f)
    node_prices, link_delays = get_network_info(sub_path)

    mk = ["rac", "lar", "r2c", "avg_cost", "avg_delay", "success_count"]
    titles = ["RAC (Acceptance Rate)", "LAR (Avg Revenue)", "LT-R2C (Rev/Cost)",
              "Average Embedding Cost", "Average Path Delay", "Total Success Count"]
    ylabels = ["RAC", "LAR", "R2C", "Cost", "Delay", "Successes"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    colors = {"mp_vne": "#1f77b4", "oa_mp_vne": "#ff7f0e"}

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Comparison: {' vs '.join(args.algos)} | {args.runs} runs | {args.scenario}")
    print(f"{'='*70}")

    final_metrics = {}

    for algo in args.algos:
        hist = {k: [] for k in mk}
        tp = None
        for r in range(1, args.runs + 1):
            sp = os.path.join(res_dir, f"run_{r}", f"solutions_{algo}.json")
            if not os.path.exists(sp):
                print(f"  WARN: {sp} not found, skipping")
                continue
            with open(sp) as f:
                sols = json.load(f)
            m = calc_metrics(requests, sols, node_prices, link_delays)
            bt, bm = bin_metrics(np.array(m["time"]), {k: m[k] for k in mk})
            for k in mk:
                hist[k].append(bm[k])
            if tp is None:
                tp = bt

        if not hist["rac"]:
            continue

        # Final values (averaged across runs)
        final_metrics[algo] = {}
        for k in mk:
            data = np.array(hist[k])
            avg = np.mean(data, axis=0)
            final_metrics[algo][k] = avg[-1] if len(avg) else 0

        for ai, k in enumerate(mk):
            data = np.array(hist[k])
            avg = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            c = colors.get(algo, "#333")
            axes[ai].plot(tp, avg, label=algo, color=c, linewidth=2)
            if len(data) > 1:
                axes[ai].fill_between(tp, avg - std, avg + std, color=c, alpha=0.15)

    # Print summary
    print(f"\n{'Metric':<25} {'mp_vne':>15} {'oa_mp_vne':>15} {'Diff':>12}")
    print("-" * 70)
    for k, title in zip(mk, titles):
        v1 = final_metrics.get("mp_vne", {}).get(k, 0)
        v2 = final_metrics.get("oa_mp_vne", {}).get(k, 0)
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        print(f"  {title:<23} {v1:>15.4f} {v2:>15.4f} {pct:>+10.2f}%")
    print(f"{'='*70}\n")

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.suptitle(f"mp_vne vs oa_mp_vne ({args.runs}-run average, {args.scenario})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(res_dir, "mp_vne_vs_oa_mp_vne.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")

if __name__ == "__main__":
    main()
