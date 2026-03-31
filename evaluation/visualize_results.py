import json
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import glob

def get_network_info(substrate_path):
    with open(substrate_path, 'r') as f:
        substrate = json.load(f)
    node_prices = {}
    link_delays = {}
    
    for domain in substrate["domains"]:
        for node in domain["nodes"]:
            node_prices[node["id"]] = node.get("cpu_price", 1.0)
        # Handle links within domain
        # The structure might be a list or a dict depending on the generator
        links = domain.get("links", [])
        if isinstance(links, list):
            for link in links:
                u, v = link["source"], link["target"]
                link_delays[(u, v)] = link.get("transmission_delay", 1.0)
                link_delays[(v, u)] = link.get("transmission_delay", 1.0)
        elif isinstance(links, dict):
             for key, link in links.items():
                u, v = link["source"], link["target"]
                link_delays[(u, v)] = link.get("transmission_delay", 1.0)
                link_delays[(v, u)] = link.get("transmission_delay", 1.0)

    inter_domain = substrate.get("inter_domain_links", [])
    if isinstance(inter_domain, list):
        for link in inter_domain:
            u, v = link["source"], link["target"]
            link_delays[(u, v)] = link.get("transmission_delay", 5.0)
            link_delays[(v, u)] = link.get("transmission_delay", 5.0)
    elif isinstance(inter_domain, dict):
        for key, link in inter_domain.items():
            u, v = link["source"], link["target"]
            link_delays[(u, v)] = link.get("transmission_delay", 5.0)
            link_delays[(v, u)] = link.get("transmission_delay", 5.0)

    return node_prices, link_delays

def calculate_metrics_for_run(requests, solutions, node_prices, link_delays):
    solutions_dict = {s["vnr_id"]: s for s in solutions}
    requests = sorted(requests, key=lambda x: x["arrival_time"])
    
    successful_count = 0
    cumulative_rev = 0
    cumulative_cost = 0
    total_inst_cost = 0
    total_inst_delay = 0
    
    metrics = {
        "rac": [], "lar": [], "r2c": [],
        "avg_cost": [], "avg_delay": [], "success_count": [],
        "time": []
    }
    
    for i, req in enumerate(requests):
        vnr_id = req["id"]
        sol = solutions_dict.get(vnr_id)
        is_success = sol and sol.get("is_successful", False)
        
        inst_cost = 0
        inst_delay = 0
        
        if is_success:
            successful_count += 1
            rev = sum(n["cpu_demand"] for n in req["virtual_network"]["nodes"])
            rev += sum(l["bandwidth_demand"] for l in req["virtual_network"]["links"])
            
            # Node mapping cost
            for vnode_id, snode_id in sol["node_mapping"].items():
                vnode = next(n for n in req["virtual_network"]["nodes"] if n["id"] == vnode_id)
                inst_cost += vnode["cpu_demand"] * node_prices.get(snode_id, 1.0)
                
            # Link mapping cost & delay
            vlink_count = len(sol["link_mapping"])
            vlink_delays = []
            for vlink_id, paths in sol["link_mapping"].items():
                for path_info in paths:
                    path = path_info["path"]
                    bw = path_info["allocated_bandwidth"]
                    inst_cost += bw * len(path)
                    
                    # Calculate delay for this path
                    path_delay = 0
                    for link_str in path:
                        u, v = link_str.split("->")
                        path_delay += link_delays.get((u, v), 1.0)
                    vlink_delays.append(path_delay)
            
            if vlink_delays:
                inst_delay = sum(vlink_delays) / len(vlink_delays)
            
            duration = req["lifetime"]
            cumulative_rev += rev * duration
            cumulative_cost += inst_cost * duration
            total_inst_cost += inst_cost
            total_inst_delay += inst_delay
            
        current_time = req["arrival_time"]
        metrics["rac"].append(successful_count / (i + 1))
        metrics["lar"].append(cumulative_rev / current_time if current_time > 0 else 0)
        metrics["r2c"].append(cumulative_rev / cumulative_cost if cumulative_cost > 0 else 0)
        metrics["avg_cost"].append(total_inst_cost / successful_count if successful_count > 0 else 0)
        metrics["avg_delay"].append(total_inst_delay / successful_count if successful_count > 0 else 0)
        metrics["success_count"].append(successful_count)
        metrics["time"].append(current_time)
        
    return metrics

def bin_metrics(time_pts, metrics_dict, bin_size=1000):
    if len(time_pts) == 0: return [], {k: [] for k in metrics_dict}
    
    max_t = time_pts[-1]
    bins = np.arange(0, max_t + bin_size, bin_size)
    binned_time = bins[1:] # We use the end of the bin as the time point
    binned_metrics = {k: [] for k in metrics_dict}
    
    for i in range(len(bins)-1):
        t_start, t_end = bins[i], bins[i+1]
        mask = (time_pts >= t_start) & (time_pts < t_end)
        
        for k, vals in metrics_dict.items():
            bin_vals = np.array(vals)[mask]
            if len(bin_vals) > 0:
                binned_metrics[k].append(np.mean(bin_vals))
            else:
                # If bin is empty, use the last value before this bin (if exists)
                prev_mask = (time_pts < t_start)
                if any(prev_mask):
                    binned_metrics[k].append(np.array(vals)[prev_mask][-1])
                else:
                    binned_metrics[k].append(0)
                    
    return binned_time, binned_metrics

def main():
    parser = argparse.ArgumentParser(description="Visualize VNE results by averaging multiple runs and comparing algorithms.")
    parser.add_argument("--scenario", type=str, default="test_1", help="Scenario name")
    parser.add_argument("--project_root", type=str, default=".", help="Project root directory")
    args = parser.parse_args()
    
    scenario_dir = os.path.join(args.project_root, "datasets", args.scenario)
    results_dir = os.path.join(args.project_root, "results", args.scenario)
    
    substrate_path = os.path.join(scenario_dir, "substrate.json")
    requests_path = os.path.join(scenario_dir, "virtual_requests.json")
    
    if not os.path.exists(requests_path):
        print(f"Error: Dataset files not found in {scenario_dir}")
        return

    with open(requests_path, 'r') as f:
        requests = json.load(f)
    node_prices, link_delays = get_network_info(substrate_path)
    
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
    if not run_dirs:
        print(f"No run directories found in {results_dir}")
        return

    # Detect all algorithms
    algos = set()
    for run_dir in run_dirs:
        sol_files = glob.glob(os.path.join(run_dir, "solutions_*.json"))
        for f in sol_files:
            algo_name = os.path.basename(f).replace("solutions_", "").replace(".json", "")
            algos.add(algo_name)
    
    algos = sorted(list(algos))
    algos = ['mp_vne', 'oa_mp_vne']
    print(f"Detected algorithms: {algos}")
    
    # Plotting setup
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    metric_keys = ["rac", "lar", "r2c", "avg_cost", "avg_delay", "success_count"]
    titles = [
        "RAC (Acceptance Rate)", "LAR (Avg Revenue)", "LT-R2C (Rev/Cost)",
        "Average Embedding Cost", "Average Path Delay", "Total Success Count"
    ]
    ylabels = ["RAC", "LAR", "R2C", "Cost", "Delay", "Successes"]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    
    for algo_idx, algo_name in enumerate(algos):
        algo_metrics_history = {k: [] for k in metric_keys}
        time_pts = None
        
        print(f"Analyzing {algo_name}...")
        for run_dir in run_dirs:
            sol_path = os.path.join(run_dir, f"solutions_{algo_name}.json")
            if not os.path.exists(sol_path):
                continue
            with open(sol_path, 'r') as f:
                solutions = json.load(f)
            
            run_metrics = calculate_metrics_for_run(requests, solutions, node_prices, link_delays)
            # Re-structure for binning
            raw_metrics = {k: run_metrics[k] for k in metric_keys}
            b_time, b_metrics = bin_metrics(np.array(run_metrics["time"]), raw_metrics, bin_size=1000)
            
            for k in metric_keys:
                algo_metrics_history[k].append(b_metrics[k])
            if time_pts is None:
                time_pts = b_time
        
        if not algo_metrics_history["rac"]: continue
        
        for ax_idx, k in enumerate(metric_keys):
            ax = axes[ax_idx]
            color = colors[algo_idx]
            data = np.array(algo_metrics_history[k])
            avg = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            ax.plot(time_pts, avg, label=algo_name, color=color, linewidth=2)
            if len(data) > 1:
                ax.fill_between(time_pts, avg - std, avg + std, color=color, alpha=0.15)
 
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    output_path = os.path.join(results_dir, "algorithm_comparison_plots.png")
    plt.savefig(output_path)
    print(f"Comparison plots saved to {output_path}")

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    output_path = os.path.join(results_dir, "algorithm_comparison_plots.png")
    plt.savefig(output_path)
    print(f"Comparison plots saved to {output_path}")

if __name__ == "__main__":
    main()
