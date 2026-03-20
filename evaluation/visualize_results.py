import json
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import glob

def get_node_prices(substrate_path):
    with open(substrate_path, 'r') as f:
        substrate = json.load(f)
    node_prices = {}
    for domain in substrate["domains"]:
        for node in domain["nodes"]:
            node_prices[node["id"]] = node.get("cpu_price", 1.0)
    return node_prices

def calculate_metrics_for_run(requests, solutions, node_prices):
    solutions_dict = {s["vnr_id"]: s for s in solutions}
    requests = sorted(requests, key=lambda x: x["arrival_time"])
    
    successful_count = 0
    cumulative_rev = 0
    cumulative_cost = 0
    
    rac_list = []
    lar_list = []
    r2c_list = []
    time_pts = []
    
    for i, req in enumerate(requests):
        vnr_id = req["id"]
        sol = solutions_dict.get(vnr_id)
        is_success = sol and sol.get("is_successful", False)
        
        if is_success:
            successful_count += 1
            rev = sum(n["cpu_demand"] for n in req["virtual_network"]["nodes"])
            rev += sum(l["bandwidth_demand"] for l in req["virtual_network"]["links"])
            
            cost = 0
            # Node mapping cost
            for vnode_id, snode_id in sol["node_mapping"].items():
                vnode = next(n for n in req["virtual_network"]["nodes"] if n["id"] == vnode_id)
                cost += vnode["cpu_demand"] * node_prices.get(snode_id, 1.0)
                
            # Link mapping cost
            for vlink_id, paths in sol["link_mapping"].items():
                # paths is a list of {"path": ["s1->s2", ...], "allocated_bandwidth": 10.0}
                for path_info in paths:
                    # Some paths might be just strings if it was an older format, but we unified them.
                    path_len = len(path_info["path"]) if isinstance(path_info, dict) else 1
                    bw = path_info["allocated_bandwidth"] if isinstance(path_info, dict) else 0.0
                    cost += bw * path_len
            
            duration = req["lifetime"]
            cumulative_rev += rev * duration
            cumulative_cost += cost * duration
            
        current_time = req["arrival_time"]
        rac = successful_count / (i + 1)
        lar = cumulative_rev / current_time if current_time > 0 else 0
        r2c = cumulative_rev / cumulative_cost if cumulative_cost > 0 else 0
        
        rac_list.append(rac)
        lar_list.append(lar)
        r2c_list.append(r2c)
        time_pts.append(current_time)
        
    return np.array(time_pts), np.array(rac_list), np.array(lar_list), np.array(r2c_list)

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
    node_prices = get_node_prices(substrate_path)
    
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
    print(f"Detected algorithms: {algos}")
    
    # Plotting setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["RAC (Request Acceptance Rate)", "LAR (Long-term Average Revenue)", "LT-R2C (Revenue-to-Cost ratio)"]
    ylabels = ["RAC", "LAR", "R2C"]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    
    for algo_idx, algo_name in enumerate(algos):
        all_rac = []
        all_lar = []
        all_r2c = []
        time_pts = None
        
        print(f"Analyzing {algo_name}...")
        for run_dir in run_dirs:
            sol_path = os.path.join(run_dir, f"solutions_{algo_name}.json")
            if not os.path.exists(sol_path):
                continue
            with open(sol_path, 'r') as f:
                solutions = json.load(f)
            
            t, rac, lar, r2c = calculate_metrics_for_run(requests, solutions, node_prices)
            all_rac.append(rac)
            all_lar.append(lar)
            all_r2c.append(r2c)
            if time_pts is None:
                time_pts = t
        
        if not all_rac: continue
        
        # Averages
        avg_metrics = [
            np.mean(all_rac, axis=0),
            np.mean(all_lar, axis=0),
            np.mean(all_r2c, axis=0)
        ]
        std_metrics = [
            np.std(all_rac, axis=0),
            np.std(all_lar, axis=0),
            np.std(all_r2c, axis=0)
        ]
        
        for ax_idx, (avg, std) in enumerate(zip(avg_metrics, std_metrics)):
            ax = axes[ax_idx]
            color = colors[algo_idx]
            ax.plot(time_pts, avg, label=algo_name, color=color, linewidth=2)
            if len(all_rac) > 1:
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

if __name__ == "__main__":
    main()
