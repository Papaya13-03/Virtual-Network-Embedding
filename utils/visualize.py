import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Set global matplotlib style for "paper-style" publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def visualize(x_arrays: List[List[float]], y_arrays: List[List[float]], labels: List[str], 
              title: str, xlabel: str, ylabel: str, save_path: Optional[str] = None, 
              figsize: Tuple[int, int] = (8, 5)):
    """
    Renders a publication-ready line plot for the provided metrics.
    """
    plt.figure(figsize=figsize)
    
    # Pre-defined nice colors and markers ensuring distinguishability
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']

    for i, (x, y, label) in enumerate(zip(x_arrays, y_arrays, labels)):
        if not x or not y:
            continue
        plt.plot(x, y, 
                 marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], 
                 label=label, 
                 alpha=0.85)

    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    
    # Add a nice shadow legend
    plt.legend(loc="best", frameon=True, shadow=True, fancybox=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Saved high-quality plot to: {save_path}")
    
    plt.show()


def cumulative_acceptance_rate(records: List[tuple], step: float = 100.0) -> Tuple[List[float], List[float]]:
    """Calculates the rolling acceptance rate up to each time step."""
    xs, ys = [], []
    total = 0
    accepted = 0
    next_t = step

    for arrival_time, success, _, _ in records:
        total += 1
        if success:
            accepted += 1

        if arrival_time >= next_t:
            xs.append(next_t)
            ys.append(accepted / total)
            next_t += step

    return xs, ys


def cumulative_avg_metric(records: List[tuple], metric_idx: int, step: float = 100.0) -> Tuple[List[float], List[float]]:
    """
    Calculates the rolling average of a specific metric up to each time step.
    metric_idx: 2 for Time, 3 for Cost
    """
    xs, ys = [], []
    acc_sum = 0.0
    acc_cnt = 0
    next_t = step

    for record in records:
        arrival_time = record[0]
        success = record[1]
        val = record[metric_idx]
        
        if success and val is not None:
            acc_sum += val
            acc_cnt += 1

        if arrival_time >= next_t and acc_cnt > 0:
            xs.append(next_t)
            ys.append(acc_sum / acc_cnt)
            next_t += step

    return xs, ys


def plot_simulation_metrics(all_algorithms_records: Dict[str, List[tuple]], output_dir: str, step: float = 100.0):
    """
    Aggregates logic for Acceptance Rate, Mapping Time, and Cost plots.
    all_algorithms_records maps -> "AlgorithmName": [(arrival_time, success, time, cost), ...]
    """
    algorithm_names = list(all_algorithms_records.keys())
    
    # 1. Acceptance Rate Plot
    x_arrays, y_arrays = [], []
    for alg in algorithm_names:
        x, y = cumulative_acceptance_rate(all_algorithms_records[alg], step=step)
        x_arrays.append(x)
        y_arrays.append(y)
        
    visualize(
        x_arrays, y_arrays,
        labels=algorithm_names,
        title="Virtual Network Request Acceptance Rate",
        xlabel="Simulation Time (time units)",
        ylabel="Acceptance Rate",
        save_path=os.path.join(output_dir, "acceptance_rate.png")
    )
    
    # 2. Average Mapping Time Plot
    x_arrays, y_arrays = [], []
    for alg in algorithm_names:
        x, y = cumulative_avg_metric(all_algorithms_records[alg], metric_idx=2, step=step)
        x_arrays.append(x)
        y_arrays.append(y)
        
    visualize(
        x_arrays, y_arrays,
        labels=algorithm_names,
        title="Average Mapping Time over Simulation Time",
        xlabel="Simulation Time (time units)",
        ylabel="Average Time (s)",
        save_path=os.path.join(output_dir, "avg_mapping_time.png")
    )
    
    # 3. Average Embedding Cost Plot
    x_arrays, y_arrays = [], []
    for alg in algorithm_names:
        x, y = cumulative_avg_metric(all_algorithms_records[alg], metric_idx=3, step=step)
        x_arrays.append(x)
        y_arrays.append(y)
        
    visualize(
        x_arrays, y_arrays,
        labels=algorithm_names,
        title="Average Cost over Simulation Time",
        xlabel="Simulation Time (time units)",
        ylabel="Average Cost",
        save_path=os.path.join(output_dir, "avg_cost.png")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VNE Solver Results")
    parser.add_argument("--json_path", type=str, required=True, help="Path to aggregated JSON results")
    parser.add_argument("--output_dir", type=str, default="./results/figures", help="Directory to save generated plots")
    parser.add_argument("--step", type=float, default=100.0, help="Time step for cumulative aggregation")
    args = parser.parse_args()
    
    import json
    if not os.path.exists(args.json_path):
        print(f"Error: Could not find data file at {args.json_path}")
        exit(1)
        
    with open(args.json_path, "r") as f:
        data = json.load(f)
        
    # Expected format for visualization script:
    # {"MP_VNE": [[arrival_1, success_boolean, execution_time, cost], ...], "DU_VNE": ...}
    plot_simulation_metrics(data, args.output_dir, step=args.step)
