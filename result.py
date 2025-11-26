import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ================================
#      HÀM VISUALIZE CHUNG
# ================================
def visualize(x_arrays, y_arrays, labels=None, title="", xlabel="", ylabel="", save_path=None, figsize=(8,5)):
    plt.figure(figsize=figsize)
    for i, (x, y) in enumerate(zip(x_arrays, y_arrays)):
        label = labels[i] if labels else None
        plt.plot(x, y, marker='o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

# ================================
#      ĐỌC DỮ LIỆU JSON
# ================================
json_path = "./assets/result/simulation_data.json"
algorithm_names = ["MP_VNE", "MC_VNM"]  # Có thể thêm thuật toán khác sau này

with open(json_path, "r") as f:
    all_runs = json.load(f)  # list of dicts, mỗi dict là 1 lần chạy
print(all_runs[0].keys())
# ================================
#      HÀM TÍNH TRUNG BÌNH
# ================================

def average_time_series(all_runs, alg_key, field):
    series_list = []

    for i, run in enumerate(all_runs):
        if alg_key not in run:
            print(f"[WARN] Run {i} missing algorithm {alg_key}, skipped")
            continue
        if field not in run[alg_key]:
            print(f"[WARN] Run {i} missing field {field} in {alg_key}, skipped")
            continue

        series = run[alg_key][field]

        # Replace None with np.nan
        series = [v if v is not None else np.nan for v in series]
        series_list.append(series)

    if not series_list:
        raise ValueError(f"No valid data found for {alg_key}.{field}")

    # Pad tất cả series thành cùng độ dài bằng np.nan
    max_len = max(len(s) for s in series_list)
    padded = [s + [np.nan] * (max_len - len(s)) for s in series_list]

    arr = np.array(padded, dtype=float)

    # Tính mean theo cột, bỏ qua nan
    avg = np.nanmean(arr, axis=0)

    return avg.tolist()


# ================================
#      TÍNH TRUNG BÌNH CHO TẤT CẢ
# ================================
avg_per_request_time = {}
avg_per_request_cost = {}

for alg in algorithm_names:
    avg_per_request_time[alg] = average_time_series(all_runs, alg, "per_request_time")
    avg_per_request_cost[alg] = average_time_series(all_runs, alg, "per_request_cost")

# ================================
#      VẼ BIỂU ĐỒ
# ================================
x_arrays = []
y_arrays_time = []
y_arrays_cost = []

for alg in algorithm_names:
    x_arrays.append(list(range(1, len(avg_per_request_time[alg]) + 1)))
    y_arrays_time.append(avg_per_request_time[alg])
    y_arrays_cost.append(avg_per_request_cost[alg])

# --- Average Time per Request ---
visualize(
    x_arrays=x_arrays,
    y_arrays=y_arrays_time,
    labels=algorithm_names,
    title="Average Mapping Time per Request",
    xlabel="Request Index",
    ylabel="Time (s)",
    save_path="./assets/result/avg_time_per_request.png"
)

# --- Average Cost per Request ---
visualize(
    x_arrays=x_arrays,
    y_arrays=y_arrays_cost,
    labels=algorithm_names,
    title="Average Cost per Request",
    xlabel="Request Index",
    ylabel="Cost",
    save_path="./assets/result/avg_cost_per_request.png"
)
