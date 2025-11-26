import sys
import time
import matplotlib.pyplot as plt
import os
import json
import copy

from src.algorithms.MC_VNM.mc_vnm import MC_VNM
from src.algorithms.MP_VNE.mp_vne import MP_VNE
from src.utils.load_dataset_from_json import load_dataset_from_json

# ================================
#       FILE JSON KẾT QUẢ
# ================================
os.makedirs("./assets/result", exist_ok=True)
json_path = "./assets/result/simulation_data.json"

# Load cũ nếu tồn tại, nếu không tạo mảng mới
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        old_data = json.load(f)
else:
    old_data = []

# ================================
#       CHẠY 50 DATASETS
# ================================
for i in range(1, 101):
    print(f"\n================= RUN DATASET {i} =================")
    dataset_file = f"./datasets/small_{i}.json"

    dataset = load_dataset_from_json(dataset_file)

    snetwork_mp = copy.deepcopy(dataset["substrate_network"])
    snetwork_mc = copy.deepcopy(dataset["substrate_network"])

    virtual_requests = dataset["virtual_requests"]
    virtual_requests.sort(key=lambda r: r["arrival_time"])

    mp_vne = MP_VNE(snetwork_mp)
    mc_vnm = MC_VNM(snetwork_mc)

    current_time = 0.0
    time_step = 1.0
    pending_requests = virtual_requests.copy()

    # Stats: lưu data dạng time series
    stats = {
        "dataset": dataset_file,
        "MP_VNE": {"accepted": 0, "failed": 0, "times": [], "costs": [], "per_request_time": [], "per_request_cost": [], "success": []},
        "MC_VNM": {"accepted": 0, "failed": 0, "times": [], "costs": [], "per_request_time": [], "per_request_cost": [], "success": []}
    }

    while pending_requests:
        new_arrivals = [r for r in pending_requests if r["arrival_time"] <= current_time]

        for req in new_arrivals:
            # -------------------- MP-VNE --------------------
            try:
                t0 = time.time()
                rid_mp, cost_mp, mapping_mp = mp_vne.handle_mapping_request(req, current_time)
                t1 = time.time()

                stats["MP_VNE"]["accepted"] += 1
                stats["MP_VNE"]["times"].append(t1 - t0)
                stats["MP_VNE"]["costs"].append(cost_mp)
                stats["MP_VNE"]["per_request_time"].append(t1 - t0)
                stats["MP_VNE"]["per_request_cost"].append(cost_mp)
                stats["MP_VNE"]["success"].append(True)

            except Exception as e:
                stats["MP_VNE"]["failed"] += 1
                stats["MP_VNE"]["per_request_time"].append(None)
                stats["MP_VNE"]["per_request_cost"].append(None)
                stats["MP_VNE"]["success"].append(False)

            # -------------------- MC-VNE --------------------
            try:
                t0 = time.time()
                rid_mc, cost_mc, mapping_mc = mc_vnm.handle_mapping_request(req["vnetwork"], current_time, req["lifetime"])
                t1 = time.time()

                stats["MC_VNM"]["accepted"] += 1
                stats["MC_VNM"]["times"].append(t1 - t0)
                stats["MC_VNM"]["costs"].append(cost_mc)
                stats["MC_VNM"]["per_request_time"].append(t1 - t0)
                stats["MC_VNM"]["per_request_cost"].append(cost_mc)
                stats["MC_VNM"]["success"].append(True)

            except Exception as e:
                stats["MC_VNM"]["failed"] += 1
                stats["MC_VNM"]["per_request_time"].append(None)
                stats["MC_VNM"]["per_request_cost"].append(None)
                stats["MC_VNM"]["success"].append(False)

            pending_requests.remove(req)

        # Release expired
        mc_vnm.release_expired_requests(current_time)
        current_time += time_step

    # Append kết quả lần chạy này
    old_data.append(stats)
    print(f"Finished dataset {i}")

# Ghi lại tất cả kết quả
with open(json_path, "w") as f:
    json.dump(old_data, f, indent=4)

print(f"Appended all 50 datasets simulation data to {json_path}")
