from src.algorithms.MP_VNE.mp_vne import MP_VNE
from src.utils.load_dataset_from_json import load_dataset_from_json
import time

from src.utils.print_mapping import print_mapping

# ----------------- 1. Load dataset -----------------
dataset = load_dataset_from_json("./datasets/small_2.json")
snetwork = dataset["substrate_network"]
virtual_requests = dataset["virtual_requests"]

# ----------------- 2. Sắp xếp theo arrival_time -----------------
virtual_requests.sort(key=lambda r: r["arrival_time"])

# ----------------- 3. Khởi tạo MP_VNE -----------------
mp_vne = MP_VNE(snetwork)

# ----------------- 4. Mô phỏng theo thời gian -----------------
current_time = 0.0
time_step = 1.0  # đơn vị thời gian mô phỏng, có thể điều chỉnh

pending_requests = virtual_requests.copy()
active_requests = []

# Thống kê
total_requests = len(virtual_requests)
accepted_requests = 0
failed_requests = 0
mapping_times = []
mapping_costs = []

print(f"[t={current_time}] Mapping virtual network (arrival_time={pending_requests[0]['arrival_time']})...")

start_time = time.time()
request_id, cost, mapping = mp_vne.handle_mapping_request(pending_requests[0], current_time)
end_time = time.time()

print_mapping(mapping, 1)

print(f"[t={current_time}] Mapping done. Request ID = {request_id}, cost = {cost:.2f}, time = {end_time - start_time:.3f}s")

accepted_requests += 1
mapping_times.append(end_time - start_time)
mapping_costs.append(cost)

# ----------------- 5. Thống kê kết quả -----------------
print(accepted_requests, total_requests)
acceptance_rate = accepted_requests / total_requests
avg_mapping_time = sum(mapping_times)/len(mapping_times) if mapping_times else 0
avg_mapping_cost = sum(mapping_costs)/len(mapping_costs) if mapping_costs else 0

print("\n=== Simulation Results ===")
print(f"Total requests: {total_requests}")
print(f"Accepted requests: {accepted_requests}")
print(f"Failed requests: {failed_requests}")
print(f"Acceptance rate: {acceptance_rate*100:.2f}%")
print(f"Average mapping time: {avg_mapping_time:.3f}s")
print(f"Average mapping cost: {avg_mapping_cost:.2f}")
