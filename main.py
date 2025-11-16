from src.algorithms.MP_VNE.mp_vne import MP_VNE
from src.utils.load_dataset_from_json import load_dataset_from_json

# ----------------- 1. Load dataset -----------------
dataset = load_dataset_from_json("./datasets/data_2.json")
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

print("Number of request: ", len(virtual_requests))

while pending_requests:
    # 4a. Kiểm tra request mới đến
    new_arrivals = [r for r in pending_requests if r["arrival_time"] <= current_time]

    for req in new_arrivals:
        print(f"[t={current_time}] Mapping virtual network (arrival_time={req['arrival_time']})...")
        request_id = mp_vne.handle_mapping_request(req, current_time)
        print(f"[t={current_time}] Mapping done. Request ID = {request_id}")
        pending_requests.remove(req)

    # 4b. Giải phóng các request hết lifetime
    mp_vne.release_expired_requests(current_time)

    current_time += time_step
