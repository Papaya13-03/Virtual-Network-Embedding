# Tóm tắt Project: Virtual Network Embedding dựa trên MP-VNE

> **Paper tham chiếu:** *"A Multi-Domain VNE Algorithm Based on Multi-Objective Optimization for IoD Architecture in Industry 4.0"* — Peiying Zhang, Chao Wang, Zeyu Qin, Haotong Cao (2022, arXiv: 2202.12830)

---

## 1. Giới thiệu (Introduction)

Ảo hóa mạng (Network Virtualization) cho phép nhiều **Virtual Network (VN)** cùng tồn tại trên một **Substrate Network (SN)** vật lý dùng chung. Bài toán **Virtual Network Embedding (VNE)** đặt ra yêu cầu: ánh xạ các yêu cầu mạng ảo đến từ người dùng lên hạ tầng vật lý sao cho vừa tôn trọng ràng buộc tài nguyên, vừa tối ưu chi phí và độ trễ.

Trong kiến trúc **IoD (Internet of Drones)** phục vụ Industry 4.0, hạ tầng vật lý bị phân tán thành nhiều miền (multi-domain) do giới hạn địa lý và quyền quản trị. Điều này khiến VNE trở thành bài toán tối ưu đa mục tiêu (chi phí + độ trễ) trên đồ thị nhiều miền, phức tạp hơn đáng kể so với VNE đơn miền.

Project này **tái hiện và đánh giá** thuật toán **MP-VNE** từ paper trên, đồng thời xây dựng một framework chuẩn hóa để thử nghiệm và so sánh các phương pháp khác nhau.

---

## 2. Tìm hiểu bài toán (Problem Formulation)

Bài toán VNE đa miền trong paper được mô hình hóa như sau và được tái hiện trong thư mục `problem/` của project:

- **Substrate Network** `G^s = (G_i^s, L^s)`: đồ thị vô hướng gồm nhiều miền con, mỗi node có **CPU, đơn giá, processing delay**; mỗi link có **bandwidth, đơn giá, transmission delay**.
- **Virtual Network Request**: đồ thị ảo tới theo phân phối Poisson, thời gian sống theo phân phối Exponential. Mỗi node/link ảo có yêu cầu tài nguyên tương ứng.
- **Mục tiêu**: tối thiểu hóa hàm chi phí tổng hợp (weighted sum của chi phí tài nguyên và độ trễ) — chuyển bài toán đa mục tiêu về đơn mục tiêu.
- **Ràng buộc**: tổng yêu cầu ánh xạ tại mỗi node/link vật lý ≤ dung lượng còn lại.

Data model của project được cài đặt trong:
- `problem/substrate_network.py`, `problem/domain.py`
- `problem/virtual_network.py`, `problem/request.py`
- `problem/embedding_solution.py`

---

## 3. Đề xuất (Proposed Approach)

Trên cơ sở bài toán và mô hình của paper, project đề xuất:

- **Tái hiện đầy đủ MP-VNE** như đường cơ sở (baseline) để kiểm chứng tính đúng đắn của framework và các chỉ số đánh giá.
- **Xây dựng một framework mở**, trong đó thuật toán được xem như một module có thể thay thế, tuân theo interface thống nhất `solve(substrate_network, virtual_request)`.
- **Chuẩn hóa quy trình thực nghiệm** (dataset, metric, trực quan hóa) để mọi phương pháp được đánh giá trên cùng điều kiện, đảm bảo so sánh công bằng và tái sản sinh được.

Các phương pháp cải tiến sẽ được phát triển và công bố trong phiên bản sau khi đã ổn định.

---

## 4. Triển khai (Implementation)

### 4.1. Kiến trúc mã nguồn

```
Virtual-Network-Embedding/
├── main.py              # CLI chạy 1 thuật toán trên 1 kịch bản
├── problem/             # Data model VNE
├── algorithms/          # Các thuật toán VNE
│   └── registry.py      # Ánh xạ tên -> class
├── configs/             # Hyperparameters YAML cho từng thuật toán
├── constraints/         # Ràng buộc CPU / BW
├── datasets/            # Kịch bản SN + VNR
├── evaluation/          # So sánh & trực quan hóa
├── experiments/         # Script thí nghiệm
├── results/             # Output JSON
├── scripts/             # Shell utilities
├── tests/               # pytest (unit + integration)
└── docs/                # Paper summary, slide deck
```

### 4.2. Interface chuẩn hóa

Tất cả thuật toán đều tuân theo một interface chung, cho phép dễ dàng thay thế và so sánh:

```python
solution = algorithm.solve(substrate_network, virtual_request)
```

Việc bổ sung một phương pháp mới chỉ cần implement `solve()` rồi đăng ký vào `algorithms/registry.py`, không cần sửa đổi phần đánh giá.

### 4.3. Tech stack

- Python ≥ 3.9, quản lý phụ thuộc bằng `hatchling` / `uv`
- `numpy`, `pyyaml`, `matplotlib`
- `torch` (cho các biến thể có học máy)
- `pytest` (tests trong `tests/`)

### 4.4. Cấu hình

Mỗi thuật toán có file YAML riêng trong `configs/`. Các tham số có fallback mặc định trong code để đảm bảo tính tái sản sinh.

---

## 5. Thực nghiệm (Experiments)

### 5.1. Sinh dataset

```bash
./scripts/generate_dataset.sh
```

Kịch bản mặc định tuân theo thiết lập của paper:
- **Physical Network**: 4 miền × 30 node, mỗi miền có 2 boundary node, xác suất kết nối 50%
- **Node**: CPU ∈ [100, 300], price ∈ [1, 10]
- **Link**: BW ∈ [1000, 3000], price ∈ [1, 10]
- **VNR**: 6 node/request, CPU demand ∈ [1, 10], BW demand ∈ [1, 10]
- **Arrival**: Poisson (trung bình 10 VNR / 100 time units)
- **Lifetime**: Exponential (trung bình 1000 time units)

Dataset sinh ra được lưu trong `datasets/scenario_*/`.

### 5.2. Chạy thí nghiệm

**Một thuật toán, một kịch bản:**

```bash
python main.py \
  --algorithm mp_vne \
  --substrate datasets/scenario_1/substrate.json \
  --requests  datasets/scenario_1/virtual_requests.json \
  --output    results/scenario_1/solutions.json
```

**Benchmark nhiều thuật toán × nhiều lần chạy:**

```bash
./scripts/run_experiments.sh scenario_1 5 200
# 5 runs mỗi thuật toán, giới hạn 200 VNR mỗi run
```

Kết quả mỗi run được ghi vào `results/<scenario>/run_<i>/solutions_<algo>.json`.

### 5.3. Chỉ số đánh giá

Theo đúng các metric trong paper:

| Metric | Ý nghĩa |
|--------|---------|
| **Acceptance Rate** | Tỉ lệ VNR được embed thành công |
| **Average Mapping Cost** | Chi phí tài nguyên trung bình của các VNR thành công |
| **Mapping Delay** | Tổng processing delay + transmission delay |
| **Composite Cost** | Weighted sum — chính là hàm mục tiêu |

### 5.4. Trực quan hóa

```bash
./scripts/visualize.sh
```

Các biểu đồ được sinh bằng `evaluation/visualize_results.py` và `evaluation/compare_two.py`. Ví dụ đầu ra: `algorithm_comparison_plots.png` ở thư mục gốc.

---

## 6. Kết quả tham khảo (Reported Results)

Theo paper, MP-VNE vượt trội đồng thời trên cả ba chỉ số so với các baseline (MC-VNM, VNE-PSO, LID-VNE):

- **Acceptance rate** ổn định ~60% (baseline rớt về ~30%)
- **Average cost** thấp nhất (~650–750 so với 1150–1400 của baseline)
- **Mapping delay** thấp nhất (~460 so với 600–800)
- **Composite cost** dưới 650 (baseline từ 950 đến hơn 1250)

Kết quả thực nghiệm trên framework của project được ghi lại trong `results/` sau mỗi lần chạy và có thể so sánh trực tiếp với kết quả công bố.

---

## 7. Đóng góp của project (Contributions)

1. **Tái hiện MP-VNE** trong codebase Python có cấu trúc module rõ ràng, tách biệt Global/Local controllers đúng như thiết kế trong paper.
2. **Framework benchmark tái sản sinh được**: dataset generator, multi-run script, bộ chỉ số đánh giá đồng nhất — thay thế cho các thí nghiệm ad-hoc.
3. **Interface chuẩn hóa** cho phép thêm thuật toán mới chỉ bằng `solve()` + đăng ký vào `registry.py`.
4. **Hệ thống kiểm thử và trực quan hóa** tích hợp (`tests/`, `evaluation/`), hỗ trợ đánh giá nhanh khi có phương pháp mới.

---

## 8. Tài liệu tham khảo

- Paper gốc: `docs/2202.12830v1.pdf`
- Tóm tắt chi tiết thuật toán: `docs/MP_VNE_Summary.md`
- Slide/presentation: `docs/VNE_Presentation.pptx`, `docs/Slide_Deck.md`
- Các paper liên quan:
  - `docs/FlagVNE_Summary.md`
  - `docs/Swarm_PSO_RL_Summary.md`
  - `docs/Virne_Benchmark_Summary.md`
