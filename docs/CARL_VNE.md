# Thuật toán CARL-VNE

CARL-VNE (Candidate-RL VNE) là phương pháp đề xuất cho bài toán nhúng mạng ảo (Virtual Network Embedding) trên hạ tầng đa miền (multi-domain).

Ý tưởng cốt lõi: dùng một mạng policy phân cấp để chấm điểm các node hạ tầng ứng viên, huấn luyện hai giai đoạn (bắt chước chuyên gia rồi tinh chỉnh bằng học tăng cường), và triển khai qua PSO nhiều lần khởi động (multi-restart) để chọn lời giải chi phí thấp nhất.

---

## 1. Bài toán

Cho một hạ tầng vật lý (substrate) gồm nhiều miền (domain), mỗi miền có các node (CPU capacity, cpu_price, processing_delay) và link nội miền (bandwidth, bandwidth_price, transmission_delay); các miền nối với nhau qua inter-domain link. Lần lượt đến các yêu cầu mạng ảo (VNR): mỗi VNR là một đồ thị nhỏ gồm vnode (cpu_demand, có thể bị ràng buộc `allowed_domains`) và vlink (bandwidth_demand), kèm thời điểm đến và lifetime.

Nhiệm vụ: với mỗi VNR, ánh xạ mỗi vnode vào một snode (thỏa CPU và ràng buộc miền) và mỗi vlink vào một đường đi vật lý (thỏa băng thông), tối ưu acceptance rate và chi phí, trong điều kiện tài nguyên bị chiếm dụng theo thời gian.

Giả định đa miền: node của một miền không hiển thị toàn cục. Chỉ có thông tin tổng hợp mức miền và các inter-domain link được trao đổi. Kiến trúc của CARL-VNE tôn trọng giả định này (khác baseline mp_vne vốn xử lý liên miền bằng shortest path hậu kỳ, không có biểu diễn liên miền lúc ra quyết định).

---

## 2. Kiến trúc mạng policy

File: `algorithms/carl_vne/policy_network.py`.

### 2.1. Bộ mã hóa hạ tầng phân cấp (HierarchicalSubstrateEncoder)

- **Mức 1 (nội miền):** với từng miền độc lập, một GCN (3 lớp, hidden 32, LayerNorm + residual) chạy trên đồ thị nội miền cho ra embedding từng snode; max-pool ra "tóm tắt miền".
- **Mức 2 (liên miền):** một GCN (2 lớp, hidden 32) chạy trên ma trận kề liên miền (D x D) áp lên các tóm tắt miền, cho ra embedding miền có nhận thức xuyên miền (nhưng chỉ qua tín hiệu tổng hợp mức miền, không lộ snode của miền khác).

### 2.2. Bộ mã hóa mạng ảo

- GCN (2 lớp) trên đồ thị VN cho ra embedding vnode (nhận thức láng giềng).
- Self-attention đa đầu (4 đầu) giữa các vnode cho ra embedding nhận thức toàn cục VN.

### 2.3. Các đầu ra (heads)

- **node_head:** điểm cho từng vnode = MLP([vn_embed, substrate_ctx]). Dùng để xếp thứ tự nhúng vnode.
- **link_head:** điểm cho từng vlink = MLP([vlink_feat, src_vn_embed, dst_vn_embed]). Chỉ nhìn hai đầu mút.
- **cand_head (JointCandidateHead):** đây là đầu ra trọng tâm của CARL-VNE. Cross-attention giữa vnode và từng snode ứng viên, với ngữ cảnh snode = [intra_embed, inter_domain_embed]. Đầu vào snode được bổ sung đặc trưng chi phí tường minh (cost-aware):
  - `cpu_slack` = available_cpu trừ cpu_demand (chuẩn hóa),
  - `cpu_cost` = cpu_demand x cpu_price (chuẩn hóa) — chính là số hạng node của PreCost, cấp thẳng cho mạng thay vì bắt nó tự học lại tích cpu x price,
  - `is_boundary` = 1 nếu snode có inter-domain link,
  - `link_cost` (chuẩn hóa).
  - Có các tham số học được (`cost_bias` khởi tạo -3.0, `link_cost_bias` -3.0, `boundary_bias` 0.0) cộng thẳng vào điểm: mạng khởi đầu với thứ hạng kiểu PreCost (giống mp_vne) rồi học các SAI LỆCH so với prior này. Ứng viên không đủ CPU bị mask về -1e4.
- **value_head:** ước lượng V(s) làm baseline cho học tăng cường.

---

## 3. Pipeline suy luận (solve)

File: `algorithms/carl_vne/base_vne.py`, `topk_inference.py`.

Với mỗi VNR:

1. **Mã hóa:** chạy policy network ra node_scores, link_scores, candidate_scores, value.
2. **Xếp thứ tự vnode:** theo node_head (greedy sort, hoặc Plackett-Luce sampling khi huấn luyện).
3. **Sinh ứng viên (top-K per domain):** với mỗi vnode, trong từng miền cho phép lấy top-K snode theo điểm cand_head (đã lọc đủ CPU), rồi dedupe. CARL-VNE dùng top-1 ứng viên mỗi miền cho phép (đúng phát biểu Algorithm 1 của paper: mỗi vnode chọn một ứng viên trong một miền).
4. **Chọn ánh xạ node:**
   - Chế độ PSO (triển khai chính): chạy PSO trên không gian ứng viên.
   - Chế độ direct: giải mã tự hồi quy, mỗi vnode lấy 1 snode, có mask chống đụng độ (dùng khi rollout RL).
5. **Commit ánh xạ link:** mỗi vlink ánh xạ thành một đường đi đơn (shortest path) qua bộ điều khiển toàn cục/cục bộ. Thành công nếu bottleneck băng thông của đường >= bandwidth_demand.
6. **VNR thành công** khi mọi vnode đủ CPU và mọi vlink tìm được đường đủ băng thông, không đụng độ.

### 3.1. PSO và Multi-restart

PSO mỗi lần gọi: **20 particle x 15 vòng lặp**, w = 0.7, c1 = c2 = 1.5, mutation_rate = 0.1. Mỗi particle là vector chỉ số (particle[j] trỏ vào ứng viên thứ j). Có cơ chế sửa khi hai vnode trùng snode.

Multi-restart (`multi_restart.py`): chạy PSO **K = 3 lần** (mặc định) với seed khác nhau (`k*1337 + master_seed`), chọn particle có cost thấp nhất. Lý do: tăng acceptance (nhiều lần thử thì cơ hội ít nhất một lần thành công cao hơn), giảm cost trung bình (best-of-K), giảm delay (tương quan với đường ngắn). Siêu tham số PSO từng lần giữ nguyên như baseline.

### 3.2. Hàm chi phí (cost / fitness)

```
cost = node_cost + link_cost
node_cost = Σ (vnode.cpu_demand × snode.cpu_price)
link_cost = Σ_path (transmission_delay + bandwidth_price × bandwidth_demand)
```

Đây là "PreCost". cand_head được cấp các số hạng PreCost dưới dạng đặc trưng tường minh để chấm điểm ứng viên. Trong fitness của PSO còn có thiên lệch theo policy: trừ `alpha × Σ cand_weights` (alpha = 50.0) để PSO ưu tiên các ứng viên mà cand_head cho điểm cao.

---

## 4. Huấn luyện hai giai đoạn

### Giai đoạn 1: Tiền huấn luyện bắt chước (IL)

cand_head được huấn luyện bắt chước chuyên gia MP-VNE (xếp hạng theo PreCost). Giai đoạn này cũng đóng vai trò baseline ablation: chỉ bắt chước, chưa có học tăng cường. Kết quả tham chiếu trên 100-node: khoảng 24.3% acceptance (qua PSO).

Bộ huấn luyện (`trainer.py`) dùng actor-critic với entropy annealing (hệ số entropy giảm tuyến tính từ 0.05 xuống 0.005 trong 60 batch) và critic warm-up (8 batch đầu hoãn gradient policy để V(s) bớt nhiễu).

### Giai đoạn 2: Tinh chỉnh bằng học tăng cường (direct-decoding)

Đây là điểm khác biệt tạo nên CARL-VNE so với phương án chỉ bắt chước.

Phương pháp huấn luyện: **REINFORCE / advantage actor-critic**. Mỗi batch thực hiện một bước cập nhật gradient on-policy, dùng critic V(s) làm baseline và chuẩn hóa advantage, kèm neo KL về policy sau giai đoạn bắt chước và entropy bonus. Các checkpoint dùng trong thí nghiệm (cả 100-node và 50-node, nhánh cost-focused) đều theo cách này.

- **Rollout "direct":** dùng `rank_direct(sample=True)` lấy mẫu 1 snode mỗi vnode (khám phá thực sự trên không gian ứng viên). Thứ tự vnode cố định theo node_head, chỉ phần chọn ứng viên là ngẫu nhiên.
- **Reward (chế độ cost_rel, thông số thực tế của recipe cost-focused):**
  - `reward_success = α + λ × rel_cost`, với α = 0.5, λ = 1.0.
  - `rel_cost = (cost_EMA − actual_cost) / cost_EMA`, trong đó cost_EMA là trung bình trượt mũ (decay 0.95) của chi phí các lần thành công gần đây. Tức là thưởng thêm khi nhúng rẻ hơn mức nền. Recipe cost-focused đặt λ cao (= 1.0) để nhấn mạnh số hạng chi phí.
  - `reward_fail = −0.5`.
  - (Giá trị mặc định trong argparse là α = 1.0, λ = 0.3, fail = −1.0; recipe cost-focused override như trên. Có thêm chế độ `rev_cost` thưởng theo tỉ lệ revenue/cost cho thí nghiệm size-invariant.)
- **Kỷ luật đóng băng (freeze):** target = `cand`, đóng băng node_head và link_head; chỉ huấn luyện encoder + cand_head + value_head. Lý do: gradient từ các head ngẫu nhiên/không liên quan sẽ làm hỏng encoder dùng chung.
- **Neo KL:** phạt KL(π_new || π_ref) trên phân phối cand so với policy sau giai đoạn bắt chước, hệ số β_KL = 0.1. Mục tiêu là tinh chỉnh, không để phân phối sụp đổ.
- **Critic + advantage:** advantage = reward − V(s), có chuẩn hóa; β_entropy = 0.01; value_coef = 0.5 cho MSE của critic.
- **Tối ưu:** AdamW, lr = 3e-4 (thấp hơn IL 1e-3), weight_decay = 1e-4, episodes = 5000/epoch, batch_size = 16.

Kết quả tham chiếu (100-node, ghi nhận trong code): cand_head học chọn snode tốt hơn (KL drift, entropy sắc lại), acceptance tăng từ khoảng 24.3% (giai đoạn bắt chước, qua PSO) lên khoảng 25.9% (CARL-VNE qua PSO). Đây là phần khám phá mà đường argmax cứng nhắc của giai đoạn bắt chước thiếu.

### Recipe "cost-focused" (CF)

Nhánh huấn luyện cost-focused (dùng cho cả checkpoint 100-node và 50-node trong các sweep robustness) đặt λ cost cao (= 1.0) trong reward để nhấn mạnh chi phí, cho acceptance trên test nhỉnh hơn nhánh normal.

---

## 5. Bảng siêu tham số chính

| Tham số | Giá trị | Vị trí |
|---|---|---|
| GCN nội miền (lớp / hidden) | 3 / 32 | policy_network.py |
| GCN liên miền (lớp / hidden) | 2 / 32 | policy_network.py |
| VN GCN (lớp) / self-attention (đầu) | 2 / 4 | policy_network.py |
| PER_DOMAIN_K (CARL-VNE) | 1 (top-1 mỗi miền) | topk_inference.py |
| PSO particle x iteration | 20 x 15 | base_vne.py |
| PSO w, c1, c2, mutation | 0.7, 1.5, 1.5, 0.1 | base_vne.py |
| Multi-restart K | 3 | multi_restart.py |
| policy_bias alpha (PSO) | 50.0 | base_vne.py / topk_inference.py |
| Phương pháp RL fine-tune | REINFORCE + critic baseline | script fine-tune RL |
| Reward CF: α success / λ cost / fail | 0.5 / 1.0 / −0.5 | run log (cost-focused) |
| cost EMA decay | 0.95 | script fine-tune RL |
| β_KL / β_entropy | 0.1 / 0.01 | script fine-tune RL |
| value_coef | 0.5 | script fine-tune RL |
| LR / weight_decay (RL fine-tune) | 3e-4 / 1e-4 | script fine-tune RL |
| episodes / batch_size | 5000 / 16 | script fine-tune RL |

---

## 6. Kết quả thực nghiệm

Trên hai quy mô hạ tầng (50-node và 100-node), CARL-VNE (cost-focused) được so với baseline MP-VNE qua 5 trục phân phối VNR (lifetime, size, density, resource, region), mỗi trục 5 điểm. CARL-VNE thắng ở đa số điểm; lợi thế rộng hơn ở 50-node (center +7.2 điểm) so với 100-node (center +2.5 điểm), thu hẹp ở các vùng cực đoan (lifetime rất dài, VNR rất nhỏ hoặc rất lớn). Chi tiết hình và bảng số: `experiments/robustness_figures/` và README đi kèm.

---

## 7. Quan hệ với baseline và các biến thể

| Thành phần | Vai trò |
|---|---|
| MP-VNE | Baseline heuristic, PSO theo PreCost (không có mạng học). |
| Giai đoạn bắt chước (IL) | Tiền huấn luyện cand_head; cũng là baseline ablation (chỉ bắt chước). |
| CARL-VNE | Phương pháp đề xuất (kiến trúc phân cấp + cand_head tinh chỉnh bằng RL: REINFORCE + critic baseline). |
| Chế độ direct | Giải mã trực tiếp, 1 snode/vnode tự hồi quy. |
| Chế độ PSO | Triển khai qua PSO multi-restart (cách dùng chính khi đánh giá). |
