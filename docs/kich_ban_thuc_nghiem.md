# Kịch bản thực nghiệm — CARL-VNE

Tài liệu này chi tiết hoá các kịch bản so sánh/đánh giá cho phương pháp đề xuất
(cấu hình chuẩn **V19 normal-reward**). Mỗi kịch bản nêu rõ: *mục tiêu / câu hỏi
nghiên cứu*, *biến độc lập*, *thiết lập & lệnh chạy*, *độ đo*, *đối tượng so sánh*,
*kết quả kỳ vọng*, và *trạng thái dữ liệu* (đã có / cần chạy / cần sửa code).

> Quy ước trạng thái: ✅ đã có số liệu · 🟡 chạy được ngay (chỉ cần thời gian) ·
> 🔧 cần sửa/bổ sung code trước khi chạy.

---

## 0. Tổng quan & công cụ

### 0.1 Biến và độ đo chung

- **Biến độc lập** (thay đổi giữa các kịch bản): thời gian sống VNR, kích thước/mật
  độ tô-pô (số miền, số nút/miền), thành phần kiến trúc (bật/tắt), thuật toán tinh
  chỉnh (actor-critic vs PPO), chế độ thưởng.
- **Độ đo phụ thuộc**: (1) tỉ lệ chấp nhận (acceptance ratio) — *chính*; (2) doanh
  thu; (3) chi phí trung bình; (4) tỉ số doanh thu/chi phí (R/C); (5) độ trễ trung
  bình; (6) thời gian chạy. Với huấn luyện: loss, KL, entropy, value loss, succ_rate
  trực tuyến.

### 0.2 Công cụ và lệnh gốc

| Việc | Script | Tham số chính |
|---|---|---|
| Sinh dữ liệu | `scripts/generate_dataset.py` | `--scenario --requests --domains --nodes-per-domain --arrival-rate --seed` |
| Huấn luyện (tinh chỉnh) | `scripts/ppo_finetune.py` | `--ppo-mode {reinforce,ppo} --target {cand,ordering} --rollout {direct,pso} --reward-mode {cost_rel,rev_cost} --beta-kl --episodes ...` |
| Đánh giá | `scripts/run_eval.py` | `--algorithm --substrate --requests --checkpoint --output --seed` |
| Đa hạt giống | `scripts/run_multiseed.py` | (lặp `run_eval` nhiều seed) |

Tô-pô có sẵn: `datasets/scenario_{50,100,200,500,1000}nodes{,_train}`.
Mỗi mạng chia **10 miền**; mặc định `nodes-per-domain=10`, `requests=3000`,
**thời gian sống trung bình = 500** (`expovariate(1/500)`), `arrival_rate = 0.1 ×
(nodes_per_domain/10)` → **số VNR đồng thời ≈ arrival_rate × 500 ≈ 50** (giữ mức tải
~không đổi khi mạng to lên).

### 0.3 Nhắc lại: actor-critic là gì, và quan hệ với PPO

- **Actor** = chính sách $\pi_\theta(a\mid s)$ — *quyết định hành động* (ở đây: đầu
  chọn ứng viên).
- **Critic** = hàm giá trị $V_\phi(s)$ — *đánh giá trạng thái* (ở đây: `value_head`).
  Critic được học bằng hồi quy về lợi ích thực tế (MSE).
- Actor được cập nhật theo **lợi thế** $A(s,a) = r - V_\phi(s)$ thay cho phần thưởng
  thô → **giảm phương sai gradient** → ổn định hơn REINFORCE thuần.
- **PPO** *cũng là* actor-critic, nhưng thêm **hàm mục tiêu thay thế có cắt** (clipped
  surrogate, $\epsilon=0.2$) cho phép tái dùng dữ liệu qua nhiều vòng gradient mà
  không trôi chính sách.
- **Trong code này**: `--ppo-mode reinforce` (mặc định) = actor-critic
  (REINFORCE + critic + KL + entropy) — *đây là cấu hình tạo kết quả chính*.
  `--ppo-mode ppo` = thêm clipping. → Phép so sánh "PPO có bắt kịp actor-critic"
  chính là **`--ppo-mode ppo` vs `--ppo-mode reinforce`** (xem KB6).

---

## KB1 — Thời gian sống VNR: ngắn / dài (và acceptance theo lifetime)

*(gộp ý 1 và ý 4 của thầy: lifetime ngắn/dài, và acceptance thay đổi ra sao khi
request sống lâu hơn)*

- **Câu hỏi**: chính sách phản ứng thế nào khi VNR sống ngắn (luân chuyển nhanh, tài
  nguyên giải phóng sớm) so với sống dài (tài nguyên bị giữ lâu, tải dồn cao)?
- **Biến độc lập**: thời gian sống trung bình $\bar{L} \in \{125, 250, 500, 1000,
  2000\}$ (quét quanh mốc 500 hiện tại).
- **Điểm tinh tế (quan trọng)**: số VNR đồng thời $\approx$ arrival\_rate $\times
  \bar{L}$. Có **hai cách thiết kế**:
  - **(a) Giữ arrival\_rate cố định** → lifetime dài hơn ⇒ tải đồng thời tăng ⇒
    acceptance *giảm*. Đây là kịch bản "áp lực trực tuyến" — phản ánh thực tế.
  - **(b) Giữ tải đồng thời cố định** (đặt arrival\_rate $\propto 1/\bar{L}$) → cô
    lập *thuần* ảnh hưởng của churn/độ dài, loại nhiễu do tải. Nên báo cáo cả hai để
    tách bạch nguyên nhân.
- **Thiết lập**: 🔧 *generator hardcode lifetime=500 (dòng 141), cần thêm cờ
  `--lifetime-mean`*. Sau khi thêm:
  ```bash
  for L in 125 250 500 1000 2000; do
    # cách (a): arrival cố định
    python scripts/generate_dataset.py --scenario 50nodes_life$L --lifetime-mean $L
    # cách (b): giữ tải cố định (ví dụ ~50 VN): arrival = 50 / L
    python scripts/generate_dataset.py --scenario 50nodes_life${L}_iso --lifetime-mean $L --arrival-rate $(python -c "print(50/$L)")
  done
  ```
  Đánh giá dùng **cùng checkpoint V19** (không train lại): `run_eval.py` cho từng
  dataset, vẽ acceptance theo $\bar{L}$.
- **Độ đo**: acceptance, R/C, chi phí theo $\bar{L}$; so V19 vs MP-VNE-V4.
- **Kỳ vọng**: cách (a) — acceptance giảm đơn điệu theo $\bar{L}$ ở cả hai phương
  pháp, nhưng V19 giữ khoảng cách dương so với baseline. Cách (b) — acceptance gần
  như phẳng (chứng tỏ chênh lệch ở (a) là do tải chứ không do bản chất lifetime).
- **Trạng thái**: 🔧 cần thêm `--lifetime-mean` rồi 🟡 chạy.

---

## KB2 — Cải tiến đến từ đâu? (ablation thành phần)

*(ý 2: chạy có/không có từng thành phần để xem ảnh hưởng)*

- **Câu hỏi**: mỗi thành phần đóng góp bao nhiêu vào kết quả?
- **Các thành phần bật/tắt** (mỗi dòng là một biến thể, giữ nguyên phần còn lại):

  | Thành phần | "Có" | "Không" | Cờ / cách |
  |---|---|---|---|
  | Giai đoạn RL (vs chỉ IL) | V19 | R2 (`il_mp_vne_v17`) | dùng checkpoint trước/sau tinh chỉnh |
  | Khám phá ứng viên | lấy mẫu | argmax | `sample_cand` true/false khi train |
  | Neo KL | $\beta_{KL}=0.1$ | $\beta_{KL}=0$ | `--beta-kl 0` |
  | Đóng băng đầu xếp hạng | đóng băng | train cả ordering | `--no-freeze-shared` / `--target ordering` |
  | Critic baseline | có | không (baseline thô) | `--no-use-critic` |
  | Phần thưởng tạo hình | cost\_rel | rev\_cost | `--reward-mode rev_cost` |
  | Bộ mã hoá | GCN (V19) | GAT (V21/V22) | đổi `--algorithm` |
- **Độ đo**: acceptance + R/C của từng biến thể, trình bày dạng bảng ablation; mỗi
  dòng "tắt" cho thấy mức sụt so với cấu hình đầy đủ.
- **Kết quả kỳ vọng / đã có**:
  - RL vs IL: ✅ **đã có** — 100 nút: 26.0% (V19) vs 17.2% (R2), tức RL đóng góp
    **+8.8 đpt** (đây là ablation mạnh nhất, đã chứng minh).
  - Khám phá ứng viên: tắt khám phá ⇒ không có tín hiệu RL ⇒ về mức R2 (luận điểm
    trung tâm). 🟡 cần chạy biến thể argmax-train để minh hoạ số.
  - Neo KL / đóng băng: kỳ vọng tắt ⇒ KL bùng nổ, acceptance sụt (mất ổn định). 🟡.
- **Trạng thái**: RL-vs-IL ✅; các biến thể còn lại 🟡 chạy được ngay bằng cờ tương ứng.

---

## KB3 — Thiết kế dữ liệu theo tô-pô

*(ý 3: nhiều node trong 1 domain, số node/domain tăng, mật độ node)*

- **Câu hỏi**: chính sách tổng quát hoá thế nào khi cấu trúc tô-pô thay đổi?
- **Ba trục biến thiên** (mỗi trục một loạt dataset):
  1. **Số nút/miền** (mật độ trong miền): `--nodes-per-domain ∈ {5, 10, 20, 40}`
     (giữ `--domains 10`).
  2. **Số miền**: `--domains ∈ {5, 10, 20}` (giữ `--nodes-per-domain 10`).
  3. **Quy mô tổng**: tô-pô 50 → 100 → 200 → 500 → 1000 nút (đã có sẵn dataset).
- **Mật độ liên kết**: nếu generator cho điều chỉnh xác suất cạnh, thêm trục mật độ
  cạnh; nếu chưa, 🔧 cần thêm tham số. (Hiện substrate 50 nút có ~50 cạnh, 100 nút
  ~110 cạnh — mật độ thấp, ~1.1 cạnh/nút.)
- **Thiết lập**:
  ```bash
  for npd in 5 10 20 40; do
    python scripts/generate_dataset.py --scenario d10_npd$npd --domains 10 --nodes-per-domain $npd
  done
  # đánh giá: dùng checkpoint train ở 50 nút (kiểm tra transfer) + checkpoint train riêng
  ```
- **Độ đo**: acceptance/R/C theo từng trục; so V19 (train 50 nút, áp thẳng) vs V19
  (train riêng từng quy mô) vs MP-VNE-V4.
- **Kết quả kỳ vọng / đã có**:
  - Quy mô: ✅ **đã có** 50/100/200 nút. Acceptance giảm khi quy mô tăng
    (50: 33.1% → 100: 26.0% → 200: 12.5%); V19 train-50 ≈ V19 train-200 (12.5% vs
    12.0%) ⇒ **chuyển giao tốt**. 🟡 còn 500/1000 nút chưa chạy.
  - Số nút/miền & số miền: 🟡 cần sinh dataset + chạy.
- **Trạng thái**: quy mô ✅ (một phần); các trục mật độ/miền 🟡 (đôi chỗ 🔧 nếu cần
  tham số mật độ cạnh).

---

## KB4 — Acceptance rate theo độ dài sống của request

→ **Đã gộp vào [KB1](#kb1--thời-gian-sống-vnr-ngắn--dài-và-acceptance-theo-lifetime)**
(đường cong acceptance theo $\bar{L}$, hai cách thiết kế a/b).

---

## KB5 — Chứng minh mô hình cài đặt đúng (hành vi huấn luyện)

*(ý 5: quá trình train biểu hiện như thế nào)*

- **Câu hỏi**: cài đặt có đúng không — mô hình có *học* được không, hay chỉ dao động
  ngẫu nhiên?
- **Bằng chứng cần trưng ra** (từ log epoch `logs/ppo_v19_50nodes_*_epoch_summary.csv`):
  1. **succ_rate trực tuyến tăng** theo epoch rồi đạt bình nguyên (✅ đã có:
     ~0.308 → ~0.35).
  2. **acceptance trên tập kiểm thử tăng** so với mốc R2 ban đầu (✅: đỉnh 33.1% @e19).
  3. **value loss giảm** (critic học được giá trị) — kiểm chứng critic không "chết".
  4. **entropy giảm có kiểm soát** (✅: 9.4 → 4.3) — chính sách dần tự tin, không sụp
     tức thì.
  5. **Sanity checks**: tắt RL ⇒ đứng yên ở R2; tăng seed khác nhau ⇒ xu hướng lặp
     lại (dùng `run_multiseed.py`).
- **Độ đo**: các đường cong trên + khoảng tin cậy qua nhiều seed.
- **Trạng thái**: ✅ phần lớn đã có (Hình 5.1, Hình hội tụ trong Ch5); 🟡 nên bổ sung
  chạy ≥3 seed để có dải tin cậy.

---

## KB6 — Hội tụ & ổn định; PPO có bắt kịp actor-critic?

*(ý 6: phải hội tụ, ổn định loss/PPO, so PPO vs actor-critic)*

- **Câu hỏi**: (a) quá trình huấn luyện có hội tụ ổn định không? (b) biến thể PPO có
  cắt có ngang bằng/tốt hơn actor-critic chuẩn không?
- **Phần (a) — ổn định**: vẽ theo epoch: `mean_loss`, `mean_policy_loss`,
  `mean_value_loss`, `mean_kl`, `mean_entropy`, và (ở chế độ PPO) `mean_ratio`,
  `mean_clip_frac`. Tiêu chí "ổn định": KL dao động quanh mức nhỏ (≈0.1–0.25) **không
  bùng nổ**, loss không phân kỳ, entropy giảm mượt. ✅ đã có cho actor-critic.
- **Phần (b) — PPO vs actor-critic**: chạy hai cấu hình **cùng seed, cùng dữ liệu,
  cùng số episode**, khác đúng một cờ:
  ```bash
  # actor-critic (đã là kết quả chính)
  python scripts/ppo_finetune.py ... --ppo-mode reinforce --log-file logs/ac_50.csv  --checkpoint checkpoints/v19_ac.pt
  # PPO có cắt
  python scripts/ppo_finetune.py ... --ppo-mode ppo --ppo-clip 0.2 --ppo-epochs 2 \
         --log-file logs/ppo_50.csv --checkpoint checkpoints/v19_ppo.pt
  # rồi run_eval cả hai checkpoint trên cùng tập kiểm thử
  ```
- **Độ đo so sánh**: (i) acceptance cuối cùng; (ii) tốc độ hội tụ (số epoch đạt 95%
  giá trị bình nguyên); (iii) độ ổn định (độ lệch chuẩn của succ_rate ở giai đoạn
  bình nguyên); (iv) chi phí tính toán/epoch (PPO tốn hơn do K vòng gradient + phát
  lại trạng thái).
- **Kết quả kỳ vọng**: PPO *ngang bằng hoặc hơi tốt hơn* về ổn định (nhờ ràng buộc
  vùng tin cậy của clipping), đổi lại chậm hơn mỗi epoch. Nếu PPO **không** vượt
  actor-critic đáng kể ⇒ kết luận: với bài toán này, actor-critic + neo KL đã đủ, và
  đó là lý do chọn nó làm cấu hình chuẩn.
- **Trạng thái**: actor-critic ✅; **PPO 🟡 cần chạy** (đây là TODO đang để mở ở
  Ch5 §5.4.3).

---

## Bảng tổng hợp ưu tiên

| KB | Nội dung | Trạng thái | Việc cần làm |
|---|---|---|---|
| KB1 | Lifetime ngắn/dài + acceptance vs lifetime | 🔧→🟡 | thêm `--lifetime-mean`, sinh 5 mức, eval |
| KB2 | Ablation thành phần | ✅(RL-vs-IL) + 🟡 | chạy các biến thể cờ còn lại |
| KB3 | Tô-pô (nút/miền, số miền, mật độ, quy mô) | ✅(quy mô một phần) + 🟡 | sinh dataset trục mới, 500/1000 nút |
| KB5 | Chứng minh cài đặt đúng | ✅ + 🟡 | bổ sung đa seed |
| KB6 | Hội tụ + PPO vs actor-critic | ✅(AC) + 🟡(PPO) | chạy `--ppo-mode ppo`, so sánh |

**Đề xuất thứ tự thực hiện**: KB6 (PPO, nhanh, lấp TODO Ch5) → KB2 (ablation cờ, nhanh)
→ KB1 (cần sửa code nhỏ) → KB3 (tốn thời gian sinh + train nhiều dataset).
