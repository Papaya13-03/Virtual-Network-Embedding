# Cấu trúc đồ án (theo example.pdf — HUST graduation thesis)

Cấu trúc bám theo `example.pdf`: phần **MỞ ĐẦU** và **KẾT LUẬN** là chương không đánh số, thân bài gồm **4 chương đánh số** (Nền tảng → Bài toán → Phương pháp → Thực nghiệm), giữ lại hai phụ lục.

## Tổng phân bổ

| Phần | File | Đánh số |
|---|---|---|
| Bìa | `Bia.tex` | (không số) |
| Bìa lót | `Bia_lot.tex` | (không số) |
| Lời cảm ơn | `Chuong/0_2_Loi_cam_on.tex` | roman |
| Tóm tắt Việt | `Chuong/0_3_Tom_tat_noi_dung.tex` | roman |
| Abstract EN | `Chuong/0_4_Tom_tat_noi_dung_eng.tex` | roman |
| Mục lục, Danh mục hình/bảng, Từ viết tắt | (tự sinh) | roman |
| **MỞ ĐẦU** | `Chuong/0_5_Mo_dau.tex` | arabic, **không đánh số** |
| **Chương 1–4** | (xem dưới) | arabic |
| **KẾT LUẬN** | `Chuong/5_Ket_luan.tex` | arabic, **không đánh số** |
| Tài liệu tham khảo | (tự sinh từ `.bib`) | arabic |
| Phụ lục A, B | `Chuong/Phu_luc_A.tex`, `Phu_luc_B.tex` | phụ lục |

## Thân bài

### MỞ ĐẦU — `Chuong/0_5_Mo_dau.tex` (chương không đánh số)
- Đặt vấn đề
- Mục tiêu và định hướng giải pháp
- Đóng góp của đồ án
- Bố cục đồ án

### Chương 1 — Nền tảng lý thuyết — `Chuong/1_Nen_tang.tex`
- 1.1 Học tăng cường sâu (MDP, policy gradient & actor-critic, PPO, Plackett-Luce)
- 1.2 Mạng nơ-ron đồ thị (truyền tin, GCN, GAT)
- 1.3 Cơ chế attention cho lựa chọn ứng viên
- 1.4 Các hướng tiếp cận cho bài toán VNE (heuristic, metaheuristic, học tăng cường, so sánh)

### Chương 2 — Bài toán nhúng mạng ảo — `Chuong/2_Bai_toan.tex`
- 2.1 Mạng vật lý và mạng ảo
- 2.2 Hai pha: ánh xạ nút và ánh xạ liên kết
- 2.3 Hàm mục tiêu và độ đo hiệu năng
- 2.4 Phát biểu bài toán dưới dạng ILP

### Chương 3 — Phương pháp đề xuất — `Chuong/3_Phuong_phap.tex`
- 3.1 Tổng quan giải pháp
- 3.2 Bộ mã hoá đồ thị (Graph Encoder)
- 3.3 Bộ giải mã chọn ứng viên (Candidate Decoder)
- 3.4 Môi trường học tăng cường
- 3.5 Quy trình huấn luyện hai giai đoạn
- 3.6 Phân tích lý thuyết (độ phức tạp tính toán, bộ nhớ, hội tụ)

### Chương 4 — Đánh giá thực nghiệm — `Chuong/4_Thuc_nghiem.tex`
- 4.1 Cấu hình thực nghiệm
- 4.2 Baseline và độ đo
- 4.3 Kết quả tổng thể (50 / 100 / 200 nút)
- 4.4 Phân tích robustness theo phân phối VNR (sweep 5 trục, 50 nút)
- 4.5 Phân tích ablation
- 4.6 Trực quan hoá quỹ đạo huấn luyện
- 4.7 Thảo luận

### KẾT LUẬN — `Chuong/5_Ket_luan.tex` (chương không đánh số)
- Kết quả đạt được
- Hạn chế
- Hướng phát triển trong tương lai

## Ghi chú

- `MỞ ĐẦU` và `KẾT LUẬN` dùng `\chapter*` + `\addcontentsline` + `\markboth` trong `DoAn.tex`; các mục bên trong dùng `\section*` để không bị đánh số nhầm theo bộ đếm chương.
- Chương 4 cũ (Phân tích lý thuyết) đã được gộp vào Chương 3 dưới mục 3.6.
- Phần khảo sát các hướng tiếp cận (related works) nằm ở Chương 1 (mục 1.4), tách khỏi phát biểu bài toán ở Chương 2.
