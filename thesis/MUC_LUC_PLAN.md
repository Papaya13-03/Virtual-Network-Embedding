# Kế hoạch độ dài đồ án (~60 trang)

Toàn bộ section/subsection đã được cài đặt sẵn trong `Chuong/*.tex` để khi compile `\tableofcontents` tự sinh đúng mục lục. Mỗi section trong file `.tex` có comment `% TODO: ...` mô tả nội dung cần viết và độ dài mục tiêu.

## Tổng phân bổ

| Phần | File | Trang | Đánh số |
|---|---|---:|---|
| Bìa | `Bia.tex` | 1 | (không số) |
| Bìa lót | `Bia_lot.tex` | 1 | (không số) |
| Lời cảm ơn | `Chuong/0_2_Loi_cam_on.tex` | 1 | roman |
| Tóm tắt Việt | `Chuong/0_3_Tom_tat_noi_dung.tex` | 1 | roman |
| Abstract EN | `Chuong/0_4_Tom_tat_noi_dung_eng.tex` | 1 | roman |
| Mục lục | (tự sinh) | 2 | roman |
| Danh mục hình | (tự sinh) | 1 | roman |
| Danh mục bảng | (tự sinh) | 1 | roman |
| Danh mục từ viết tắt | `Tu_viet_tat.tex` | 1 | roman |
| **Body** | | **48** | arabic |
| Tài liệu tham khảo | (tự sinh từ `.bib`) | 2 | arabic |
| **Tổng** | | **~60** | |

## Body (48 trang)

### Chương 1 — Giới thiệu đề tài (4 trang)
| Section | Trang |
|---|---:|
| 1.1 Đặt vấn đề | 1.25 |
| 1.2 Các giải pháp hiện tại và hạn chế | 1.25 |
| &nbsp;&nbsp;1.2.1 Hướng heuristic / metaheuristic | — |
| &nbsp;&nbsp;1.2.2 Hướng lập trình toán học | — |
| &nbsp;&nbsp;1.2.3 Hướng học tăng cường | — |
| 1.3 Mục tiêu và định hướng giải pháp | 0.75 |
| 1.4 Đóng góp của đồ án | 0.4 |
| 1.5 Bố cục đồ án | 0.35 |

### Chương 2 — Nền tảng lý thuyết (10 trang)
| Section | Trang |
|---|---:|
| 2.1 Mô hình hoá bài toán nhúng mạng ảo | 3.0 |
| &nbsp;&nbsp;2.1.1 Mạng vật lý và mạng ảo | — |
| &nbsp;&nbsp;2.1.2 Hai pha: ánh xạ nút và liên kết | — |
| &nbsp;&nbsp;2.1.3 Hàm mục tiêu và độ đo | — |
| &nbsp;&nbsp;2.1.4 Phát biểu ILP | — |
| 2.2 Các hướng tiếp cận hiện tại | 2.5 |
| 2.3 Học tăng cường sâu | 2.5 |
| &nbsp;&nbsp;2.3.1 MDP | — |
| &nbsp;&nbsp;2.3.2 Policy gradient và Actor-Critic | — |
| &nbsp;&nbsp;2.3.3 PPO | — |
| 2.4 Mạng nơ-ron đồ thị | 1.5 |
| 2.5 Cơ chế attention | 0.5 |

### Chương 3 — Phương pháp đề xuất (15 trang)
| Section | Trang |
|---|---:|
| 3.1 Tổng quan giải pháp | 2.0 |
| 3.2 Bộ mã hoá đồ thị (Graph Encoder) | 3.5 |
| 3.3 Bộ giải mã chọn ứng viên (Candidate Decoder) | 3.5 |
| 3.4 Môi trường học tăng cường | 2.5 |
| 3.5 Quy trình huấn luyện hai giai đoạn | 3.5 |

### Chương 4 — Phân tích lý thuyết (4 trang)
| Section | Trang |
|---|---:|
| 4.1 Độ phức tạp tính toán | 1.5 |
| 4.2 Độ phức tạp bộ nhớ | 1.0 |
| 4.3 Phân tích hội tụ | 1.5 |

### Chương 5 — Đánh giá thực nghiệm (13 trang)
| Section | Trang |
|---|---:|
| 5.1 Cấu hình thực nghiệm | 2.5 |
| 5.2 Baseline và độ đo | 2.0 |
| 5.3 Kết quả tổng thể | 3.5 |
| 5.4 Phân tích ablation | 2.5 |
| 5.5 Trực quan hoá quỹ đạo huấn luyện | 1.5 |
| 5.6 Thảo luận | 1.0 |

### Chương 6 — Kết luận (2 trang)
| Section | Trang |
|---|---:|
| 6.1 Kết quả đạt được | 0.8 |
| 6.2 Hạn chế | 0.6 |
| 6.3 Hướng phát triển | 0.6 |

## Mẹo căn trang khi viết

- **Mật độ chữ**: template dùng font 13pt, `\onehalfspacing`, lề 3.5/2.5/2/2 cm. Một trang body chứa ~280–320 từ.
- **Quy đổi**: 0.5 trang ≈ 150 từ, 1 trang ≈ 300 từ, 2 trang ≈ 600 từ.
- **Hình ảnh**: một hình cỡ `width=0.85\textwidth` chiếm khoảng 0.4–0.5 trang. Một thuật toán `algorithm2e` cỡ trung bình chiếm 0.3–0.4 trang.
- **Bảng**: bảng ngang 5 cột × 4 hàng chiếm khoảng 0.25 trang.

Nếu sau khi viết bạn lệch quá 10% so với target, ưu tiên điều chỉnh ở Chương 3 (đề xuất) và Chương 5 (thực nghiệm) — đây là 2 chương có biên độ co giãn lớn nhất.
