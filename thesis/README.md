# Đồ án tốt nghiệp: Reinforcement Learning cho Virtual Network Embedding

Repo viết đồ án tốt nghiệp dựa trên mẫu **SOICT_DATN_Research_VIE_Template** của Trường CNTT&TT - ĐH Bách khoa Hà Nội.

## Cấu trúc

```
DoAn.tex                            # File chính
Bia.tex / Bia_lot.tex               # Bìa ngoài và bìa lót
Tu_viet_tat.tex                     # Danh mục từ viết tắt
lstlisting.tex                      # Cấu hình listings (chèn code)
Danh_sach_tai_lieu_tham_khao.bib    # Bibliography
Chuong/
  0_2_Loi_cam_on.tex
  0_3_Tom_tat_noi_dung.tex
  0_4_Tom_tat_noi_dung_eng.tex      # Abstract tiếng Anh
  1_Gioi_thieu.tex                  # Chương 1: Giới thiệu đề tài
  2_Co_so_ly_thuyet.tex             # Chương 2: Cơ sở lý thuyết + Related work
  3_De_xuat.tex                     # Chương 3: Phương pháp đề xuất
  4_Phan_tich_ly_thuyet.tex         # Chương 4: Phân tích lý thuyết (tuỳ chọn)
  5_Thuc_nghiem.tex                 # Chương 5: Thực nghiệm
  6_Ket_luan.tex                    # Chương 6: Kết luận
  7_Tai_lieu_tham_khao.tex          # Một số lưu ý về TLTK
  Phu_luc_A.tex / Phu_luc_B.tex     # Phụ lục
Hinh_ve/                            # Thư mục chứa hình (PNG, PDF, ...)
Makefile / latexmkrc                # Build tooling
```

## Yêu cầu hệ thống

Cần một bản phân phối TeX có pdflatex + biblatex + glossaries + algorithm2e.

### macOS

```bash
# Cách 1: MacTeX đầy đủ (~5GB)
brew install --cask mactex-no-gui

# Cách 2: BasicTeX gọn nhẹ + cài thêm package
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk biblatex biber glossaries glossaries-extra \
    titlesec algorithm2e ifoddpage relsize tools enumitem subfiles \
    appendix outlines fancybox indentfirst array subfiles chngcntr \
    pdflscape capt-of multirow xurl scrextend tocbasic blindtext \
    hypcap glossary-superragged tracklang vietnam vntex
```

Mở terminal mới sau khi cài, kiểm tra:
```bash
which pdflatex latexmk
```

## Build

```bash
make           # build PDF (DoAn.pdf)
make watch     # auto-rebuild khi sửa file
make view      # build và mở PDF (macOS)
make clean     # xoá file phụ
make distclean # xoá cả PDF
```

Hoặc build thủ công:
```bash
latexmk DoAn.tex
```

## Mẹo viết

- **Chèn hình**: đặt vào `Hinh_ve/` rồi `\includegraphics[width=...]{ten_file}` (không cần phần mở rộng, không cần đường dẫn vì `\graphicspath` đã cấu hình).
- **Cite**: thêm entry vào `Danh_sach_tai_lieu_tham_khao.bib` rồi `\cite{key}`.
- **Từ viết tắt**: thêm vào `Tu_viet_tat.tex`. Lần đầu xuất hiện dùng `\gls{vne}` để in dạng đầy đủ, các lần sau in viết tắt tự động.
- **Code**: bọc trong `\begin{lstlisting}[language=Python,caption=...]{...}\end{lstlisting}`.

## Tiến độ viết

Mỗi chương đã được khởi tạo với phần khung phù hợp đề tài VNE+RL. Tìm các marker `(TODO)` / `% TODO` để biết chỗ cần viết tiếp.

## Khôi phục template gốc

Template gốc nằm ở `../SOICT_DATN_Research_VIE_Template/` (cùng cấp với thư mục `thesis/`).
