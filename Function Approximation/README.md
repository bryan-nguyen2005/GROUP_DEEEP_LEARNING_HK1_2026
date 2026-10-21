# BÀI TẬP LỚN HỌC SÂU: XẤP XỈ HÀM SỐ DÙNG MẠNG MLP (PYTORCH)

> **Môn học:** Học sâu (Deep Learning)  
> **Chủ đề:** Xấp xỉ hàm mật độ xác suất Gaussian 2 chiều bằng mạng Perceptron đa tầng (MLP) trên PyTorch.  
> **Môi trường thực thi & kiểm chứng:**  
> - **Môi trường thực tế đã kiểm chứng:** Python 3.12.0 trên Windows 11 (64-bit).  
> - **Thư viện chính đã kiểm chứng:** PyTorch `2.14.0+cpu` (thuần CPU, không yêu cầu GPU). Cú pháp và nạp module (`py_compile`) đã vượt qua 100%.  
> - **Khả năng tương thích lý thuyết (Dự kiến):** Hỗ trợ Python 3.10–3.12, Linux và macOS (mã nguồn sử dụng thư viện chuẩn đa nền tảng, chưa nghiệm thu chạy end-to-end trên các hệ điều hành này).  
> - **Căn cứ tài liệu:** Tuân thủ nghiêm ngặt 100% yêu cầu kỹ thuật và Rubric từ `MLP_Exercise - Yeu Cau.pdf` và slide `MLP_Exercise.pdf` ($\boldsymbol{\mu} = [0, 0]^T$, $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$, PyTorch trên CPU). Tài liệu `Sample_Report.pdf` (2018) chỉ dùng để tham khảo bố cục báo cáo học thuật, nhóm tuyệt đối không dùng code TensorFlow 1.x hay thông số cũ ($C = I$) từ mẫu này.

---

## 👥 THÔNG TIN NHÓM THỰC HIỆN & PHÂN CÔNG (TỔNG 100%)

| STT | Họ và Tên | MSSV | Vai trò chính | Nhiệm vụ code | Nhiệm vụ Báo cáo & Slide | Đóng góp |
|:---:|:---|:---:|:---|:---|:---|:---:|
| 1 | *[Nguyễn Văn A]* | *[20210001]* | **Data & Architecture Lead** | `src/data_generator.py`<br>`src/models.py`<br>`src/visualize.py` (3D) | Mục 1 & 2 Báo cáo<br>Slide 1, 2, 3, 4 | **25%** |
| 2 | *[Trần Thị B]* | *[20210002]* | **Training Pipeline & Exp A** | `src/metrics.py`<br>`src/trainer.py`<br>`src/run_experiment_A.py` | Mục 3 & 4.1 Báo cáo<br>Slide 5, 6 | **25%** |
| 3 | *[Lê Văn C]* | *[20210003]* | **Experiments B & C Lead** | `src/run_experiment_B.py`<br>`src/run_experiment_C.py` | Mục 4.2 & 4.3 Báo cáo<br>Slide 7, 8 | **25%** |
| 4 | *[Phạm Thị D]* | *[20210004]* | **Evaluation & Lead Editor** | `src/evaluate_best_model.py`<br>`src/visualize.py` (Trends/Curves/Parity) | Mục 5 & 6 Báo cáo<br>Slide 9-12; Tổng hợp & format | **25%** |

*(Ghi chú: Thành viên khi tải repo về hãy cập nhật Họ Tên và MSSV chính thức của nhóm mình vào bảng trên).*

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN CHUẨN HÓA

```text
.
├── data/                               # Chứa dữ liệu CSV sinh ra (trống ban đầu)
│   ├── test_dataset_100.csv            # Tập test độc lập cố định 100 mẫu (seed=42)
│   └── train_val_pool.csv              # Pool 5000 mẫu (seed=2024): 400 val, 4600 train
├── src/                                # Khung sườn mã nguồn mô-đun hóa
│   ├── __init__.py
│   ├── data_generator.py               # TV1: Hàm tính Gaussian 2D, sinh & lưu dữ liệu, hàm chia tập
│   ├── models.py                       # TV1: Lớp GaussianMLP tùy biến n tầng, n nơ-ron
│   ├── metrics.py                      # TV2: Hàm tính MSE, MAE, RMSE, R2
│   ├── trainer.py                      # TV2: Pipeline huấn luyện, Early Stopping, Checkpoint, History loss
│   ├── visualize.py                    # TV1 & TV4: Tiện ích vẽ 3D scatter, surface, trends, loss curves, parity
│   ├── run_experiment_A.py             # TV2: Script chạy Thực nghiệm A (Số mẫu lồng nhau)
│   ├── run_experiment_B.py             # TV3: Script chạy Thực nghiệm B (Độ rộng B)
│   ├── run_experiment_C.py             # TV3: Script chạy Thực nghiệm C (Độ sâu C, đo thời gian & loss hội tụ)
│   └── evaluate_best_model.py          # TV4: Thẩm định validation & đánh giá mô hình tối ưu trên test set
├── logs/                               # Kết quả logs CSV để truy vết số liệu
│   ├── exp_A_results.csv               # Bảng tổng hợp Thực nghiệm A (Mean ± Std)
│   ├── raw_runs_exp_A.csv              # Nhật ký chi tiết từng run của Exp A (3 seeds x 10 mốc)
│   ├── exp_B_results.csv               # Bảng tổng hợp Thực nghiệm B (Mean ± Std)
│   ├── raw_runs_exp_B.csv              # Nhật ký chi tiết từng run của Exp B (3 seeds x cấu hình B)
│   ├── exp_C_results.csv               # Bảng tổng hợp Thực nghiệm C (Mean ± Std)
│   ├── raw_runs_exp_C.csv              # Nhật ký chi tiết từng run của Exp C (3 seeds x độ sâu C, thời gian)
│   ├── best_model_test_metrics.csv     # Bảng metrics mô hình tối ưu trên tập Test 100 mẫu
│   ├── history_best_model.csv          # Lịch sử loss per-epoch của Best Model
│   └── history_exp_*.csv               # Lịch sử loss per-epoch cho TẤT CẢ các run của A, B, C
├── checkpoints/                        # Trọng số mô hình tốt nhất (.pth)
├── figures/                            # Biểu đồ đồ thị xuất ra chất lượng cao (300 DPI)
│   ├── ground_truth_surface.png        # Bề mặt 3D hàm Gaussian giải tích
│   ├── test_3d_scatter.png             # Phân tán 3D các điểm dữ liệu
│   ├── exp_A_samples_vs_error.png      # Đồ thị sai số theo quy mô mẫu A
│   ├── exp_B_width_vs_error.png        # Đồ thị sai số theo độ rộng B
│   ├── exp_C_depth_vs_error.png        # Đồ thị sai số theo độ sâu C
│   ├── exp_C_depth_vs_time.png         # Biểu đồ cột thời gian huấn luyện theo C
│   ├── exp_C_loss_convergence.png      # Đồ thị đường cong hội tụ loss theo epoch (Thực nghiệm C)
│   ├── best_model_learning_curve.png   # Đường cong học tập của mô hình tối ưu
│   ├── best_model_parity_and_residuals.png # Biểu đồ Parity (y vs ŷ) và phần dư
│   └── best_model_3d_comparison.png    # Đối sánh bề mặt 3D: Hàm thật vs MLP dự đoán
├── reports/                            # Thư mục chứa Báo cáo viết (.docx & .pdf)
│   └── report_outline.md               # Dàn ý chi tiết báo cáo 10 - 20 trang
├── slides/                             # Thư mục chứa Slide thuyết trình (.pptx & .pdf)
│   └── slide_outline.md                # Dàn ý chi tiết 12 slides thuyết trình
├── main.ipynb                          # Notebook hướng dẫn chạy lần lượt từng bước
├── plan.md                             # Bản kế hoạch, ma trận phân công & Rubric 100 điểm
├── requirements.txt                    # Danh sách các thư viện cần cài đặt
└── README.md                           # Hướng dẫn sử dụng và tái lập
```

> [!NOTE]
> **Hiện trạng dự án (Project Status):** Cây thư mục hiện tại đang ở trạng thái **Scaffold & Code Skeleton** (khung sườn cấu trúc chuẩn hóa và các mẫu mã nguồn mô-đun hóa; các hàm nghiệp vụ cốt lõi đang giữ chỗ với `raise NotImplementedError` chờ 4 thành viên viết mã theo phân công). Toàn bộ 10 file mã nguồn trong `src/` đã được kiểm tra biên dịch cú pháp (`py_compile`) thành công và kiểm thử hàm chia dữ liệu `get_train_val_split` đạt chuẩn. Các file dữ liệu CSV, raw logs chi tiết từng run, 9 biểu đồ 300 DPI và checkpoint `.pth` sẽ được sinh tự động khi các thành viên hoàn thiện mã nguồn và chạy thực nghiệm; báo cáo Word/PDF và slide PPTX/PDF do nhóm tự biên soạn và lưu trữ theo tiến độ.

---

## ⚖️ NGUYÊN TẮC KHOA HỌC CỐT LÕI & CHỐNG RÒ RỈ DỮ LIỆU (ZERO DATA LEAKAGE)

1. **Phân định rõ 3 tầng Seed & Quy tắc chia tập dùng chung:**
   * `seed_data_gen`: `seed=42` cho tập test độc lập 100 mẫu (`data/test_dataset_100.csv`); `seed=2024` cho data pool 5000 mẫu (`data/train_val_pool.csv`).
   * `seed_split` (Quy chuẩn lát cắt chỉ số cố định qua API `get_train_val_split`):
     - **Tập Validation cố định:** 400 mẫu đầu tiên của data pool (`iloc[0:400]`). Cố định cho toàn bộ các thực nghiệm và thẩm định Best Model.
     - **Tập Train Pool:** 4600 mẫu còn lại (`iloc[400:5000]`), đảm bảo tập Train không bao giờ chứa tập Validation.
     - **Thực nghiệm B & C:** Lấy đúng $N=1500$ mẫu train đầu tiên của Train Pool (`iloc[400:1900]`).
     - **Thực nghiệm A:** Lấy lát cắt lồng nhau `iloc[400 : 400+A]` với $A \in [200, 2000]$, đảm bảo $D_{200} \subset D_{400} \subset \dots \subset D_{2000}$.
   * `seed_train`: Toàn bộ thí nghiệm chạy trên **3 seeds cố định `[42, 100, 2024]`** cho khởi tạo trọng số và DataLoader.
2. **Quy chế sử dụng tập Test & Validation:**
   * Tập Test 100 mẫu độc lập **tuyệt đối KHÔNG** được sử dụng để huấn luyện, tính gradient, điều chỉnh siêu tham số hay kích hoạt Early Stopping.
   * `val_mse` là thước đo duy nhất để kích hoạt Early Stopping và chọn mô hình tối ưu; `test_mse` là kết quả ngoại kiểm khách quan (unseen evaluation) đo đạc sau khi mô hình đã hoàn tất huấn luyện và đóng băng trọng số.
3. **Quy trình xác lập Best Model khoa học (Không ghép tham số tùy tiện):**
   * Mô hình tối ưu tại [`src/evaluate_best_model.py`](src/evaluate_best_model.py) phải trải qua bước thẩm định trên tập Validation (400 mẫu) qua cả **3 seeds `[42, 100, 2024]`**.
   * Nếu đề xuất cấu hình tổ hợp mới, bắt buộc phải đạt Validation MSE trung bình vượt trội hơn cấu hình tốt nhất từ A/B/C (đọc từ `logs/exp_{A,B,C}_results.csv`); nếu không vượt trội, hệ thống kích hoạt **Fallback** giữ nguyên mô hình tốt nhất cũ.
   * Khóa checkpoint đại diện theo quy tắc duy nhất (gần mean nhất, hòa chọn seed nhỏ nhất) và chỉ đánh giá kiểm thử một lần duy nhất trên tập Test độc lập 100 mẫu.

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & TÁI LẬP TOÀN DIỆN

### 1. Cài đặt môi trường
Mở terminal trong thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### 2. Trình tự thực hiện theo phân công thành viên

* **Giai đoạn 1 (Hoàn thiện mã nguồn lõi & Tiện ích):**
  - **Thành viên 1:** Viết `src/data_generator.py`, `src/models.py`, các hàm 3D trong `src/visualize.py`. Sau đó chạy lệnh sinh dữ liệu:
    ```bash
    python src/data_generator.py
    ```
  - **Thành viên 2:** Viết `src/metrics.py`, `src/trainer.py` (tích hợp ghi nhận loss per-epoch cho mọi run).
  - **Thành viên 4:** Viết ngay các tiện ích trực quan trong `src/visualize.py` (`plot_experiment_trend`, `plot_learning_curve`) để bàn giao cho TV2/TV3.

* **Giai đoạn 2 (Thực thi các nhóm thực nghiệm):**
  - **Thành viên 2:** Viết và chạy Thực nghiệm A:
    ```bash
    python src/run_experiment_A.py
    ```
  - **Thành viên 3:** Viết và chạy Thực nghiệm B & C:
    ```bash
    python src/run_experiment_B.py
    python src/run_experiment_C.py
    ```

* **Giai đoạn 3 (Thẩm định Best Model & Báo cáo):**
  - **Thành viên 4:** Viết và chạy script đánh giá Best Model:
    ```bash
    python src/evaluate_best_model.py
    ```
  - **Cả nhóm:** Viết báo cáo Word theo [reports/report_outline.md](reports/report_outline.md) và thiết kế slide theo [slides/slide_outline.md](slides/slide_outline.md). TV4 chủ trì tổng hợp định dạng.

### 3. Kiểm thử luồng tổng thể qua Jupyter Notebook
Sau khi hoàn thiện các module, mở Jupyter Notebook để chạy kiểm tra từ đầu đến cuối:
```bash
jupyter notebook main.ipynb
```
