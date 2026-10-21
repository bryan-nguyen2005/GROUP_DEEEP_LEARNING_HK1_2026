# DÀN Ý CHI TIẾT BÁO CÁO BÀI TẬP LỚN (10 - 20 TRANG)
## ĐỀ TÀI: XẤP XỈ HÀM MẬT ĐỘ GAUSSIAN 2 CHIỀU BẰNG MẠNG MLP (PYTORCH)
> **Định dạng:** Microsoft Word (`.docx`) và xuất sang `.pdf`.  
> **Nguyên tắc phân công:** Mỗi thành viên tự viết nội dung phần mình phụ trách (khoảng 3–4 trang/người). Thành viên 4 đóng vai trò Trưởng ban biên tập (Lead Editor) để kiểm tra tính nhất quán, đánh số hình/bảng và đồng bộ định dạng.

---

### TRANG BÌA & BẢNG PHÂN CÔNG ĐÓNG GÓP (TỔNG 100%)
* **Bảng phân công:**
  | STT | Họ và Tên | MSSV | Vai trò chính | Nhiệm vụ viết báo cáo | Tỷ lệ đóng góp |
  |:---:|:---|:---:|:---|:---|:---:|
  | 1 | [Điền tên TV1] | [MSSV] | Data & Architecture Lead | Viết Mục 1 & Mục 2 (Dữ liệu, Gaussian 2D, Scatter/Surface 3D) | **25%** |
  | 2 | [Điền tên TV2] | [MSSV] | Training Pipeline & Exp A | Viết Mục 3 & Mục 4.1 (Pipeline PyTorch, Baseline, Thực nghiệm A) | **25%** |
  | 3 | [Điền tên TV3] | [MSSV] | Experiments B & C Lead | Viết Mục 4.2 & Mục 4.3 (Thực nghiệm B Độ rộng, Thực nghiệm C Độ sâu, Thời gian & Loss hội tụ) | **25%** |
  | 4 | [Điền tên TV4] | [MSSV] | Evaluation & Lead Editor | Viết Mục 5 & Mục 6 (Best Model, Thảo luận, Kết luận); Tổng hợp và chuẩn hóa format | **25%** |

---

### 1. GIỚI THIỆU & PHÁT BIỂU BÀI TOÁN (Phụ trách: Thành viên 1 - ~2 trang)
* **1.1. Đặt vấn đề & Mục tiêu:** Ý nghĩa của bài toán xấp xỉ hàm (Function Approximation), định lý xấp xỉ phổ quát (Universal Approximation Theorem).
* **1.2. Phát biểu toán học:** Công thức Gaussian 2D với $\boldsymbol{\mu} = [0, 0]^T, C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$, miền $x_1 \in [-2, 2], x_2 \in [-4, 4]$.
* **1.3. Bảng phân công nhiệm vụ nhóm & cam kết trung thực học thuật.**

---

### 2. DỮ LIỆU & KHẢO SÁT BAN ĐẦU (Phụ trách: Thành viên 1 - ~2 trang)
* **2.1. Phân chia tập dữ liệu chuẩn khoa học (Zero Data Leakage):**
  * Tách biệt 3 tầng Seed: `seed_data_gen` (42 cho test, 2024 cho pool), `seed_split` (quy tắc chỉ số dòng cố định), `seed_train` (`[42, 100, 2024]`).
  * Tập Test độc lập cố định 100 mẫu (`data/test_dataset_100.csv`) - chỉ dùng đánh giá cuối cùng.
  * Pool dữ liệu (`data/train_val_pool.csv`):
    - Tập Validation cố định: 400 mẫu đầu tiên (`iloc[0:400]`).
    - Tập Train Pool: 4600 mẫu còn lại (`iloc[400:5000]`), bảo đảm tập train không bao giờ chứa tập validation.
    - Quy tắc trích mẫu cho B/C (`iloc[400:1900]`, $N=1500$) và A (tập con lồng nhau `iloc[400:400+A]`).
* **2.2. Khảo sát trực quan:**
  * Bề mặt lý thuyết Gaussian 2 chiều (`figures/ground_truth_surface.png`).
  * Phân tán 3D Scatter của tập mẫu (`figures/test_3d_scatter.png`).
  * Bảng thống kê mô tả (Mean, Min, Max, Std) của $x_1, x_2, y$.

---

### 3. KIẾN TRÚC MÔ HÌNH & THIẾT LẬP HUẤN LUYỆN (Phụ trách: Thành viên 2 - ~3 trang)
* **3.1. Kiến trúc GaussianMLP bằng PyTorch (`src/models.py`):**
  * Sơ đồ mạng nơ-ron (2 đầu vào $\to C$ tầng ẩn $B$ nơ-ron $\to 1$ đầu ra tuyến tính).
  * Lựa chọn hàm kích hoạt: Khẳng định `ReLU` làm chuẩn Baseline hiện đại; phân tích đối chiếu với `Tanh`, `Sigmoid`.
* **3.2. Pipeline huấn luyện chuẩn & Cơ chế Early Stopping (`src/trainer.py`):**
  * DataLoader, `nn.MSELoss()`, optimizer `Adam` (lr=0.005, batch 32, epochs 150).
  * Vòng lặp train (`model.train()`, `zero_grad()`, `backward()`, `step()`) và validation (`model.eval()`, `torch.no_grad()`).
  * Checkpoint và Early Stopping theo Validation Loss (patience = 15).
  * Ghi nhận lịch sử loss từng epoch ra DataFrame/CSV (`logs/history_*.csv`).
* **3.3. Cấu hình Baseline chuẩn cố định & Bộ chỉ số đo lường (`src/metrics.py`):**
  * Bảng công bố cấu hình Baseline chuẩn (Adam, lr=0.005, batch 32, epochs 150, 1 layer 64 units, ReLU).
  * Công thức toán và ý nghĩa của MSE, MAE, RMSE, R2.

---

### 4. KẾT QUẢ CÁC NHÓM THỰC NGHIỆM (Phụ trách: Thành viên 2 & 3 - ~6 trang)
* **4.1. Thực nghiệm A: Ảnh hưởng của Quy mô dữ liệu (TV2 - ~2 trang):**
  * Áp dụng kỹ thuật **tập con lồng nhau (Nested Subsets)**: $D_{200} \subset D_{400} \subset \dots \subset D_{2000}$.
  * Bảng số liệu tổng hợp Mean $\pm$ Std từ `logs/exp_A_results.csv` và truy vết `logs/raw_runs_exp_A.csv`.
  * Đồ thị sai số theo $A$: `figures/exp_A_samples_vs_error.png`.
  * Phân tích xu hướng: Tốc độ giảm sai số và ngưỡng bão hòa kích thước mẫu.
* **4.2. Thực nghiệm B: Ảnh hưởng của Độ rộng tầng ẩn $B$ (TV3 - ~2 trang):**
  * Cố định $N=1500$ mẫu train, $N=400$ mẫu val. Khảo sát $B \in [1, 10+]$ (MLP 1 tầng ẩn).
  * Bảng số liệu Mean $\pm$ Std từ `logs/exp_B_results.csv` và đồ thị `figures/exp_B_width_vs_error.png`.
  * Phân tích hiện tượng: Thiếu khớp (Underfitting khi $B=1, 2$) và quá khớp (Overfitting khi $B$ lớn).
* **4.3. Thực nghiệm C: Ảnh hưởng của Độ sâu mạng $C$ & Thời gian huấn luyện (TV3 - ~2 trang):**
  * Cố định 32 nơ-ron/tầng ẩn, khảo sát $C \in [1, 2, 3, 4, 5]$.
  * Bảng số liệu từ `logs/exp_C_results.csv`: Test MSE, epoch hội tụ và **thời gian huấn luyện (giây)**.
  * Đồ thị sai số: `figures/exp_C_depth_vs_error.png` và biểu đồ cột thời gian: `figures/exp_C_depth_vs_time.png`.
  * **Đường cong hội tụ loss qua các epoch:** `figures/exp_C_loss_convergence.png` (minh chứng trực quan loss hội tụ theo yêu cầu đề bài).
  * Phân tích: Đánh đổi giữa chi phí tính toán và độ chính xác; hiện tượng bão hòa hoặc vanishing gradient khi tăng độ sâu.

---

### 5. ĐÁNH GIÁ MÔ HÌNH TỐI ƯU & THẢO LUẬN (Phụ trách: Thành viên 4 - ~3 trang)
* **5.1. Thẩm định và khóa cấu hình tối ưu (Minh chứng khoa học chặt chẽ):**
  * Nguyên tắc: Không tự ý ghép thông số chưa qua kiểm chứng.
  * Thẩm định cấu hình ứng viên trên 3 seeds cố định `[42, 100, 2024]` trên tập Validation (400 mẫu).
  * Đối sánh công bằng với benchmark từ các thực nghiệm A/B/C và kích hoạt cơ chế Fallback nếu ứng viên không vượt trội.
  * Khóa checkpoint đại diện theo quy tắc duy nhất (gần mean nhất, hòa chọn seed nhỏ nhất) và đánh giá ngoại kiểm lần cuối trên **tập Test 100 mẫu độc lập**: Bảng metrics từ `logs/best_model_test_metrics.csv` (MSE, MAE, RMSE, R2).
  * Đường cong học tập của mô hình tối ưu: `figures/best_model_learning_curve.png`.
* **5.2. Trực quan hóa đối sánh cao cấp:**
  * Biểu đồ Parity ($y$ vs $\hat{y}$) và Residual Plot: `figures/best_model_parity_and_residuals.png`.
  * Biểu đồ 3D đối sánh Bề mặt thật vs Bề mặt mô hình dự đoán: `figures/best_model_3d_comparison.png`.
* **5.3. Thảo luận, Giới hạn & Đề xuất cải tiến:**
  * So sánh với phương pháp của báo cáo cũ 2018 (ưu thế của PyTorch hiện đại, early stopping).
  * Hạn chế của MLP đối với hàm mật độ xác suất và đề xuất mở rộng (Normalizing Flows).

---

### 6. KẾT LUẬN & TÀI LIỆU THAM KHẢO (Phụ trách: Thành viên 4 - ~1 trang)
* Tóm tắt toàn bộ kết quả thực nghiệm và các bài học rút ra.
* Danh mục tài liệu tham khảo chuẩn IEEE / APA.
