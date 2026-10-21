# KẾ HOẠCH & PHÂN CÔNG NHIỆM VỤ CHUẨN HÓA BÀI TẬP LỚN HỌC SÂU
## ĐỀ TÀI: XẤP XỈ HÀM SỐ DÙNG MẠNG MLP (FUNCTION APPROXIMATION USING MLP)
> **Quy mô nhóm:** 04 thành viên  
> **Căn cứ tài liệu:**  
> - **Quy chuẩn kỹ thuật bắt buộc:** Bám sát 100% nội dung, yêu cầu kỹ thuật và Rubric trong `MLP_Exercise - Yeu Cau.pdf` cùng slide bài tập `MLP_Exercise.pdf` (hàm Gaussian 2D với $\boldsymbol{\mu} = [0, 0]^T$, ma trận hiệp phương sai $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$, miền $[-2, 2] \times [-4, 4]$, triển khai thuần **PyTorch** trên **CPU**).  
> - **Tài liệu tham khảo cấu trúc:** `Sample_Report.pdf` (2018) chỉ được sử dụng để tham khảo cách bố cục báo cáo học thuật. Nhóm **tuyệt đối không** sử dụng mã nguồn cũ (TensorFlow 1.x) hay các tham số cũ ($C = I$) của tài liệu mẫu này theo đúng lưu ý của đề bài.  
> **Cam kết khoa học:** Triển khai hoàn toàn bằng **PyTorch** trên **CPU**, thực nghiệm có kiểm soát (tập con lồng nhau, cô lập biến), chống rò rỉ dữ liệu test (Zero Data Leakage).

---

## I. NGUYÊN TẮC KHOA HỌC & CHUẨN HÓA THỰC NGHIỆM

### 1. Phân định rõ 3 loại Random Seed & Quy tắc chia dữ liệu dùng chung (Zero Data Leakage)
Để đảm bảo tính tái lập 100% và loại bỏ hoàn toàn khả năng rò rỉ dữ liệu, nhóm phân định rạch ròi 3 tầng Seed và quy ước chỉ số (Index Partition):

* **Tầng 1 - Seed sinh dữ liệu (`seed_data_gen`):**
  - `data/test_dataset_100.csv`: Sinh độc lập đúng **100 mẫu** với `seed = 42`.
  - `data/train_val_pool.csv`: Sinh độc lập **5000 mẫu** với `seed = 2024`.
* **Tầng 2 - Quy chuẩn phân chia dữ liệu bằng chỉ số dòng cố định (`seed_split` / Deterministic Slicing):**
  - **Tập Validation (400 mẫu cố định):** Lấy đúng 400 dòng đầu tiên của `data/train_val_pool.csv` (chỉ số `iloc[0:400]`). Tập này cố định tuyệt đối xuyên suốt mọi thực nghiệm A, B, C và bước thẩm định Best Model. Dùng để theo dõi quá trình học, kích hoạt Early Stopping và là căn cứ duy nhất để chọn cấu hình mô hình.
  - **Tập Train Pool nguồn (4600 mẫu còn lại):** Lấy từ dòng 400 đến 5000 của `data/train_val_pool.csv` (chỉ số `iloc[400:5000]`). Đảm bảo mọi tập Train **tuyệt đối không bao giờ chứa hay chồng lấn với tập Validation**.
  - **Thực nghiệm B & C (Cố định $N = 1500$ mẫu train):** Lấy đúng 1500 mẫu đầu tiên của Train Pool, tức chỉ số `iloc[400:1900]` của `data/train_val_pool.csv`.
  - **Thực nghiệm A (Tập con lồng nhau $A \in [200, 2000]$):** Với mỗi mốc $A$, trích xuất đúng $A$ mẫu đầu tiên của Train Pool, tức chỉ số `iloc[400 : 400+A]`. Quy tắc này đảm bảo quan hệ tập con lồng nhau nghiêm ngặt:
    $$D_{200} \subset D_{400} \subset D_{600} \subset \dots \subset D_{2000}$$
    giúp cô lập hoàn toàn tác động của kích thước dữ liệu, loại bỏ nhiễu ngẫu nhiên.
  - **API chuẩn hóa bàn giao:** `get_train_val_split(pool_X: np.ndarray, pool_y: np.ndarray, n_train: int, n_val: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` trả về `(X_train, y_train, X_val, y_val)`.
* **Tầng 3 - Seed huấn luyện mô hình (`seed_train`):**
  - Nhóm sử dụng 3 random seeds cố định: **`[42, 100, 2024]`** dùng cho việc khởi tạo trọng số mạng (`torch.manual_seed`) và xáo trộn mini-batch trong `DataLoader`. Mọi báo cáo kết quả đều trình bày dưới dạng $\text{Mean} \pm \text{Std}$ qua cả 3 seeds.

### 2. Quy chuẩn minh chứng khoa học cho Best Model (Tránh ghép cấu hình tùy tiện)
- **Cảnh báo khoa học:** Tuyệt đối không tự ý ghép các thông số đơn lẻ từ các thực nghiệm độc lập (ví dụ ghép $A=2000$ từ Exp A, $B=64$ từ Exp B, $C=2$ từ Exp C) rồi mặc nhiên tuyên bố đó là "Best Model" khi cấu hình tổ hợp này chưa từng được huấn luyện hay xác nhận trên tập Validation.
- **Quy trình 2 bước chuẩn hóa để xác lập Best Model tại [`src/evaluate_best_model.py`](src/evaluate_best_model.py):**
  1. **Bước 1 - Thẩm định Validation Đa-seed & Cơ chế Fallback (Validation Confirmation):**
     - *Nguồn đối sánh Benchmark:* Đọc giá trị `val_mse_mean` thấp nhất từ các file tổng kết `logs/exp_{A,B,C}_results.csv` làm ngưỡng so sánh; nếu chưa có kết quả thực nghiệm thì dừng lại thông báo, không thực hiện đánh giá test.
     - *Khảo sát ứng viên tổ hợp mới:* Nếu đề xuất cấu hình tổ hợp mới (ví dụ $A=2000, C=2, B=64$ hoặc $B=32$), mô hình BẮT BUỘC phải được huấn luyện trên **cùng 3 seeds `[42, 100, 2024]`** trên tập train ($A=2000$) với Early Stopping trên tập Validation (400 mẫu cố định).
     - *So sánh công bằng:* Tính $\text{Mean} \pm \text{Std}$ Validation MSE qua 3 seeds để so sánh trực tiếp với benchmark từ A/B/C.
     - *Cơ chế Fallback:* Nếu Validation MSE trung bình của tổ hợp mới không tốt hơn benchmark từ A/B/C, hệ thống **bắt buộc fallback giữ nguyên cấu hình tốt nhất đã ghi nhận từ A/B/C**.
     - *Khóa checkpoint đại diện duy nhất (Deterministic Tie-Breaker):* Chọn checkpoint của seed có Validation MSE gần với giá trị trung bình (`mean_val_mse`) nhất; nếu hòa khoảng cách (bằng nhau tuyệt đối) thì chọn seed có giá trị số nhỏ nhất (ví dụ ưu tiên seed 42) làm mô hình đại diện chính thức (`checkpoints/best_model.pth`).
  2. **Bước 2 - Đánh giá ngoại kiểm khách quan trên tập Test (Final Unseen Evaluation):**
     - Nạp checkpoint đại diện (`checkpoints/best_model.pth`) và đánh giá kiểm thử một lần duy nhất trên tập Test độc lập (100 mẫu).
     - Báo cáo đầy đủ 4 chỉ số: MSE, MAE, RMSE, R2 Score.
     - Xuất biểu đồ Parity & Residuals, biểu đồ đối sánh bề mặt 3D thật vs dự đoán, và đường cong học tập.

### 3. Cố định cấu hình Baseline chuẩn (Strict Baseline Specification)
Mọi thực nghiệm đều lấy cấu hình chuẩn sau làm mốc so sánh (không dùng khoảng tham số hay lựa chọn hoặc/hoặc):
* **Kiến trúc mạng Baseline:** MLP 1 tầng ẩn, **64 nơ-ron** (`num_hidden_layers=1, hidden_units=64`).
* **Hàm kích hoạt:** **`ReLU`** cho các tầng ẩn, tầng ra là hàm tuyến tính (Linear).
* **Thuật toán tối ưu (Optimizer):** **`Adam`** với Learning Rate $\eta = 0.005$.
* **Kích thước Batch:** **`32`**.
* **Số Epoch tối đa:** **`150`**.
* **Cơ chế dừng sớm (Early Stopping):** `patience = 15` epochs dựa trên Validation Loss.
* **Thiết bị tính toán:** **`CPU`** (`torch.device('cpu')`).
* **Quy chuẩn độ sâu ở Thực nghiệm C:** Cố định **32 nơ-ron/tầng** cho mọi tầng ẩn khi khảo sát $C \in [1, 2, 3, 4, 5]$.

---

## II. BẢNG DANH MỤC ĐƯỜNG DẪN CHUẨN HÓA (CANONICAL PATHS)

Toàn bộ 4 thành viên phải tuân thủ tuyệt đối danh mục tên file và đường dẫn dưới đây:

| Phân loại | Đường dẫn chuẩn | Trách nhiệm chính | Mô tả |
|:---|:---|:---:|:---|
| **Dữ liệu** | `data/test_dataset_100.csv` | TV1 | Tập test độc lập đúng 100 mẫu (`seed=42`) |
| | `data/train_val_pool.csv` | TV1 | Pool 5000 mẫu (`seed=2024`): 400 val, 4600 train |
| **Mã nguồn** | `src/data_generator.py` | TV1 | Sinh dữ liệu Gaussian 2D, lưu/đọc CSV, hàm chia tập `get_train_val_split` |
| | `src/models.py` | TV1 | Class `GaussianMLP` tùy biến PyTorch |
| | `src/metrics.py` | TV2 | Hàm tính MSE, MAE, RMSE, R2 |
| | `src/trainer.py` | TV2 | Training loop, EarlyStopping, checkpoint, xuất loss history per-epoch |
| | `src/visualize.py` | TV1, TV4 | Tiện ích 3D (TV1) và trends/curves/parity/residuals (TV4) |
| | `src/run_experiment_A.py` | TV2 | Script chạy tự động Thực nghiệm A (lưu raw run, summary, histories) |
| | `src/run_experiment_B.py` | TV3 | Script chạy tự động Thực nghiệm B (lưu raw run, summary, histories) |
| | `src/run_experiment_C.py` | TV3 | Script chạy tự động Thực nghiệm C & xuất loss hội tụ |
| | `src/evaluate_best_model.py`| TV4 | Thẩm định validation 3 seeds, fallback & đánh giá test mô hình tối ưu |
| **Logs CSV** | `logs/exp_A_results.csv` | TV2 | Bảng kết quả tổng hợp Thực nghiệm A (Mean ± Std) |
| | `logs/raw_runs_exp_A.csv` | TV2 | Nhật ký chi tiết từng run của Exp A (tự chứa mọi tham số và kết quả) |
| | `logs/exp_B_results.csv` | TV3 | Bảng kết quả tổng hợp Thực nghiệm B (Mean ± Std) |
| | `logs/raw_runs_exp_B.csv` | TV3 | Nhật ký chi tiết từng run của Exp B (tự chứa mọi tham số và kết quả) |
| | `logs/exp_C_results.csv` | TV3 | Bảng kết quả tổng hợp Thực nghiệm C (Mean ± Std) |
| | `logs/raw_runs_exp_C.csv` | TV3 | Nhật ký chi tiết từng run của Exp C (tự chứa mọi tham số, thời gian) |
| | `logs/best_model_test_metrics.csv` | TV4 | Bảng metrics mô hình tối ưu trên test set |
| | `logs/history_best_model.csv` | TV4 | Lịch sử loss theo epoch của Best Model run |
| | `logs/history_exp_*.csv` | TV2, TV3 | Lịch sử loss theo epoch cho TẤT CẢ các run của A, B, C |
| **Hình vẽ** | `figures/ground_truth_surface.png` | TV1 | Bề mặt 3D hàm Gaussian lý thuyết |
| | `figures/test_3d_scatter.png` | TV1 | Đồ thị 3D scatter của tập dữ liệu |
| | `figures/exp_A_samples_vs_error.png`| TV2 | Đồ thị sai số theo số mẫu A |
| | `figures/exp_B_width_vs_error.png` | TV3 | Đồ thị sai số theo độ rộng B |
| | `figures/exp_C_depth_vs_error.png` | TV3 | Đồ thị sai số theo độ sâu C |
| | `figures/exp_C_depth_vs_time.png` | TV3 | Biểu đồ cột thời gian huấn luyện theo C |
| | `figures/exp_C_loss_convergence.png`| TV3 | Đồ thị đường cong hội tụ loss theo epoch cho Thực nghiệm C |
| | `figures/best_model_learning_curve.png` | TV4 | Đường cong học tập (train/val loss) của Best Model |
| | `figures/best_model_parity_and_residuals.png` | TV4 | Biểu đồ Parity ($y$ vs $\hat{y}$) & phần dư |
| | `figures/best_model_3d_comparison.png` | TV4 | Đối sánh 3D bề mặt thật vs dự đoán |
| **Tài liệu** | `reports/BTL01_BaoCao_MLP.docx` & `.pdf` | Cả nhóm (TV4 tổng hợp)| Báo cáo khoa học 10–20 trang |
| | `slides/BTL01_ThuyetTrinh_MLP.pptx` & `.pdf` | Cả nhóm (TV4 tổng hợp)| Slide thuyết trình 12–15 slide |
| | `README.md` & `requirements.txt` | TV4 | Hướng dẫn cài đặt & chạy tái lập |
| | `main.ipynb` | Cả nhóm | Notebook tổng hợp demo |

---

## III. QUY ƯỚC GIAO THỨC BÀN GIAO & SCHEMA DỮ LIỆU (HANDOVER PROTOCOLS)

Để loại bỏ hoàn toàn tình trạng tắc nghẽn giao tiếp giữa các thành viên, nhóm quy định cụ thể:

### 1. Chuẩn hóa Schema các file CSV Logs (Đầy đủ tham số tự chứa - Self-contained)
* **Nhật ký chi tiết từng run (`logs/raw_runs_exp_{A,B,C}.csv`):** Đảm bảo ghi nhận trọn vẹn toàn bộ tham số cố định và biến đổi để tái lập độc lập 100%:
  ```text
  run_id, exp_name, seed, param_name, param_value, num_layers, hidden_units, activation, lr, batch_size, max_epochs, patience, n_train, n_val, total_epochs, best_epoch, train_mse, val_mse, test_mse, train_time_sec, checkpoint_file, history_file
  ```
* **Bảng kết quả tổng hợp (`logs/exp_{A,B,C}_results.csv`):**
  ```text
  param_value, train_mse_mean, train_mse_std, val_mse_mean, val_mse_std, test_mse_mean, test_mse_std, best_epoch_mean, train_time_mean
  ```
* **Lịch sử loss theo epoch cho mọi run (`logs/history_exp_{A,B,C}_{param_name}_{param_val}_seed{seed}.csv`):**
  ```text
  epoch, train_loss, val_loss, epoch_time_sec
  ```
* **Kết quả Best Model trên tập Test (`logs/best_model_test_metrics.csv`):**
  ```text
  model_name, num_layers, hidden_units, activation, train_samples, mse, mae, rmse, r2, eval_time_sec
  ```

### 2. Quy ước đặt tên Checkpoint Model
* Checkpoint từng run: `checkpoints/exp_{A,B,C}_{param_name}_{param_val}_seed{seed}.pth`
* Checkpoint mô hình tối ưu cuối cùng: `checkpoints/best_model.pth`

### 3. Chuẩn trả về của Pipeline huấn luyện (`train_mlp_pipeline`)
Hàm `train_mlp_pipeline` do TV2 cài đặt bắt buộc trả về một `Dict[str, Any]` có cấu trúc cố định:
```python
{
    "model": model,                   # Mô hình PyTorch đã nạp lại trọng số tốt nhất
    "best_epoch": int,                # Epoch đạt validation loss thấp nhất
    "best_val_loss": float,           # Giá trị validation loss tốt nhất
    "total_epochs": int,              # Tổng số epoch đã chạy trước khi dừng
    "history": pd.DataFrame,          # DataFrame chứa cột: ['epoch', 'train_loss', 'val_loss', 'epoch_time_sec']
    "checkpoint_path": str            # Đường dẫn file checkpoint .pth đã lưu
}
```

---

## IV. MA TRẬN ĐỐI CHIẾU RUBRIC ĐÁNH GIÁ (RUBRIC TRACEABILITY MATRIX - 100 ĐIỂM)

Nhóm cam kết đối soát 100% minh chứng thực tế theo đúng Rubric chấm điểm của giảng viên:

| Tiêu chí Rubric | Điểm | Yêu cầu minh chứng cụ thể | TV chịu trách nhiệm chính | Minh chứng kiểm tra |
|:---|:---:|:---|:---:|:---|
| **1. Dữ liệu & bài toán** | **10** | Đúng công thức Gaussian 2D ($\boldsymbol{\mu}, C$), miền $[-2,2]\times[-4,4]$ (5đ); chia tập và seed hợp lý, tránh rò rỉ (3đ); trực quan hóa 3D surface & scatter (2đ). | **TV1** | `src/data_generator.py`, `figures/ground_truth_surface.png`, `figures/test_3d_scatter.png`, Mục 1 & 2 Báo cáo |
| **2. Triển khai PyTorch** | **25** | MLP tùy biến đúng bằng `nn.Module` (8đ); vòng lặp train/backprop/optimizer chuẩn (8đ); validation/checkpoint/early stopping theo val loss (5đ); cấu trúc code, README, chạy lại được (4đ). | **TV1, TV2** | `src/models.py`, `src/trainer.py`, `src/metrics.py`, `README.md`, Mục 3 Báo cáo |
| **3. Thực nghiệm A** | **10** | Cấu hình baseline công bằng (3đ); đầy đủ kết quả bảng/đồ thị theo $A \in [200, 2000]$ lồng nhau (5đ); phân tích ảnh hưởng số mẫu (2đ). | **TV2** | `src/run_experiment_A.py`, `logs/exp_A_results.csv`, `logs/raw_runs_exp_A.csv`, `figures/exp_A_samples_vs_error.png`, Mục 4.1 Báo cáo |
| **4. Thực nghiệm B** | **10** | Cấu hình baseline công bằng (3đ); đầy đủ bảng/đồ thị $B \in [1, 64]$ (5đ); phân tích hiện tượng thiếu khớp (underfitting) và quá khớp (overfitting) (2đ). | **TV3** | `src/run_experiment_B.py`, `logs/exp_B_results.csv`, `logs/raw_runs_exp_B.csv`, `figures/exp_B_width_vs_error.png`, Mục 4.2 Báo cáo |
| **5. Thực nghiệm C** | **10** | Cấu hình baseline công bằng, cố định 32 units/tầng (3đ); bảng/đồ thị test error, loss hội tụ, thời gian huấn luyện (5đ); phân tích độ sâu (2đ). | **TV3** | `src/run_experiment_C.py`, `logs/exp_C_results.csv`, `logs/raw_runs_exp_C.csv`, `figures/exp_C_depth_vs_error.png`, `figures/exp_C_depth_vs_time.png`, `figures/exp_C_loss_convergence.png`, Mục 4.3 Báo cáo |
| **6. Đánh giá & phân tích chung** | **10** | Đầy đủ MSE và MAE/RMSE trên tập test 100 mẫu độc lập, biểu đồ Parity $y$ vs $\hat{y}$ & residuals, đối sánh 3D (5đ); diễn giải, thẩm định validation đa-seed có fallback, thảo luận giới hạn (5đ). | **TV4** | `src/evaluate_best_model.py`, `logs/best_model_test_metrics.csv`, `figures/best_model_parity_and_residuals.png`, `figures/best_model_3d_comparison.png`, `figures/best_model_learning_curve.png`, Mục 5 Báo cáo |
| **7. Báo cáo viết** | **15** | Đầy đủ bố cục 10–20 trang, rõ phương pháp & tham số (5đ); hình/bảng và số liệu truy vết được 100% với logs CSV (5đ); trình bày học thuật, trích dẫn tài liệu tham khảo đúng chuẩn (5đ). | **TV4 chủ trì** + Cả nhóm cùng viết | `reports/BTL01_BaoCao_MLP.docx` & `.pdf` |
| **8. Slide & thuyết trình** | **10** | Slide 12–15 trang trực quan, rõ trục/nhãn (4đ); thuyết trình & kịch bản phân vai 4 thành viên (3đ); sẵn sàng demo code và trả lời câu hỏi phản biện (3đ). | **TV4 chủ trì** + Cả nhóm cùng làm | `slides/BTL01_ThuyetTrinh_MLP.pptx` & `.pdf` |
| **TỔNG CỘNG** | **100** | | | |

---

## V. TÁI CÂN BẰNG PHÂN CÔNG NHIỆM VỤ (25% THỰC CHẤT CHO MỖI THÀNH VIÊN)

Mỗi thành viên chịu trách nhiệm 25% khối lượng cả về mã nguồn lẫn báo cáo và slide.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TỔNG QUAN PHÂN BỔ KHỐI LƯỢNG (100%)                              │
├───────────────────┬───────────────────┬───────────────────┬────────────────────────────┤
│ Thành viên 1: 25% │ Thành viên 2: 25% │ Thành viên 3: 25% │     Thành viên 4: 25%      │
│  Data & Models    │  Train & Exp A    │    Exp B & C      │ Eval, Lead Editor & Slides │
└───────────────────┴───────────────────┴───────────────────┴────────────────────────────┘
```

### 1. THÀNH VIÊN 1: Data Pipeline, Kiến Trúc Mạng & Phần Mở Đầu Báo Cáo (25%)
* **Nhiệm vụ Kỹ thuật:**
  - Viết module `src/data_generator.py`:
    + Cài đặt hàm mật độ xác suất Gaussian 2D với $\boldsymbol{\mu} = [0, 0]^T$ và $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$.
    + Sinh ngẫu nhiên mẫu trong miền $x_1 \in [-2, 2], x_2 \in [-4, 4]$.
    + Sinh và lưu cố định: `data/test_dataset_100.csv` (100 mẫu, seed 42) và `data/train_val_pool.csv` (5000 mẫu, seed 2024).
    + Cài đặt hàm tiện ích chuẩn hóa phân chia tập dữ liệu:  
      `get_train_val_split(pool_X: np.ndarray, pool_y: np.ndarray, n_train: int, n_val: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  - Viết module `src/models.py`:
    + Cài đặt class `GaussianMLP` kế thừa `nn.Module`.
    + Hỗ trợ cấu hình linh hoạt số tầng ẩn $C$, số nơ-ron $B$, hàm kích hoạt `ReLU` (cùng `LeakyReLU`, `Tanh`, `Sigmoid`).
    + Khởi tạo trọng số Kaiming/Xavier chuẩn.
  - Viết các hàm 3D ban đầu trong `src/visualize.py`:
    + `plot_3d_scatter` $\to$ xuất `figures/test_3d_scatter.png`.
    + `plot_ground_truth_surface` $\to$ xuất `figures/ground_truth_surface.png`.
* **Nhiệm vụ Báo cáo & Slide:**
  - Viết **Mục 1 (Giới thiệu & Phát biểu bài toán)** và **Mục 2 (Dữ liệu & Khảo sát ban đầu)** trong Báo cáo Word (khoảng 3–4 trang).
  - Thiết kế **Slide 1, 2, 3, 4** (Tiêu đề nhóm, Mục tiêu bài toán, Dữ liệu 3D, Kiến trúc mạng MLP).

---

### 2. THÀNH VIÊN 2: Training Pipeline, Thiết lập Baseline & Thực Nghiệm A (25%)
* **Nhiệm vụ Kỹ thuật:**
  - Viết module `src/metrics.py`: Cài đặt các hàm đo lường sai số MSE, MAE, RMSE, R2.
  - Viết module `src/trainer.py`:
    + Đóng gói dữ liệu bằng `DataLoader` và `TensorDataset`.
    + Vòng lặp train (`model.train()`, `zero_grad()`, `backward()`, `step()`).
    + Vòng lặp validation (`model.eval()`, `torch.no_grad()`).
    + Lớp `EarlyStopping` tự động lưu checkpoint model theo val loss.
    + `train_mlp_pipeline` ghi nhận đầy đủ loss từng epoch ra DataFrame `history` và lưu file CSV lịch sử cho mọi run.
  - Viết script `src/run_experiment_A.py`:
    + Thiết lập cấu hình Baseline chuẩn (Adam, lr=0.005, batch 32, epochs 150, 1 layer 64 units, ReLU).
    + Khảo sát 10 mốc $A \in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]$.
    + Áp dụng quy tắc **tập con lồng nhau (Nested Subsets)**: trích mẫu theo chỉ số cố định `iloc[400 : 400+A]`.
    + Chạy 3 seeds `[42, 100, 2024]`, tính $\text{Mean} \pm \text{Std}$.
    + Lưu lịch sử loss per-epoch cho mọi run: `logs/history_exp_A_samples_{A}_seed{seed}.csv`.
    + Xuất kết quả tổng hợp ra `logs/exp_A_results.csv`, lưu chi tiết từng lần chạy vào `logs/raw_runs_exp_A.csv` theo schema tự chứa và vẽ đồ thị `figures/exp_A_samples_vs_error.png` (sử dụng hàm từ TV4 bàn giao).
* **Nhiệm vụ Báo cáo & Slide:**
  - Viết **Mục 3 (Kiến trúc mô hình & Thiết lập huấn luyện)** và **Mục 4.1 (Chi tiết Thực nghiệm A: Quy mô dữ liệu)** trong Báo cáo Word.
  - Thiết kế **Slide 5, 6** (Pipeline huấn luyện/Baseline và Kết quả Thực nghiệm A kèm nhận xét).

---

### 3. THÀNH VIÊN 3: Thực Nghiệm B (Độ Rộng) & Thực Nghiệm C (Độ Sâu & Hội Tụ) (25%)
* **Nhiệm vụ Kỹ thuật:**
  - Viết script `src/run_experiment_B.py`:
    + Cố định tập train ($N = 1500$, chỉ số `iloc[400:1900]`) và tập val ($N = 400$, chỉ số `iloc[0:400]`).
    + Khảo sát số nơ-ron tầng ẩn $B \in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]$ (mạng 1 tầng ẩn).
    + Chạy 3 seeds `[42, 100, 2024]`, lưu history mọi run `logs/history_exp_B_width_{B}_seed{seed}.csv`.
    + Xuất kết quả tổng hợp ra `logs/exp_B_results.csv`, nhật ký chi tiết từng seed ra `logs/raw_runs_exp_B.csv` và vẽ đồ thị `figures/exp_B_width_vs_error.png`.
    + Thu thập dữ liệu phân tích hiện tượng thiếu khớp (Underfitting khi $B=1, 2$) và quá khớp (Overfitting khi $B$ lớn).
  - Viết script `src/run_experiment_C.py`:
    + Cố định tập train ($N = 1500$, chỉ số `iloc[400:1900]`) và tập val ($N = 400$, chỉ số `iloc[0:400]`).
    + Cố định mỗi tầng ẩn có **32 nơ-ron**, khảo sát số tầng ẩn $C \in [1, 2, 3, 4, 5]$.
    + Đo lường: Train MSE, Validation MSE, Test MSE (đáp ứng bắt buộc yêu cầu bảng/đồ thị test error nhóm C theo đề bài), epoch hội tụ và **Thời gian huấn luyện (Training Time tính bằng giây)** cho từng $C$.
    + Lưu lại lịch sử loss theo epoch cho TẤT CẢ các run `logs/history_exp_C_depth_{C}_seed{seed}.csv`.
    + Chạy 3 seeds `[42, 100, 2024]`, xuất kết quả tổng hợp ra `logs/exp_C_results.csv` và nhật ký chi tiết từng seed ra `logs/raw_runs_exp_C.csv`.
    + Vẽ 3 đồ thị: Sai số theo $C$ (`figures/exp_C_depth_vs_error.png`), Biểu đồ cột thời gian huấn luyện (`figures/exp_C_depth_vs_time.png`) và **Đồ thị đường cong hội tụ loss qua các epoch** (`figures/exp_C_loss_convergence.png`).
* **Nhiệm vụ Báo cáo & Slide:**
  - Viết **Mục 4.2 (Chi tiết Thực nghiệm B: Độ rộng tầng ẩn)** và **Mục 4.3 (Chi tiết Thực nghiệm C: Độ sâu & Thời gian huấn luyện)** trong Báo cáo Word.
  - Thiết kế **Slide 7, 8** (Kết quả Thực nghiệm B & C, đồ thị so sánh, đường cong hội tụ loss và phân tích thời gian huấn luyện).

---

### 4. THÀNH VIÊN 4: Đánh Giá Best Model, Tiện Ích Trực Quan Hóa, Trưởng Ban Biên Tập (25%)
* **Nhiệm vụ Kỹ thuật:**
  - **Hoàn thành ngay Giai đoạn 1** các hàm tiện ích trực quan hóa trong `src/visualize.py`:
    + `plot_experiment_trend`: vẽ đường xu hướng kèm dải $\pm \text{std}$ (bàn giao ngay cho TV2/TV3 dùng trong GĐ 2).
    + `plot_learning_curve`: vẽ đường cong loss train/val qua các epoch.
    + `plot_parity_and_residuals`: biểu đồ Parity $y$ vs $\hat{y}$ và Residual plot.
    + `plot_3d_model_comparison`: đối sánh bề mặt 3D thực tế vs mô hình dự đoán.
  - Viết script `src/evaluate_best_model.py`:
    + Chạy thẩm định ứng viên cấu hình trên cả **3 seeds `[42, 100, 2024]`** trên tập Validation 400 mẫu cố định.
    + Đọc benchmark từ logs/exp_{A,B,C}_results.csv; so sánh công bằng với benchmark từ A/B/C; áp dụng **cơ chế Fallback** nếu ứng viên không tốt hơn.
    + Chọn checkpoint đại diện theo quy tắc duy nhất (gần mean nhất, hòa chọn seed nhỏ nhất) để đánh giá kiểm thử ngoại kiểm lần cuối trên **tập Test 100 mẫu độc lập** (tuyệt đối không dùng test để tune).
    + Xuất kết quả ra `logs/best_model_test_metrics.csv` và lưu lịch sử loss ra `logs/history_best_model.csv`.
    + Vẽ đường cong học tập Best Model: `figures/best_model_learning_curve.png`.
    + Vẽ biểu đồ Parity & Residuals: `figures/best_model_parity_and_residuals.png`.
    + Vẽ biểu đồ đối sánh 3D thật vs xấp xỉ: `figures/best_model_3d_comparison.png`.
* **Nhiệm vụ Báo cáo, Slide & Đóng gói:**
  - Viết **Mục 5 (Đánh giá mô hình tối ưu & Thảo luận sâu)** và **Mục 6 (Kết luận & Tài liệu tham khảo)** trong Báo cáo Word (khoảng 3–4 trang).
  - Thu thập nội dung từ TV1, TV2, TV3 để tổng hợp, định dạng thống nhất toàn bộ file Báo cáo Word/PDF (10–20 trang, kiểm tra số thứ tự Hình, Bảng và công thức toán).
  - Thiết kế **Slide 9, 10, 11, 12** (Best Model metrics, 3D surface comparison, Thảo luận & Kết luận) và tổng hợp bộ Slide PowerPoint hoàn chỉnh.
  - Viết hoàn thiện `README.md`, kiểm tra `main.ipynb` và đóng gói file nén `BTL01_Nhom_HoTen_MSSV.zip`.

---

## VI. TIẾN ĐỘ THỰC HIỆN THEO 4 GIAI ĐOẠN (TIMELINE)

1. **Giai đoạn 1 (Ngày 1 - Ngày 3): Hoàn thiện Core Pipeline, Tiện ích Trực quan & Dữ liệu**
   * TV1: Viết `data_generator.py` (sinh 2 file CSV vào `data/`, cung cấp hàm chia tập `get_train_val_split`), viết `models.py` và các hàm vẽ 3D trong `visualize.py`.
   * TV2: Viết `metrics.py`, `trainer.py` (chuẩn hóa `history` per-epoch cho mọi run), test vòng lặp với mô hình baseline.
   * **TV4 hoàn thiện ngay tiện ích trực quan hóa trong `visualize.py` (`plot_experiment_trend`, `plot_learning_curve`) để TV2 và TV3 có sẵn công cụ vẽ đồ thị cho Giai đoạn 2.**
   * TV1 & TV2 viết nháp Mục 1, 2, 3 Báo cáo.
2. **Giai đoạn 2 (Ngày 4 - Ngày 6): Thực thi 3 Nhóm Thực nghiệm A, B, C**
   * TV2: Chạy `run_experiment_A.py` (tập con lồng nhau, 3 seeds), xuất raw logs tự chứa, summary và hình vẽ sai số. Viết Mục 4.1 Báo cáo & Slide 5, 6.
   * TV3: Chạy `run_experiment_B.py` và `run_experiment_C.py` (đo thời gian, lưu history mọi run, xuất đồ thị hội tụ loss), xuất logs và hình vẽ. Viết Mục 4.2, 4.3 Báo cáo & Slide 7, 8.
3. **Giai đoạn 3 (Ngày 7 - Ngày 8): Thẩm định Best Model & Tổng Hợp Tài Liệu**
   * TV4: Chạy `evaluate_best_model.py` (thẩm định 3 seeds trên validation, fallback nếu cần, đánh giá test set 100 mẫu), xuất biểu đồ Parity, residuals, 3D comparison và learning curve. Viết Mục 5, 6 Báo cáo.
   * TV4 tập hợp các phần viết của TV1, TV2, TV3, đồng bộ định dạng file Báo cáo Word và ghép Slide hoàn chỉnh.
4. **Giai đoạn 4 (Ngày 9 - Ngày 10): Tái lập độc lập, Rà soát & Đóng gói**
   * Cả 4 thành viên kiểm tra chéo (Peer Review), chạy thử từ đầu trên máy khác theo `README.md`.
   * TV4 đóng gói file nén `BTL01_Nhom_HoTen_MSSV.zip`. Nhóm tổng duyệt thuyết trình thử 10–15 phút.
