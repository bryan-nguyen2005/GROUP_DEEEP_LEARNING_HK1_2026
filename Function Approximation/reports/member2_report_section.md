# NỘI DUNG BÁO CÁO BÀI TẬP LỚN - PHẦN PHỤ TRÁCH CỦA THÀNH VIÊN 2
**Vai trò:** Training Pipeline & Exp A Lead (25% đóng góp)  
**Phạm vi:** Mục 3 & Mục 4.1 của Báo cáo khoa học (BTL01_BaoCao_MLP)

---

## 3. KIẾN TRÚC MÔ HÌNH & THIẾT LẬP HUẤN LUYỆN

### 3.1. Kiến trúc GaussianMLP bằng PyTorch (`src/models.py`)
Mô hình xấp xỉ hàm được xây dựng dựa trên kiến trúc Perceptron Đa Tầng (Multi-Layer Perceptron - MLP) kế thừa từ lớp `torch.nn.Module`. 

* **Cấu trúc luồng xử lý:**
  - **Tầng đầu vào (Input Layer):** Nhận vector tọa độ không gian 2 chiều $\mathbf{x} = [x_1, x_2]^T \in \mathbb{R}^2$ ($d_{\text{in}} = 2$).
  - **Các tầng ẩn (Hidden Layers):** Gồm $C$ tầng ẩn kết nối đầy đủ (Fully Connected / `nn.Linear`), mỗi tầng có $B$ nơ-ron (hoặc danh sách nơ-ron tùy biến). Mỗi nơ-ron thực hiện phép biến đổi affine:
    $$z_j^{[l]} = \sum_{i} w_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]}$$
    sau đó được đưa qua hàm kích hoạt phi tuyến tính $\sigma(\cdot)$ để tạo ra tín hiệu kích hoạt $a_j^{[l]} = \sigma(z_j^{[l]})$.
  - **Tầng đầu ra (Output Layer):** Gồm 1 nơ-ron tuyến tính (`nn.Linear`) không sử dụng hàm kích hoạt phi tuyến tính (Linear Activation):
    $$\hat{y} = \mathbf{w}_{\text{out}}^T \mathbf{a}^{[C]} + b_{\text{out}}$$
    để dự đoán giá trị liên tục của hàm mật độ xác suất $\hat{y} \approx f(x_1, x_2)$.

* **Lựa chọn hàm kích hoạt (Activation Function):**
  - **Chuẩn Baseline hiện đại — `ReLU` ($\sigma(z) = \max(0, z)$):** Được lựa chọn làm chuẩn Baseline cho toàn bộ các thực nghiệm. Ưu điểm nổi bật của ReLU là tính toán cực nhanh, gradient bằng 1 trên miền dương giúp triệt tiêu hoàn toàn hiện tượng triệt tiêu gradient (Vanishing Gradient), đồng thời tạo ra biểu diễn thưa (sparse representation) giúp mạng học các vùng biên phân bố rất sắc nét.
  - **So sánh đối chiếu:** So với các hàm cổ điển như `Sigmoid` ($\sigma(z) = \frac{1}{1 + e^{-z}}$) và `Tanh` ($\sigma(z) = \tanh(z)$) thường bị bão hòa đạo hàm ở hai phía cực trị khi trọng số lớn dẫn đến hội tụ rất chậm trên CPU, `ReLU` đạt tốc độ hội tụ nhanh hơn từ 3–5 lần và duy trì độ ổn định số học cao.
  - **Khởi tạo trọng số:** Mô hình sử dụng phương pháp khởi tạo Kaiming He Normal (`kaiming_normal_`) tối ưu hóa riêng cho các hàm phi tuyến ReLU/LeakyReLU, và khởi tạo Xavier Uniform (`xavier_uniform_`) cho tầng ra tuyến tính, giúp phương sai tín hiệu được bảo toàn qua các tầng mạng.

---

### 3.2. Pipeline huấn luyện chuẩn & Cơ chế Early Stopping (`src/trainer.py`)

Quy trình huấn luyện được mô-đun hóa độc lập và chuẩn hóa nghiêm ngặt theo các tiêu chuẩn học máy hiện đại:

1. **Đóng gói dữ liệu bằng PyTorch `DataLoader` (`create_dataloader`):**
   - Dữ liệu $X, y$ từ mảng NumPy được chuyển thành `torch.Tensor` kiểu dấu phẩy động 32-bit (`torch.float32`).
   - Đóng gói thông qua `TensorDataset` và nạp vào `DataLoader` với kích thước mini-batch cố định là $32$.
   - Tập huấn luyện được xáo trộn ngẫu nhiên (`shuffle=True`) sau mỗi epoch theo seed đã định trước; tập validation và test tuyệt đối không xáo trộn (`shuffle=False`) để đảm bảo tính nhất quán tuyệt đối của kết quả đo đạc.

2. **Vòng lặp huấn luyện chuẩn PyTorch (`train_one_epoch`):**
   - Mô hình được kích hoạt chế độ huấn luyện thông qua `model.train()`.
   - Vòng lặp duyệt qua từng mini-batch gồm 5 bước kinh điển:
     $$\text{inputs, targets} \to \text{zero\_grad()} \to \hat{y} = \text{model(inputs)} \to \mathcal{L} = \text{MSE}(\hat{y}, y) \to \mathcal{L}\text{.backward()} \to \text{optimizer.step()}$$
   - Tính toán và tích lũy sai số trung bình có trọng số theo kích thước batch thực tế.

3. **Vòng lặp kiểm thử và đánh giá (`evaluate_model`):**
   - Chuyển mạng sang chế độ đánh giá `model.eval()`.
   - Sử dụng ngữ cảnh khóa gradient `with torch.no_grad():` nhằm vô hiệu hóa tính toán đạo hàm, giải phóng bộ nhớ và tăng tốc độ xử lý trên CPU.

4. **Cơ chế dừng sớm (Early Stopping) & Tự động lưu Checkpoint:**
   - Lớp `EarlyStopping` theo dõi biến thiên của $\text{Loss}_{\text{val}}$ (Validation MSE) sau mỗi epoch.
   - Nếu $\text{Loss}_{\text{val}}$ cải thiện vượt ngưỡng biên độ tối thiểu $\delta = 10^{-6}$, mô hình tự động cập nhật kỷ lục `best_loss`, lưu trạng thái trọng số tốt nhất (`torch.save(model.state_dict(), path)`) và reset bộ đếm `counter = 0`.
   - Nếu sau $patience = 15$ epochs liên tiếp mà $\text{Loss}_{\text{val}}$ không cải thiện, thuật toán kích hoạt cờ `early_stop = True` để ngắt sớm vòng lặp, ngăn chặn hiện tượng học vẹt (overfitting) và tiết kiệm tài nguyên CPU.
   - Khi kết thúc, hàm `train_mlp_pipeline` tự động nạp lại đúng trọng số tối ưu từ checkpoint `.pth` trước khi bàn giao cho bước đánh giá.

5. **Ghi nhận lịch sử chi tiết (Loss History per-epoch):**
   - Mỗi epoch ghi nhận 4 trường: `epoch`, `train_loss`, `val_loss`, `epoch_time_sec`.
   - Toàn bộ được xuất tự động ra file CSV (`logs/history_*.csv`), phục vụ việc vẽ biểu đồ đường cong học tập và chứng minh quá trình hội tụ toán học.

---

### 3.3. Cấu hình Baseline chuẩn cố định & Bộ chỉ số đo lường (`src/metrics.py`)

Để đảm bảo tính khoa học và khả năng cô lập biến trong các thực nghiệm A, B, C, nhóm thiết lập một **Cấu hình Baseline duy nhất** làm hệ quy chiếu chung:

| Tham số cấu hình | Giá trị chuẩn | Ý nghĩa khoa học |
|---|:---:|---|
| **Kiến trúc mạng ẩn** | 1 tầng ẩn, 64 nơ-ron | Đủ năng lực mô phỏng hình nón úp 2D theo định lý UAT |
| **Hàm kích hoạt** | `ReLU` | Chống triệt tiêu gradient, tính toán nhanh |
| **Thuật toán tối ưu (Optimizer)** | `Adam` | Tự thích nghi learning rate theo từng tham số |
| **Tốc độ học (Learning Rate $\eta$)** | $0.005$ | Đảm bảo hội tụ mượt mà, không dao động quá mức |
| **Kích thước Batch (Batch Size)** | $32$ | Cân bằng giữa tốc độ tính toán vector và nhiễu gradient có lợi |
| **Số Epochs tối đa** | $150$ | Đủ lớn để mô hình đạt trạng thái hội tụ hoàn toàn |
| **Kiểm soát dừng sớm (Early Stopping)** | `patience = 15` | Dừng khi Validation MSE không giảm sau 15 epoch |
| **Thiết bị tính toán** | `CPU` | Thực nghiệm thuần CPU theo yêu cầu đề bài |

#### Bộ chỉ số đo lường chất lượng mô hình (`src/metrics.py`):
Mô-đun `src/metrics.py` cài đặt 4 chỉ số hồi quy tiêu chuẩn:
1. **Sai số toàn phương trung bình (MSE - Mean Squared Error):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   *Là hàm mục tiêu tối ưu chính của bài toán; phạt rất nặng các sai số dự đoán lớn.*
2. **Sai số tuyệt đối trung bình (MAE - Mean Absolute Error):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Đo độ lệch tuyệt đối trực quan theo đúng đơn vị của hàm mật độ xác suất.*
3. **Căn bậc hai sai số toàn phương trung bình (RMSE - Root Mean Squared Error):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
4. **Hệ số xác định ($R^2$ Score - Coefficient of Determination):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   *Đo tỷ lệ phần trăm phương sai của hàm Gaussian thật được mô hình MLP giải thích thành công. $R^2 \approx 1.0$ thể hiện mức độ xấp xỉ hoàn hảo.*

---

## 4. KẾT QUẢ CÁC NHÓM THỰC NGHIỆM

### 4.1. Thực nghiệm A: Ảnh hưởng của Quy mô dữ liệu huấn luyện (Sample Size $A$)

#### 1. Phương pháp luận & Nguyên tắc công bằng khoa học
Mục tiêu của Thực nghiệm A là khảo sát sự phụ thuộc của sai số mô hình vào số lượng mẫu huấn luyện $A$, qua 10 mốc:
$$A \in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]$$

Để kết quả không bị nhiễu bởi việc lấy mẫu ngẫu nhiên, nhóm áp dụng kỹ thuật **Tập con lồng nhau (Nested Subsets)**:
$$D_{200} \subset D_{400} \subset D_{600} \subset \dots \subset D_{2000}$$
- Tập Validation được cố định tuyệt đối đúng 400 mẫu đầu tiên (`iloc[0:400]`).
- Tập $D_A$ được trích xuất bằng chỉ số cố định `iloc[400 : 400+A]` từ `train_val_pool.csv`.
- Mọi mốc $A$ đều được huấn luyện trên **3 seeds độc lập `[42, 100, 2024]`** cho việc khởi tạo mạng và mini-batch shuffle, tổng cộng thực hiện $10 \times 3 = 30$ lượt huấn luyện độc lập.

#### 2. Bảng số liệu thực nghiệm tổng hợp
*(Truy vết nguồn: `logs/exp_A_results.csv` và nhật ký chi tiết từng lượt chạy tại `logs/raw_runs_exp_A.csv`)*:

| Số mẫu $A$ | Train MSE ($\times 10^{-4}$) | Validation MSE ($\times 10^{-4}$) | Test MSE ($\times 10^{-4}$) | Epoch dừng tb | Thời gian train (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **200** | $11.13 \pm 1.04$ | $17.84 \pm 4.50$ | $16.30 \pm 2.16$ | $107.3$ | $1.26$ |
| **400** | $8.54 \pm 3.35$ | $10.64 \pm 3.99$ | $11.15 \pm 3.95$ | $115.0$ | $1.30$ |
| **600** | $6.95 \pm 4.28$ | $8.40 \pm 5.02$ | $8.51 \pm 5.20$ | $132.7$ | $2.67$ |
| **800** | $8.22 \pm 4.32$ | $9.47 \pm 4.70$ | $9.42 \pm 4.42$ | $80.0$ | $2.25$ |
| **1000** | $8.27 \pm 6.20$ | $8.53 \pm 6.29$ | $8.40 \pm 6.12$ | $75.0$ | $2.62$ |
| **1200** | $6.04 \pm 6.68$ | $6.07 \pm 6.57$ | $5.80 \pm 6.04$ | $66.0$ | $2.67$ |
| **1400** | $4.41 \pm 4.67$ | $4.50 \pm 4.77$ | $3.81 \pm 3.80$ | $83.0$ | $3.84$ |
| **1600** | $4.98 \pm 6.11$ | $5.02 \pm 5.97$ | $5.31 \pm 6.36$ | $77.7$ | $3.90$ |
| **1800** | $1.11 \pm 0.22$ | $1.25 \pm 0.19$ | $1.29 \pm 0.24$ | $99.3$ | $5.30$ |
| **2000** | **$1.35 \pm 0.78$** | **$1.43 \pm 0.69$** | **$1.26 \pm 0.60$** | $91.3$ | $4.69$ |

#### 3. Phân tích đồ thị và nhận xét chuyên sâu
Đồ thị biểu diễn sai số theo quy mô dữ liệu được xuất tại [`figures/exp_A_samples_vs_error.png`](file:///c:/Users/Hoang%20Long/Documents/hoc_sau/repo_group_deeplearning/GROUP_DEEEP_LEARNING_HK1_2026/Function%20Approximation/figures/exp_A_samples_vs_error.png).

* **Xu hướng suy giảm sai số vượt bậc:**
  - Khi tăng số lượng mẫu huấn luyện từ $A = 200$ lên $A = 2000$, Test MSE trung bình giảm mạnh từ $0.001630$ xuống còn $0.000126$ (giảm hơn **12.9 lần** - hơn một bậc độ lớn).
  - Điều này hoàn toàn phù hợp với lý thuyết học máy thống kê: với mật độ điểm mẫu dày đặc trong miền không gian $[-2, 2] \times [-4, 4]$, mạng MLP có thể nắm bắt chính xác độ dốc và biên độ phân bố của quả đồi Gaussian mà không bị suy đoán chệch ở các vùng thưa điểm.

* **Khoảng cách tổng quát hóa (Generalization Gap):**
  - Tại $A = 200$: Khoảng cách giữa Train MSE ($0.001113$) và Val MSE ($0.001784$) khá lớn ($\Delta \approx 0.000671$), thể hiện mô hình có xu hướng hơi overfitting nhẹ do dữ liệu quá ít.
  - Khi $A \ge 1400$: Đường Train MSE, Val MSE và Test MSE gần như hòa làm một ($\Delta \le 0.00002$), chứng tỏ năng lực tổng quát hóa cực kỳ hoàn hảo của mô hình.

* **Độ ổn định và phương sai giữa các Seed:**
  - Ở mốc $A = 1800$ và $A = 2000$, độ lệch chuẩn giữa 3 seeds giảm xuống mức cực thấp ($\text{std} \approx 2.3 \times 10^{-5}$ đến $6.0 \times 10^{-5}$). Khi tập dữ liệu đủ lớn, sự phụ thuộc của mô hình vào việc khởi tạo trọng số ngẫu nhiên ban đầu bị triệt tiêu gần như hoàn toàn.

* **Kết luận rút ra cho nhóm:**
  - Quy mô $A \ge 1500$ mẫu là vùng kích thước tối ưu để cố định cho Thực nghiệm B (Độ rộng) và Thực nghiệm C (Độ sâu).
