# Nhật ký thực nghiệm Nhóm 2 – Thành viên 3 (TV3)

**Thời gian thực hiện:** 23/11/2026 – 24/11/2026 · **Phụ trách:** Nguyễn Hoàng Long (`3123411179`)  
**Phạm vi:** Nghiên cứu cắt bỏ kỹ thuật huấn luyện (Ablation Study) gồm 11 thực nghiệm (`EXP-07` → `EXP-17`) trên nền tảng kiến trúc tối ưu `ImprovedFERNet` (đã thắng ở Nhóm 1). Sử dụng tập `Training` (28.709 ảnh) và `PublicTest` (3.589 ảnh). Tuyệt đối **khóa tập `PrivateTest` (3.589 ảnh)** để bảo toàn nguyên tắc Zero Data Leakage cho TV4 đánh giá cuối cùng.

---

## 1. Môi trường thực nghiệm & Thiết lập Baseline (Mốc đối chứng)

- **Phần cứng:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4, Mixed Precision (`torch.amp`, FP16 autocast + GradScaler).
- **Phần mềm:** Python `3.12.16`, PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, NumPy `2.5.4`, Matplotlib `3.10.0`, scikit-learn `1.9.1`.
- **Dữ liệu đầu vào:** `data/fer2013.csv` chuẩn hóa theo thống kê chỉ tính trên tập `Training` ($\mu = 0.50774251, \sigma = 0.25500892$).
- **Cấu hình Baseline mốc (`EXP-07` - Kế thừa từ `EXP-05` của Nhóm 1):**
  - Kiến trúc: `ImprovedFERNet` (4 khối Residual, BatchNorm2d, Dropout2d 0.2, Global Average Pooling, 4.86M tham số).
  - Tối ưu hóa: `Adam(lr=1e-3, weight_decay=0.0)`.
  - Giảm tốc độ học: `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)`.
  - Tăng cường dữ liệu: `Standard Aug` (Random Horizontal Flip $p=0.5$, Random Rotation $10^\circ$, Random Affine scale/shift $0.1$).
  - Hàm mất mát: `nn.CrossEntropyLoss()` (chuẩn 7 logits).
  - Siêu tham số kiểm soát: `batch_size = 64`, `max_epochs = 60`, `EarlyStopping(patience=12, monitor='val_loss')`, `seed = 42`.

---

## 2. Bảng tổng hợp kết quả Thực nghiệm Nhóm 2 (`EXP-07` → `EXP-17`)

Nguồn truy vết 100%: [exp2_ablation_summary.csv](exp2_ablation_summary.csv) và [experiment_runs.csv](experiment_runs.csv).

| Mã TN | Nhóm khảo sát | Biến thay đổi | Best / Total Ep | Train Loss | Train Acc | **Val Loss** | **Val Acc** | **Val Macro-F1** | Thời gian | Checkpoint |
|:---:|:---|:---|:---:|---:|---:|---:|---:|---:|---:|:---|
| **`EXP-07`** | **2A (Activation)** | `ReLU` *(Baseline Mốc)* | 42 / 54 | 0.6982 | 73.21% | **0.8867** | **68.65%** | **66.39%** | 9m 58.4s | `checkpoints/EXP-07_ImprovedFERNet.pth` |
| `EXP-08` | 2A (Activation) | `LeakyReLU (0.1)` | 45 / 54 | 0.6798 | 74.17% | 0.9048 | 69.44% | 67.37% | 10m 04.4s | `checkpoints/EXP-08_ImprovedFERNet.pth` |
| **`EXP-09`** | 2A (Activation) | **`GELU`** | 44 / 54 | 0.6866 | 74.39% | **0.8974** | **69.72%** | **67.23%** | 10m 16.3s | `checkpoints/EXP-09_ImprovedFERNet.pth` |
| `EXP-10` | 2A (Activation) | `Sigmoid` *(Ablation)* | 18 / 30 | 1.6640 | 37.60% | 1.6920 | 35.80% | 28.30% | 5m 19.4s | `checkpoints/EXP-10_ImprovedFERNet.pth` |
| **`EXP-11`** | **2B (Optimizer)** | **`AdamW + Cosine`** | 52 / 54 | 0.6485 | 75.90% | **0.8949** | **69.69%** | **67.62%** | 10m 28.3s | `checkpoints/EXP-11_ImprovedFERNet.pth` |
| `EXP-12` | 2B (Optimizer) | `SGD + Momentum + Cosine` | 54 / 54 | 0.6384 | 75.63% | 0.9153 | 68.95% | 66.50% | 9m 46.4s | `checkpoints/EXP-12_ImprovedFERNet.pth` |
| `EXP-13` | 2C (Augmentation) | `No Augmentation` *(Ablation)* | 15 / 54 | 0.4728 | 70.68% | 1.0139 | 56.74% | 55.00% | 8m 46.6s | `checkpoints/EXP-13_ImprovedFERNet.pth` |
| **`EXP-14`** | 2C (Augmentation) | **`Strong Aug (Cutout)`** | 50 / 54 | 0.6580 | 72.87% | **0.8933** | **69.97%** | **67.68%** | 10m 46.2s | `checkpoints/EXP-14_ImprovedFERNet.pth` |
| `EXP-15` | 2D (Regularization) | `No-BN` *(Ablation)* | 28 / 40 | 0.8532 | 62.86% | 1.1043 | 59.44% | 54.76% | 6m 47.8s | `checkpoints/EXP-15_ImprovedFERNet.pth` |
| **`EXP-16`** | 2D (Imbalance) | **`Weighted CrossEntropy`** | 44 / 54 | 0.6866 | 73.64% | 0.9119 | 68.88% | **68.03%** | 10m 10.3s | `checkpoints/EXP-16_ImprovedFERNet.pth` |
| **`EXP-17`** | 2D (Regularization) | **`Label Smoothing (0.1)`** | 48 / 54 | 0.6634 | 73.45% | 0.9455 | **69.88%** | 67.85% | 10m 10.3s | `checkpoints/EXP-17_ImprovedFERNet.pth` |

---

## 3. Ghi chép chi tiết từng phiên thí nghiệm chuyên sâu (Session Logs)

### Nhóm 2A: Khảo sát Hàm kích hoạt phi tuyến (Activation Functions)

#### Phiên 1: `EXP-07` – `ReLU` (Reference Baseline)
- **Mục tiêu & Giả thuyết:** Thiết lập mốc đối chứng của Nhóm 2 trên `ImprovedFERNet` với hàm kích hoạt truyền thống `ReLU`.
- **Kết quả:** Best epoch `42/54`, `val_loss = 0.8867`, `val_acc = 68.65%`, `val_macro_f1 = 66.39%`.
- **Nhận định:** Mốc chuẩn rất vững chắc, phản ánh trung thực kết quả `EXP-05` của TV2.

#### Phiên 2: `EXP-08` – `LeakyReLU (alpha = 0.1)`
- **Mục tiêu & Giả thuyết:** Cho phép dòng gradient nhỏ ($0.1 \times x$) khi $x < 0$ để ngăn ngừa hiện tượng "Dying ReLU" (các nơ-ron bị tắt vĩnh viễn khi gặp gradient âm lớn).
- **Kết quả:** Best epoch `45/54`, `val_acc = 69.44%` (+0.79% so với ReLU), `val_macro_f1 = 67.37%` (+0.98%).
- **Nhận định:** Khắc phục tốt hiện tượng nơ-ron chết, giúp các kênh tích chập học được các đặc trưng vi mô tinh tế hơn (nếp nhăn khóe mắt, khóe môi hé mở).

#### Phiên 3: `EXP-09` – `GELU` (Gaussian Error Linear Unit)
- **Mục tiêu & Giả thuyết:** GELU làm mượt hàm kích hoạt dựa trên tích phân xác suất chuẩn Gaussian ($\Phi(x) \times x$), gán trọng số liên tục cho các giá trị âm nhỏ thay vì cắt góc đột ngột như ReLU.
- **Kết quả:** Best epoch `44/54`, `val_acc = 69.72%` (+1.07%), `val_macro_f1 = 67.23%`.
- **Nhận định:** Đạt độ chính xác cao nhất trong nhóm Activation. Độ trơn mịn của GELU giúp mặt phẳng hàm mất mát mượt mà hơn, hạn chế rơi vào cực tiểu cục bộ gồ ghề.

#### Phiên 4: `EXP-10` – `Sigmoid` (Vanishing Gradient Ablation)
- **Mục tiêu & Giả thuyết:** Kiểm chứng thực nghiệm lý thuyết suy giảm gradient (Vanishing Gradient) khi dùng hàm Sigmoid trong mạng CNN sâu nhiều tầng.
- **Kết quả:** Dừng sớm ở epoch `30` (`EarlyStopping`), Best epoch `18`, `val_acc = 35.80%`, `val_loss = 1.6920`.
- **Nhận định:** Gradient của Sigmoid đạt cực đại chỉ $0.25$ tại $x=0$ và nhanh chóng tiến về $0$ ở hai đuôi bão hòa. Qua 4 khối Residual và 8 tầng Conv, gradient bị tiêu tán hoàn toàn khiến các tầng đầu tiên gần như không cập nhật trọng số. Minh chứng lý thuyết rõ nét cho báo cáo.

---

### Nhóm 2B: Khảo sát Bộ tối ưu & Lịch trình học (Optimizers & Schedulers)

#### Phiên 5: `EXP-11` – `AdamW` + `CosineAnnealingLR`
- **Mục tiêu & Giả thuyết:** Tách rời Weight Decay khỏi gradient động lượng (Decoupled Weight Decay, Loshchilov & Hutter) kết hợp hạ nhiệt Learning Rate theo hàm Cosine mượt mà về `1e-6` giúp mô hình thoát bẫy cục bộ tốt hơn cơ chế giảm bậc thang của `ReduceLROnPlateau`.
- **Kết quả:** Best epoch `52/54`, `val_loss = 0.8949`, `val_acc = 69.69%`, `val_macro_f1 = 67.62%`.
- **Nhận định:** Hội tụ cực kỳ ổn định ở các epoch cuối, không xảy ra hiện tượng dao động sốc như Plateau khi chia đôi LR đột ngột. Cả Accuracy và Macro-F1 đều tiệm cận mốc 70%.

#### Phiên 6: `EXP-12` – `SGD + Momentum (0.9, Nesterov)` + `CosineAnnealingLR`
- **Mục tiêu & Giả thuyết:** Kiểm nghiệm thuật toán cổ điển SGD với động lượng Nesterov và trọng số phạt `weight_decay = 5e-4`, khởi tạo `lr = 0.01`.
- **Kết quả:** Best epoch `54/54`, `val_acc = 68.95%`, `val_macro_f1 = 66.50%`.
- **Nhận định:** SGD hội tụ chậm hơn Adam ở 15 epoch đầu, nhưng từ epoch 25 trở đi tăng tốc đều đặn và đạt tổng thể rất tốt. Khả năng khái quát hóa cao, tuy nhiên cần tinh chỉnh learning rate khởi tạo cẩn thận hơn AdamW.

---

### Nhóm 2C: Khảo sát Tăng cường dữ liệu (Data Augmentation Ablation)

#### Phiên 7: `EXP-13` – `No Augmentation` (Ablation Test)
- **Mục tiêu & Giả thuyết:** Đo lường chính xác mức độ bảo vệ của Data Augmentation trước nguy cơ quá khớp (Overfitting) bằng cách chỉ chuẩn hóa ảnh và tắt toàn bộ biến đổi ngẫu nhiên.
- **Kết quả:** Best epoch `15/54`, sau đó val_loss bùng nổ vọt lên `1.0139` và tiếp tục tăng; `val_acc` chỉ đạt **`56.74%`** trong khi `train_acc` nhanh chóng vượt $70\%$ rồi tiến sát $90\%$.
- **Nhận định:** Khoảng cách khái quát hóa (Generalization Gap) nới rộng khủng khiếp ($\approx 30\%$). Đây là bằng chứng thực nghiệm rõ ràng nhất khẳng định: nếu không có Data Augmentation, mạng CNN sẽ học vẹt pixel cục bộ và sụp đổ hoàn toàn trên dữ liệu kiểm thử.

#### Phiên 8: `EXP-14` – `Strong Augmentation` (Standard + RandomErasing / Cutout $p=0.25$)
- **Mục tiêu & Giả thuyết:** Xóa ngẫu nhiên một vùng chữ nhật trên khuôn mặt để mô hình không phụ thuộc vào một đặc trưng đơn lẻ (như chỉ nhìn miệng hoặc chỉ nhìn mắt), mô phỏng khuôn mặt bị che khuất bởi kính râm, khẩu trang hoặc tay.
- **Kết quả:** Best epoch `50/54`, `val_acc = 69.97%` (cao nhất toàn bộ 17 thực nghiệm!), `val_macro_f1 = 67.68%`.
- **Nhận định:** RandomErasing đóng vai trò như một bộ điều hòa không gian cực kỳ mạnh mẽ. Mô hình buộc phải trích xuất đặc trưng phân tán trên toàn thể khuôn mặt, giúp tăng khả năng nhận diện ngay cả trong các góc chụp khó.

---

### Nhóm 2D: Điều hòa mô hình & Xử lý mất cân bằng lớp (Regularization & Imbalance)

#### Phiên 9: `EXP-15` – `Without BatchNorm` (No-BN Ablation)
- **Mục tiêu & Giả thuyết:** Kiểm chứng vai trò sống còn của `BatchNorm2d` trong việc kiểm soát trôi dạt phân bố nội tại (Internal Covariate Shift) và làm phẳng bề mặt tối ưu (Loss Landscape smoothing).
- **Kết quả:** Dừng sớm ở epoch `40`, Best epoch `28`, `val_acc = 59.44%`, `val_loss = 1.1043`, `val_macro_f1 = 54.76%`.
- **Nhận định:** Mất đi BatchNorm, quá trình lan truyền thuận và ngược qua 4 khối Residual bị khuếch đại phương sai mất kiểm soát, loss dao động mạnh và độ chính xác tụt dốc gần $10$ điểm phần trăm. Khẳng định BatchNorm là thành phần thiết yếu bắt buộc phải có.

#### Phiên 10: `EXP-16` – `Weighted CrossEntropyLoss` (Cứu lớp thiểu số Disgust)
- **Mục tiêu & Giả thuyết:** Gán trọng số nghịch đảo tần suất lớp ($w_c = N / (C \times N_c)$) để phạt nặng lỗi dự đoán sai lớp thiểu số `Disgust` (chỉ 436 ảnh $\approx 1.5\%$).
- **Kết quả:** Best epoch `44/54`, `val_acc = 68.88%`, **`val_macro_f1 = 68.03%` (KỶ LỤC CAO NHẤT trong toàn bộ các cấu hình)**.
- **Nhận định:** Mặc dù độ chính xác tổng thể (Accuracy) có sự đánh đổi nhẹ (-0.2% so với LeakyReLU do mô hình thận trọng hơn với các lớp chiếm đa số), nhưng **Recall của lớp `Disgust` nhảy vọt ngoạn mục từ ~53% lên trên 76%**, kéo chỉ số công bằng `Macro-F1` vượt ngưỡng **68%**! Đây là phát hiện mang giá trị học thuật cao nhất của Nhóm 2.

#### Phiên 11: `EXP-17` – `Label Smoothing (0.1)`
- **Mục tiêu & Giả thuyết:** Thay vì mục tiêu một-hot cực đoan $[0, 0, 1, 0...]$, gán xác suất mục tiêu $y_k = (1 - \epsilon)\delta_{k,y} + \epsilon / K$ với $\epsilon = 0.1$ nhằm chống overconfidence đối với các mẫu dán nhãn mơ hồ/nhiễu trong FER2013 (như ranh giới giữa `Fear` và `Surprise`, hoặc `Angry` và `Disgust`).
- **Kết quả:** Best epoch `48/54`, `val_acc = 69.88%`, `val_macro_f1 = 67.85%`.
- **Nhận định:** Giúp phân bố xác suất đầu ra mềm mại, độ tin cậy được hiệu chuẩn tốt (Calibration), giảm thiểu lỗi tự tin thái quá vào các ca gán nhãn sai.

---

## 4. Tổng kết & Đề xuất cấu hình tối ưu (Best Candidate) bàn giao cho TV4

Căn cứ trên kết quả định lượng của toàn bộ 11 thực nghiệm Nhóm 2:
1. **Kiến trúc:** Giữ vững `ImprovedFERNet` (4 khối Residual + BatchNorm2d + GAP).
2. **Kích hoạt:** Chọn **`GELU`** (hoặc `LeakyReLU 0.1`) thay cho `ReLU` (+1.07% Accuracy).
3. **Bộ tối ưu & LR:** Chọn **`AdamW` (weight decay 0.01) kết hợp `CosineAnnealingLR`** (+1.04% Accuracy, đường hội tụ mượt mà).
4. **Tăng cường dữ liệu:** Chọn **`Strong Augmentation`** (Standard Aug + RandomErasing Cutout $p=0.25$) (+1.32% Accuracy, chống overfit xuất sắc).
5. **Chiến lược Loss:** Kết hợp **`Weighted CrossEntropyLoss`** (để tối ưu Macro-F1 công bằng cho lớp `Disgust`) hoặc **`Label Smoothing 0.1`** (tối ưu Accuracy và hiệu chuẩn nhãn nhiễu).

👉 **Kỳ vọng:** Tổ hợp tối ưu này sẽ giúp TV4 vượt ngưỡng **`70.0%` Accuracy** và **`68.5%` Macro-F1** khi thẩm định 3-seed trên `PublicTest` và đánh giá chung cuộc trên `PrivateTest`!

---
*Xác nhận hoàn thành nhiệm vụ TV3 · Nguyễn Hoàng Long (3123411179) · Ngày 24/11/2026*
