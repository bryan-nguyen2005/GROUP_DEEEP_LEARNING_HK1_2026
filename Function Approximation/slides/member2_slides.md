# NỘI DUNG SLIDE THUYẾT TRÌNH - PHẦN PHỤ TRÁCH CỦA THÀNH VIÊN 2
**Vai trò:** Training Pipeline & Exp A Lead  
**Slide phụ trách:** Slide 5 & Slide 6

---

## SLIDE 5: PIPELINE HUẤN LUYỆN & THIẾT LẬP BASELINE CHUẨN

### 1. Thiết kế Pipeline Huấn luyện Chuẩn hóa (PyTorch)
* **Đóng gói dữ liệu:** `TensorDataset` + `DataLoader` (mini-batch $32$, shuffle tập train, cố định validation/test).
* **Vòng lặp tối ưu hóa:** 
  $$\text{Forward} \to \text{MSE Loss} \to \text{zero\_grad()} \to \text{loss.backward()} \to \text{optimizer.step()}$$
* **Cơ chế Dừng sớm (Early Stopping) & Checkpoint:**
  - Theo dõi liên tục `val_loss` sau mỗi epoch.
  - Ngưỡng kiên nhẫn: $patience = 15$ epochs (nếu không giảm thì dừng sớm để tránh Overfitting).
  - Tự động lưu checkpoint trọng số tốt nhất (`.pth`) và nạp lại khi đánh giá.
* **Ghi nhận lịch sử (Loss History per-epoch):** Xuất toàn bộ đường cong học tập ra CSV (`logs/history_*.csv`).

### 2. Cấu hình Baseline Chuẩn Cố Định (Hệ quy chiếu cho Exp A, B, C)
* **Kiến trúc:** MLP 1 tầng ẩn, $64$ nơ-ron (`num_hidden_layers=1, hidden_units=64`).
* **Hàm kích hoạt:** `ReLU` ở tầng ẩn, Tầng ra Tuyến tính (Linear).
* **Bộ tối ưu (Optimizer):** `Adam` với tốc độ học $\eta = 0.005$.
* **Môi trường thực thi:** Thuần CPU (`torch.device('cpu')`), tối đa $150$ epochs.

### 3. Bộ Chỉ số Đo lường Đánh giá (`src/metrics.py`)
* $\text{MSE}$ (Mean Squared Error - hàm mục tiêu chính).
* $\text{MAE}$ (Mean Absolute Error) & $\text{RMSE}$ (Root Mean Squared Error).
* $R^2$ Score (Hệ số xác định phương sai).

---

## SLIDE 6: KẾT QUẢ & PHÂN TÍCH THỰC NGHIỆM A (QUY MÔ DỮ LIỆU)

### 1. Phương pháp Thực nghiệm Nghiêm ngặt (Zero Data Leakage)
* **Khảo sát:** 10 mốc kích thước mẫu $A \in [200, 400, \dots, 2000]$.
* **Tập con lồng nhau (Nested Subsets):** 
  $$D_{200} \subset D_{400} \subset \dots \subset D_{2000}$$
  - Trích xuất cố định `iloc[400 : 400+A]`, không bao giờ giao thoa với 400 mẫu Validation.
* **Giao thức Đa-seed:** Chạy trên cả 3 seeds cố định `[42, 100, 2024]` (tổng cộng 30 runs, đo $\text{Mean} \pm \text{Std}$).

### 2. Bảng Kết quả Thực nghiệm & Biểu đồ Xu hướng
*(Chèn hình đồ thị `figures/exp_A_samples_vs_error.png` vào slide)*

| Quy mô mẫu $A$ | Test MSE ($\times 10^{-4}$) | Validation MSE ($\times 10^{-4}$) | Nhận xét |
|:---:|:---:|:---:|:---|
| **$A = 200$** | $16.30 \pm 2.16$ | $17.84 \pm 4.50$ | Sai số cao nhất, gap giữa train và val lớn |
| **$A = 600$** | $8.51 \pm 5.20$ | $8.40 \pm 5.02$ | Sai số giảm mạnh gần $50\%$ |
| **$A = 1200$** | $5.80 \pm 6.04$ | $6.07 \pm 6.57$ | Hội tụ ổn định, gap tiệm cận 0 |
| **$A = 2000$** | **$1.26 \pm 0.60$** | **$1.43 \pm 0.69$** | Sai số giảm hơn **12.9 lần**, phương sai cực nhỏ |

### 3. Kết luận Chuyên môn
1. **Dữ liệu là chìa khóa:** Kích thước mẫu tăng giúp Test MSE giảm hơn một bậc độ lớn (từ $0.00163$ về $0.000126$).
2. **Khả năng tổng quát hóa:** Khi $A \ge 1400$, khoảng cách giữa Train Loss và Val/Test Loss gần như triệt tiêu.
3. **Cơ sở cho Exp B & C:** Mốc $N = 1500$ là kích thước mẫu lý tưởng, được lựa chọn cố định cho các thực nghiệm khảo sát độ rộng và độ sâu tiếp theo.
