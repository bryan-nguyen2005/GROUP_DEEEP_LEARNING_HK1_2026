# DÀN Ý BỘ SLIDE THUYẾT TRÌNH (12 SLIDES)
## ĐỀ TÀI: XẤP XỈ HÀM MẬT ĐỘ GAUSSIAN 2 CHIỀU BẰNG MẠNG MLP (PYTORCH)
> **Định dạng:** PowerPoint (`.pptx`) và xuất `.pdf`  
> **Thời lượng:** 10 - 15 phút  
> **Phân công trình bày:** Mỗi bạn trình bày các slide thuộc phần kỹ thuật mình phụ trách (mỗi người ~3 slide = ~3 phút nói).

---

### PHẦN 1: BÀI TOÁN & DỮ LIỆU (Thành viên 1 thuyết trình - Slide 1 đến 4)
* **Slide 1:** Tiêu đề đề tài, thành viên nhóm & tỷ lệ đóng góp (25% mỗi bạn).
* **Slide 2:** Mục tiêu bài toán & công thức toán Gaussian 2 chiều với $\boldsymbol{\mu} = [0, 0]^T, C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$.
* **Slide 3:** Phân chia dữ liệu khoa học (Tách 3 seed, tập Test 100 mẫu độc lập cố định, Pool 5000 mẫu: 400 val, 4600 train, Zero Data Leakage) & Biểu đồ phân tán 3D Scatter + Bề mặt thật Gaussian 2D.
* **Slide 4:** Thiết kế kiến trúc mạng `GaussianMLP` bằng PyTorch linh hoạt số tầng $C$, số nơ-ron $B$, hàm kích hoạt `ReLU`.

---

### PHẦN 2: PIPELINE HUẤN LUYỆN & THỰC NGHIỆM A (Thành viên 2 thuyết trình - Slide 5 & 6)
* **Slide 5:** Pipeline huấn luyện PyTorch chuẩn (`DataLoader`, `MSELoss`, `Adam`, `EarlyStopping`, `Checkpoint`, ghi nhận loss history từng epoch) & Bảng Baseline chuẩn cố định.
* **Slide 6:** Kết quả Thực nghiệm A (Ảnh hưởng của quy mô dữ liệu $A \in [200, 2000]$ theo tập con lồng nhau): Đồ thị sai số theo $A$, bảng Mean $\pm$ Std và nhận xét điểm bão hòa dữ liệu.

---

### PHẦN 3: THỰC NGHIỆM B & C (Thành viên 3 thuyết trình - Slide 7 & 8)
* **Slide 7:** Kết quả Thực nghiệm B (Độ rộng $B \in [1, 10+]$): Đồ thị sai số theo $B$, phân tích vùng Underfitting ($B=1, 2$) và Overfitting khi $B$ lớn.
* **Slide 8:** Kết quả Thực nghiệm C (Độ sâu $C \in [1, 5]$, cố định 32 nơ-ron/tầng): Đồ thị sai số theo $C$, Biểu đồ cột thời gian huấn luyện (giây), **Đường cong hội tụ loss theo epoch (`figures/exp_C_loss_convergence.png`)** và phân tích đánh đổi độ phức tạp.

---

### PHẦN 4: ĐÁNH GIÁ MÔ HÌNH TỐI ƯU & KẾT LUẬN (Thành viên 4 thuyết trình - Slide 9 đến 12)
* **Slide 9:** Thẩm định cấu hình tối ưu qua Validation Loss và Đánh giá ngoại kiểm trên tập Test 100 mẫu độc lập (MSE, MAE, RMSE, R2) & Biểu đồ Parity ($y$ vs $\hat{y}$) + Residuals.
* **Slide 10:** Đối sánh trực quan bề mặt 3D: Bề mặt Gaussian thật vs Bề mặt MLP xấp xỉ đặt cạnh nhau (`figures/best_model_3d_comparison.png`) & Đường cong học tập (`figures/best_model_learning_curve.png`).
* **Slide 11:** Thảo luận khoa học, bài học kinh nghiệm so với code cũ 2018, các hạn chế và đề xuất phát triển.
* **Slide 12:** Kết luận & Sẵn sàng Demo mã nguồn / Trả lời câu hỏi phản biện (Q&A).
