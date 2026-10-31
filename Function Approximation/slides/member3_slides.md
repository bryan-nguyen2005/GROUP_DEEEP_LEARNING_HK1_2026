# NỘI DUNG SLIDE THUYẾT TRÌNH - PHẦN PHỤ TRÁCH CỦA THÀNH VIÊN 3
**Vai trò:** Experiments B & C Lead (25% đóng góp)  
**Slide phụ trách:** Slide 7 & Slide 8 (Thời lượng thuyết trình: ~3 - 4 phút)

---

## SLIDE 7: THỰC NGHIỆM B - ẢNH HƯỞNG CỦA ĐỘ RỘNG TẦNG ẨN ($B$)

### 1. Mục tiêu & Thiết kế Thực nghiệm Công bằng
* **Khảo sát:** 13 mốc độ rộng tầng ẩn $B \in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]$ (mạng 1 tầng ẩn).
* **Cố định hệ quy chiếu:**
  - $N_{\text{train}} = 1500$ mẫu cố định (`iloc[400:1900]`), $N_{\text{val}} = 400$ mẫu cố định (`iloc[0:400]`).
  - Đánh giá ngoại kiểm trên $100$ mẫu Test độc lập (Zero Data Leakage).
  - Cấu hình Baseline: Adam ($\eta = 0.005$), batch 32, max epochs 150, ReLU, CPU.
* **Giao thức Đa-seed:** 3 seeds độc lập `[42, 100, 2024]` $\to$ 39 runs đo $\text{Mean} \pm \text{Std}$.

### 2. Bảng Kết quả Thực nghiệm & Đồ thị Xu hướng
*(Chèn hình đồ thị `figures/exp_B_width_vs_error.png` vào vị trí trung tâm slide)*

| Độ rộng $B$ | Test MSE ($\times 10^{-4}$) | Validation MSE ($\times 10^{-4}$) | Hiện tượng mô hình |
|:---:|:---:|:---:|:---|
| **$B = 1$** | $30.35 \pm 0.38$ | $29.76 \pm 0.27$ | **Thiếu khớp nghiêm trọng (Severe Underfitting)** |
| **$B = 2$** | $27.19 \pm 5.43$ | $26.38 \pm 5.68$ | Dung lượng quá yếu để tạo độ cong 2D |
| **$B = 4$** | $18.82 \pm 7.61$ | $17.95 \pm 6.56$ | Bắt đầu khép kín miền lồi quanh đỉnh nón |
| **$B = 6$** | $9.04 \pm 5.94$ | $8.89 \pm 5.77$ | Sai số giảm dốc đứng ($> 70\%$ so với $B=1$) |
| **$B = 16$** | **$5.04 \pm 3.58$** | $5.10 \pm 3.68$ | Tiệm cận sàn sai số của mạng 1 tầng ẩn |
| **$B = 64$** | **$5.02 \pm 6.66$** | $5.29 \pm 6.91$ | Đạt xấp xỉ tốt nhất, kiểm soát overfitting hoàn hảo |

### 3. Kết luận Chuyên môn & Ý nghĩa Toán học
1. **Minh chứng Định lý Xấp xỉ Phổ quát (UAT):** Mạng 1 tầng ẩn khi mở rộng số nơ-ron ($B \to 64$) có khả năng xấp xỉ liên tục bề mặt Gaussian 2D, sai số giảm hơn **6 lần**.
2. **Kiểm soát Quá khớp:** Dù tham số tăng gấp 4 lần, Test MSE vẫn bám sát Train MSE nhờ tập dữ liệu $N=1500$ đủ dày và cơ chế `EarlyStopping(patience=15)`.

---

## SLIDE 8: THỰC NGHIỆM C - ĐỘ SÂU MẠNG ($C$), THỜI GIAN & HỘI TỤ LOSS

### 1. Thiết lập Khảo sát Độ Sâu Chuẩn hóa (Cố định 32 nơ-ron/tầng)
* **Khảo sát:** $C \in [1, 2, 3, 4, 5]$ tầng ẩn (kiến trúc `[32]`, `[32, 32]`, ..., `[32]*5`).
* **Mục tiêu:** Đo lường sức mạnh biểu diễn tầng bậc, thời gian huấn luyện và tốc độ hội tụ loss.

### 2. Đối sánh Định lượng Độ Sâu vs Hiệu Năng
*(Bố trí 3 biểu đồ: Sai số `exp_C_depth_vs_error.png`, Cột thời gian `exp_C_depth_vs_time.png`, Đường cong hội tụ loss `exp_C_loss_convergence.png`)*

| Độ sâu $C$ | Số tham số | Test MSE ($\times 10^{-4}$) | Epoch dừng (Best) | Thời gian CPU (giây) |
|:---:|:---:|:---:|:---:|:---:|
| **$C = 1$ tầng** | $129$ | $5.64 \pm 5.31$ | $79.7$ ($64.7$) | $6.66 \pm 4.25\text{s}$ |
| **$C = 2$ tầng** | $1,185$ | **$0.93 \pm 0.37$** | $84.3$ ($69.3$) | $7.91 \pm 2.63\text{s}$ |
| **$C = 3$ tầng** | $2,241$ | **$0.50 \pm 0.22$** | $63.7$ ($48.7$) | $6.61 \pm 1.05\text{s}$ |
| **$C = 4$ tầng** | $3,297$ | **$0.49 \pm 0.05$** | $46.7$ ($31.7$) | **$5.75 \pm 0.80\text{s}$** |
| **$C = 5$ tầng** | $4,353$ | $0.57 \pm 0.53$ | $43.0$ ($28.0$) | $6.00 \pm 2.27\text{s}$ |

### 3. Đột phá Khoa học & Bài học Kiến trúc
1. **Ưu thế tuyệt đối của Độ sâu (Depth Efficiency):** 
   - Tăng từ $C=1$ lên $C=2$: Test MSE **giảm sốc hơn 6 lần** (từ $5.64$ về $0.93 \times 10^{-4}$).
   - Đạt kỷ lục tốt nhất tại **$C = 4$** ($\text{Test MSE} = 0.000049 \pm 0.000005$ - giảm hơn **11.5 lần** so với $C=1$).
   - Phân rã hàm phi tuyến lồng nhau $\mathcal{O}(B^C)$ hiệu quả vượt trội so với chỉ tăng độ rộng.
2. **Động học Hội tụ Loss:** Mạng sâu hội tụ cực nhanh (kỷ lục tại epoch $31.7$), giúp thời gian huấn luyện ngắn nhất ($5.75\text{s}$).
3. **Ngưỡng bão hòa tại $C=5$:** Sai số nhích nhẹ và phương sai giữa các seed tăng lên, khẳng định **$C = 3$ hoặc $C = 4$ là cấu hình tối ưu toàn diện (Sweet Spot)** cho Best Model.
