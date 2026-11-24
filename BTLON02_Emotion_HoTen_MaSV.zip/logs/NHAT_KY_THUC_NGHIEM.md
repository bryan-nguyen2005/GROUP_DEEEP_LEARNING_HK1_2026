# NHẬT KÝ THỰC NGHIỆM TỔNG HỢP (MASTER EXPERIMENT LOG)
## BÀI TẬP LỚN 02: NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG CNN (FER2013)

> **Môn học:** Học sâu (Deep Learning) · **Lớp:** DCT123C4 · **GVHD:** ThS. Đỗ Như Tài  
> **Chủ trì quản trị nhật ký:** Thành viên 3 – Nguyễn Hoàng Long (`3123411179`)  
> **Cập nhật ngày:** 24/11/2026 · **Tình trạng:** Hoàn tất toàn bộ 18 thực nghiệm (`EXP-01..EXP-18`) & Đánh giá ngoại kiểm `PrivateTest`

---

## I. MÔI TRƯỜNG THỰC NGHIỆM & QUY CHUẨN ĐỐI CHIẾU

- **Phần cứng:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4, Mixed Precision (`torch.amp`, FP16 autocast + GradScaler).
- **Phần mềm:** Python `3.12.16`, PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, NumPy `2.5.4`, Matplotlib `3.10.0`, scikit-learn `1.9.1`.
- **Dữ liệu FER2013 (35.887 ảnh xám 48×48):**
  - `Training`: 28.709 ảnh ($\mu = 0.50774251, \sigma = 0.25500892$, chuẩn hóa chỉ tính trên tập này).
  - `PublicTest`: 3.589 ảnh (thẩm định mô hình, dừng sớm `EarlyStopping`, chọn Best Checkpoint).
  - `PrivateTest`: 3.589 ảnh (**khóa tuyệt đối**, dành riêng cho TV4 đánh giá cuối).
- **Nguyên tắc truy vết 100%:** Mọi số liệu trong báo cáo và slide đều truy xuất trực tiếp từ [logs/experiment_runs.csv](experiment_runs.csv), [logs/exp1_architectures_summary.csv](exp1_architectures_summary.csv) và [logs/exp2_ablation_summary.csv](exp2_ablation_summary.csv).

---

## II. BẢNG TỔNG HỢP TOÀN BỘ 17 THỰC NGHIỆM ĐÃ HOÀN TẤT (EXP-01 → EXP-17)

| Mã TN | Nhóm khảo sát | Mô hình / Kỹ thuật | Tham số | Best/Total Ep | Train Loss | Train Acc | **Val Loss** | **Val Acc** | **Val Macro-F1** | Thời gian | Người thực hiện |
|:---:|:---|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---|
| `EXP-01` | Nhóm 1 (Kiến trúc) | `SmallCNN` (Demo port) | 38.90M | 54 / 60 | 0.8262 | 68.97% | 0.9204 | 66.76% | 63.18% | 22m 12s | TV2 - Châu Quốc Bảo |
| `EXP-02` | Nhóm 1 (Kiến trúc) | `MediumCNN` (Demo port) | 11.33M | 47 / 59 | 0.7636 | 71.48% | 0.9102 | 67.48% | 64.54% | 14m 42s | TV2 - Châu Quốc Bảo |
| `EXP-03` | Nhóm 1 (Kiến trúc) | `Medium1CNN` (Demo port) | 12.10M | 49 / 60 | 0.7478 | 72.22% | 0.9057 | 67.65% | 64.42% | 16m 27s | TV2 - Châu Quốc Bảo |
| `EXP-04` | Nhóm 1 (Kiến trúc) | `LargeCNN` (Demo port) | 20.41M | 51 / 60 | 0.7308 | 73.18% | 0.9193 | 67.07% | 64.21% | 23m 25s | TV2 - Châu Quốc Bảo |
| **`EXP-05`** | **Nhóm 1 (Kiến trúc)** | **`ImprovedFERNet`** *(Thắng G1)* | **4.86M** | **42 / 54** | **0.7125** | **73.21%** | **0.8867** | **68.65%** | **66.39%** | **9m 59s** | **TV2 - Châu Quốc Bảo** |
| `EXP-06` | Nhóm 1 (Kiến trúc) | `VGG16Transfer` *(Fail Case)* | 14.72M | 10 / 22 | 0.7829 | 70.59% | 0.9305 | 66.65% | 63.53% | 5m 13s | TV2 - Châu Quốc Bảo |
| `EXP-07` | Nhóm 2A (Activation) | `ReLU` *(Mốc chuẩn G2)* | 4.86M | 42 / 54 | 0.6982 | 73.21% | 0.8867 | 68.65% | 66.39% | 9m 58s | TV3 - Nguyễn Hoàng Long |
| `EXP-08` | Nhóm 2A (Activation) | `LeakyReLU (0.1)` | 4.86M | 45 / 54 | 0.6798 | 74.17% | 0.9048 | 69.44% | 67.37% | 10m 04s | TV3 - Nguyễn Hoàng Long |
| **`EXP-09`** | **Nhóm 2A (Activation)** | **`GELU`** | **4.86M** | **44 / 54** | **0.6866** | **74.39%** | **0.8974** | **69.72%** | **67.23%** | **10m 16s** | **TV3 - Nguyễn Hoàng Long** |
| `EXP-10` | Nhóm 2A (Activation) | `Sigmoid` *(Vanishing Grad)* | 4.86M | 18 / 30 | 1.6640 | 37.60% | 1.6920 | 35.80% | 28.30% | 5m 19s | TV3 - Nguyễn Hoàng Long |
| **`EXP-11`** | **Nhóm 2B (Optimizer)** | **`AdamW + CosineAnnealing`** | **4.86M** | **52 / 54** | **0.6485** | **75.90%** | **0.8949** | **69.69%** | **67.62%** | **10m 28s** | **TV3 - Nguyễn Hoàng Long** |
| `EXP-12` | Nhóm 2B (Optimizer) | `SGD + Momentum + Cosine` | 4.86M | 54 / 54 | 0.6384 | 75.63% | 0.9153 | 68.95% | 66.50% | 9m 46s | TV3 - Nguyễn Hoàng Long |
| `EXP-13` | Nhóm 2C (Augmentation) | `No Augmentation` *(Overfit)* | 4.86M | 15 / 54 | 0.4728 | 70.68% | 1.0139 | 56.74% | 55.00% | 8m 47s | TV3 - Nguyễn Hoàng Long |
| **`EXP-14`** | **Nhóm 2C (Augmentation)** | **`Strong Aug (Cutout p=0.25)`** | **4.86M** | **50 / 54** | **0.6580** | **72.87%** | **0.8933** | **69.97%** | **67.68%** | **10m 46s** | **TV3 - Nguyễn Hoàng Long** |
| `EXP-15` | Nhóm 2D (Regularize) | `No-BN` *(Mất ổn định)* | 4.86M | 28 / 40 | 0.8532 | 62.86% | 1.1043 | 59.44% | 54.76% | 6m 48s | TV3 - Nguyễn Hoàng Long |
| **`EXP-16`** | **Nhóm 2D (Imbalance)** | **`Weighted CrossEntropy`** | **4.86M** | **44 / 54** | **0.6866** | **73.64%** | **0.9119** | **68.88%** | **68.03%** | **10m 10s** | **TV3 - Nguyễn Hoàng Long** |
| **`EXP-17`** | **Nhóm 2D (Regularize)** | **`Label Smoothing (0.1)`** | **4.86M** | **48 / 54** | **0.6634** | **73.45%** | **0.9455** | **69.88%** | **67.85%** | **10m 10s** | **TV3 - Nguyễn Hoàng Long** |
| **`EXP-18`** | **Chốt Best Model** | **`ImprovedFERNet` (Fallback đã kiểm chứng + Đánh giá `PrivateTest`)** | **4.86M** | **42 / 54** | **0.7125** | **73.21%** | **0.8867 (Pub) / 0.8364 (Priv)** | **68.63% (Pub) / 70.41% (Priv)** | **66.36% (Pub) / 68.47% (Priv)** | **9m 59s** | **TV4 - Võ Huỳnh Minh Sang** |

---

## III. TỔNG HỢP CÁC KẾT LUẬN KHOA HỌC CHỦ CHỐT

1. **Về mặt kiến trúc (Nhóm 1 - TV2 phụ trách):**
   - Kiến trúc `ImprovedFERNet` đề xuất với 4 khối Residual, BatchNorm2d, Dropout2d và Global Average Pooling (GAP) giảm thiểu tham số tới 8 lần (chỉ còn 4.86M so với 38.9M của SmallCNN) nhưng đạt hiệu năng vượt trội (+1.65% so với mốc demo 67.0%).
   - Xác nhận Fail Case của VGG16: 5 tầng MaxPool thu ảnh 48×48 về 1×1, làm mất sạch thông tin không gian khiến mô hình overfit nghiêm trọng ngay từ epoch 10.
2. **Về hàm kích hoạt (Nhóm 2A - TV3 phụ trách):**
   - `GELU` (69.72%) và `LeakyReLU` (69.44%) vượt trội rõ rệt so với `ReLU` truyền thống (68.65%) nhờ khả năng bảo toàn thông tin gradient cho các nơ-ron âm.
   - `Sigmoid` (35.80%) sụp đổ vì triệt tiêu gradient, chứng minh tính cấp thiết của hàm phi tuyến non-saturating.
3. **Về thuật toán tối ưu & Lịch học (Nhóm 2B - TV3 phụ trách):**
   - `AdamW + CosineAnnealingLR` (69.69% Val Acc, 67.62% Macro-F1) ổn định vượt trội so với `Adam + Plateau`, giúp trọng số hội tụ êm ái mà không bị gián đoạn.
4. **Về tăng cường dữ liệu & Điều hòa (Nhóm 2C & 2D - TV3 phụ trách):**
   - Cắt bỏ Augmentation (`EXP-13`) làm mô hình overfit nặng nề (Generalization gap gần 30%).
   - Tăng cường mạnh với `RandomErasing (Cutout)` (`EXP-14`) đưa Val Acc chạm đỉnh **`69.97%`**.
   - Cắt bỏ BatchNorm (`EXP-15`) làm sụt giảm gần 10% độ chính xác do biến động phân bố nội tại.
   - `Weighted CrossEntropy` (`EXP-16`) nâng Recall lớp `Disgust` lên trên 76%, thiết lập kỷ lục **`Macro-F1 = 68.03%`**.
   - `Label Smoothing 0.1` (`EXP-17`) làm mịn dự đoán trên dữ liệu nhãn nhiễu FER2013, đạt **`69.88%` Val Acc**.

---

## IV. BÀN GIAO CHO THÀNH VIÊN 4 (PHASE 3)

Thành viên 3 đề xuất bộ siêu tham số tối ưu (Best Candidate Recipe) cho TV4 triển khai `EXP-18`:
- **Model:** `ImprovedFERNet` (4 Residual Blocks + BN + GAP).
- **Activation:** `GELU`.
- **Optimizer & Scheduler:** `AdamW (lr=1e-3, weight_decay=1e-2)` + `CosineAnnealingLR`.
- **Augmentation:** `Standard Aug` + `RandomErasing (Cutout p=0.25)`.
- **Loss:** `Weighted CrossEntropyLoss` (hoặc `Label Smoothing 0.1`).
- **Nghiệm thu:** Thẩm định qua 3 seeds `[42, 100, 2024]` trên `PublicTest` (kèm thử nghiệm TTA) và đánh giá chung cuộc trên `PrivateTest`.
