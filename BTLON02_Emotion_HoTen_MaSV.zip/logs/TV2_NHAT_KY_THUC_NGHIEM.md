# Nhật ký thực nghiệm Nhóm 1 – Thành viên 2 (TV2)

**Thời gian thực hiện:** 20/11/2026 – 22/11/2026 · **Phụ trách:** Châu Quốc Bảo (`3123411024`)  
**Phạm vi:** Xây dựng Core Training Pipeline (`utils.py`, `metrics.py`, `trainer.py`) và chạy Thực nghiệm Nhóm 1 (`EXP-01` → `EXP-06`) trên tập `Training` (28.709 ảnh) và `PublicTest` (3.589 ảnh). Tập `PrivateTest` (3.589 ảnh) **không dùng trong thí nghiệm Nhóm 1**.

---

## 1. Môi trường phần cứng & Thiết lập chuẩn (Baseline)

- **Phần cứng:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4, Mixed Precision (`torch.amp`, FP16 autocast + GradScaler).
- **Phần mềm:** Python `3.12.16`, PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, NumPy `2.5.4`, scikit-learn `1.9.1`.
- **Dữ liệu đầu vào:** `data/fer2013.csv` trùng khớp tuyệt đối mã SHA-256 của TV1 (`3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`). Chuẩn hóa chỉ theo thống kê `Training` ($\mu = 0.50774251, \sigma = 0.25500892$).
- **Cấu hình Baseline cố định:**
  - `seed = 42`, `batch_size = 64`, `max_epochs = 60`, `EarlyStopping(patience=12, monitor='val_loss')`.
  - `activation = relu`, `dropout = 0.2`, `aug_mode = standard` (Flip + Rotate $10^\circ$ + Affine shift/scale $0.1$).
  - `loss = nn.CrossEntropyLoss()`, `optimizer = Adam(lr=1e-3)` (riêng Fail Case fine-tune `VGG16Transfer` dùng `lr=1e-4` chuẩn transfer learning), `scheduler = ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)`.
  - Checkpoint lưu theo **epoch có `val_loss` thấp nhất** trên `PublicTest`.

---

## 2. Bảng tổng hợp kết quả Thực nghiệm Nhóm 1 (`EXP-01` → `EXP-06`)

Nguồn truy vết 100%: [exp1_architectures_summary.csv](exp1_architectures_summary.csv) và [experiment_runs.csv](experiment_runs.csv).

| Mã TN | Kiến trúc | Tham số | BN | LR | Best / Total Ep | Train Loss | Train Acc | **Val Loss** | **Val Acc (Best Loss)** | Peak Val Acc | **Val Macro-F1** | Thời gian |
|:---:|:---|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `EXP-01` | `SmallCNN` | 38.902.087 | Có | `1e-3` | 54 / 60 | 0.8262 | 68.97% | 0.9204 | 66.76% | 67.46% (ep 57) | 63.18% | 22m 11.6s |
| `EXP-02` | `MediumCNN` | 11.328.583 | Có | `1e-3` | 47 / 59 | 0.7636 | 71.48% | 0.9102 | 67.48% | 69.24% (ep 58) | 64.54% | 14m 42.0s |
| `EXP-03` | `Medium1CNN` | 12.103.623 | Có | `1e-3` | 49 / 60 | 0.7478 | 72.22% | 0.9057 | 67.65% | 69.18% (ep 60) | 64.42% | 16m 26.7s |
| `EXP-04` | `LargeCNN` | 20.410.439 | Có | `1e-3` | 51 / 60 | 0.7308 | 73.18% | 0.9193 | 67.07% | 67.07% (ep 51) | 64.21% | 23m 25.0s |
| **`EXP-05`** | **`ImprovedFERNet`** | **4.863.751** | **Có** | **`1e-3`** | **42 / 54** | **0.7125** | **73.21%** | **0.8867** | **68.65%** | **69.52% (ep 50)** | **66.39%** | **9m 59.2s** |
| `EXP-06` | `VGG16Transfer` *(Fail Case)* | 14.718.279 | Không | `1e-4` | 10 / 22 | 0.7829 *(cuối: 0.3535)* | 70.59% *(cuối: 87.05%)* | 0.9305 *(cuối: 1.2793)* | 66.65% | 68.29% (ep 16) | 63.53% | 5m 13.1s |

---

## 3. Ghi chép chi tiết từng phiên thí nghiệm (Session Log)

### Phiên 1: `EXP-01` – `SmallCNN` (Demo port, 2 khối tích chập)
- **Mục tiêu & Giả thuyết:** Tái tạo mô hình nhỏ nhất từ `demo_report.pdf` (đạt 64% trên Keras không BN) có bổ sung `BatchNorm2d` làm mốc thấp nhất cho Nhóm 1.
- **Kết quả:** Best epoch `54/60`, `val_loss = 0.9204`, `val_acc = 66.76%`, `val_macro_f1 = 63.18%`, thời gian `1331.56s` (`22m 11.6s`).
- **Quan sát & Phân tích:** Dù chỉ có 2 khối tích chập nhưng do chỉ pool 2 lần ($48 \to 24 \to 12$), tầng làm phẳng sinh vector $12 \times 12 \times 256 = 36.864$ chiều nối vào `FC(1024)` khiến tổng tham số phình lên **38,90 triệu** (nặng nhất trong 6 mô hình), chạy chậm (~22s/epoch). Accuracy cao hơn bản Keras gốc được nêu trong tài liệu 2,76 điểm phần trăm (64% lên 66,76%); do nhiều cấu hình khác nhau, không thể quy mức tăng riêng cho `BatchNorm2d`, nhưng giới hạn về độ sâu khiến biểu diễn đặc trưng cấp cao chưa đủ tốt.

### Phiên 2: `EXP-02` – `MediumCNN` (Demo port, 3 khối: 2×64, 3×128, 3×256)
- **Mục tiêu & Giả thuyết:** Giảm chiều không gian trước `FC(1024)` xuống $6 \times 6 \times 256 = 9.216$ nhờ 3 khối MaxPool sẽ vừa giảm mạnh tham số, tăng tốc huấn luyện vừa cải thiện độ chính xác.
- **Sự cố & Xử lý:** Lần chạy đầu tối 20/11 bị gián đoạn ở epoch 44 do máy khởi động lại; chạy lại sạch toàn bộ `EXP-02..EXP-06` bằng cờ `--exp-ids` sáng 22/11 (giữ nguyên `seed=42`).
- **Kết quả:** Best epoch `47/59`, `val_loss = 0.9102`, `val_acc = 67.48%` (peak `69.24%` tại ep 58), `val_macro_f1 = 64.54%`, thời gian `881.96s` (`14m 42.0s`).
- **Quan sát:** Số tham số giảm gần 3,5 lần so với `SmallCNN` (`11.33M` vs `38.90M`), vượt mốc tốt nhất của bản demo 2018 (`67%`) và cải thiện Macro-F1 thêm `+1,36%`.

### Phiên 3: `EXP-03` – `Medium1CNN` (Demo port, 3 khối sâu hơn: 3×64, 4×128, 4×256)
- **Mục tiêu & Giả thuyết:** Kiểm chứng liệu việc tăng số tầng tích chập trong mỗi khối có còn gây suy giảm hiệu năng như bản Keras không BN (`demo_report.pdf`: tụt từ 67% xuống 65%) khi đã có `BatchNorm2d`.
- **Kết quả:** Best epoch `49/60`, `val_loss = 0.9057`, `val_acc = 67.65%` (peak `69.18%` tại ep 60), `val_macro_f1 = 64.42%`, thời gian `986.66s` (`16m 26.7s`).
- **Quan sát:** Trong cấu hình này, `Medium1CNN` **không bị suy giảm** như bản demo 2018; chưa có ablation để quy kết riêng cho `BatchNorm2d` mà đạt `val_loss` tốt nhất trong 4 mô hình port từ demo (`0.9057`). Tuy nhiên mức cải thiện so với `MediumCNN` khá nhỏ (`+0,17%` accuracy).

### Phiên 4: `EXP-04` – `LargeCNN` (Demo port, 4 khối + FC 2048)
- **Mục tiêu & Giả thuyết:** Kiểm tra xem việc tăng lên 4 khối (16 tầng Conv, tới 512 kênh) và `FC(2048)` có tiếp tục nâng độ chính xác hay bắt đầu bão hòa/quá khớp do thiếu skip connection.
- **Kết quả:** Best epoch `51/60`, `val_loss = 0.9193`, `val_acc = 67.07%`, `val_macro_f1 = 64.21%`, thời gian `1404.95s` (`23m 25.0s`).
- **Quan sát:** Tốc độ huấn luyện chậm nhất (`23m 25s`), xuất hiện dao động `val_loss` lớn ở giai đoạn đầu (đỉnh `2.05` ở epoch 4) do mạng sâu 16 tầng Conv thuần chuỗi không có kết nối tắt. Độ chính xác và `val_loss` đều kém hơn `MediumCNN` và `Medium1CNN`, cho thấy tăng chiều sâu thuần túy và mở rộng tầng FC không phải hướng tối ưu cho FER2013 $48\times 48$.

### Phiên 5: `EXP-05` – `ImprovedFERNet` (Đề xuất nhóm: 4 Residual Blocks + BN + Dropout2d + GAP)
- **Mục tiêu & Giả thuyết:** Kết hợp kết nối tắt (Residual Shortcut $1\times 1$), `BatchNorm2d`, `Dropout2d` theo kênh và **Global Average Pooling** (loại bỏ hoàn toàn tầng `FC(1024/2048)` nặng tham số) sẽ giúp hội tụ nhanh, chống overfit tốt và vượt trội về cả Accuracy lẫn Macro-F1.
- **Kết quả:** Best epoch `42/54`, **`val_loss = 0.8867`**, **`val_acc = 68.65%`** (peak **`69.52%`** tại ep 50), **`val_macro_f1 = 66.39%`**, thời gian **`599.18s` (`9m 59.2s`)**, chỉ **`4.86M` tham số**.
- **Phân tích theo từng lớp trên `PublicTest`** (nguồn [exp1_best_val_per_class.csv](exp1_best_val_per_class.csv)):
  - Cao nhất: `Happy` (F1 = `87.24%`, Recall = `87.82%`) và `Surprise` (F1 = `80.77%`, Recall = `80.48%`).
  - Lớp thiểu số `Disgust` (chỉ 56 ảnh Val): đạt Precision `73.17%`, Recall `53.57%`, F1 `61.86%` — vẫn còn dư địa cải thiện Recall bằng `Weighted CE` hoặc `Strong Aug` ở Nhóm 2.
  - Khó nhất: `Fear` (F1 = `51.70%`, Recall = `47.58%`) và `Sad` (F1 = `58.63%`) do biểu cảm mơ hồ, dễ nhầm với `Angry`/`Neutral`.
- **Quyết định:** **Chốt `ImprovedFERNet` là kiến trúc ứng viên dẫn đầu Nhóm 1 trong lần chạy này** (dẫn đầu cả 4 tiêu chí: `val_loss` thấp nhất, `val_acc` cao nhất, `val_macro_f1` cao nhất, ít tham số nhất và chạy nhanh nhất trong các mô hình train từ đầu) để bàn giao cho **TV3** thực hiện Ablation Study (`EXP-07` → `EXP-17`).

### Phiên 6: `EXP-06` – `VGG16Transfer` (**Fail Case**: Fine-tune ImageNet VGG16)
- **Mục tiêu & Giả thuyết:** Kiểm chứng ca thất bại được đề cập ở mục 5.1 của `demo_report.pdf` khi áp dụng trực tiếp backbone VGG16 pretrained ImageNet (5 tầng MaxPool, không BatchNorm) lên ảnh xám $48 \times 48$.
- **Kết quả:** Tải trọng số `VGG16_Weights.IMAGENET1K_V1`, fine-tune với `Adam(lr=1e-4)`. Đạt `val_loss` thấp nhất `0.9305` (`val_acc = 66.65%`, `val_macro_f1 = 63.53%`) rất sớm tại **epoch 10**, sau đó **phân kỳ overfitting nghiêm trọng**: đến epoch 22, `train_acc` vọt lên **`87.05%`** (`train_loss = 0.3535`) trong khi `val_loss` tăng vọt lên **`1.2793`** (chênh lệch Train–Val accuracy gần **20%**), kích hoạt `EarlyStopping` ở epoch 22 (`5m 13.1s`).
- **Các giả thuyết cần kiểm chứng về nguyên nhân:**
  1. **Giảm mạnh độ phân giải không gian:** 5 tầng `MaxPool2d(2)` nén ảnh $48 \times 48 \to 24 \to 12 \to 6 \to 3 \to \mathbf{1 \times 1}$ trước bộ phân loại, có thể làm mất một phần thông tin vị trí; cần thí nghiệm bổ sung để xác định ảnh hưởng.
  2. **Lệch miền & thiếu BatchNorm:** Trọng số ImageNet học đặc trưng vật thể màu RGB kích thước $224 \times 224$; khi fine-tune 14,72M tham số không có BatchNorm trên 28.709 ảnh xám nhỏ, có thể góp phần gây quá khớp. `val_loss` tăng sau epoch 10, nhưng chưa có thí nghiệm cô lập để xác định nguyên nhân.

---

## 4. Lệnh tái lập phần việc TV2

```bash
# 1. Chạy toàn bộ unit test (TV1 + TV2)
python -m unittest discover -s tests -v

# 2. Chạy trọn bộ Thực nghiệm Nhóm 1 (EXP-01..EXP-06) trên GPU
python -m src.run_exp1_architectures --data-path data/fer2013.csv --max-epochs 60 --patience 12 --batch-size 64 --seed 42

# 3. Xuất lại riêng bảng tổng hợp & 3 biểu đồ 300 DPI từ logs có sẵn
python -m src.run_exp1_architectures --plot-only
```

> Các so sánh trên dựa trên một seed và các cấu hình đã nêu. Chênh lệch với tài liệu gốc chỉ mang tính tham khảo; chưa đủ thí nghiệm cắt bỏ để quy kết nguyên nhân.
