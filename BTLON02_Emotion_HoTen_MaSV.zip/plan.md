# KẾ HOẠCH & PHÂN CÔNG NHIỆM VỤ BÀI TẬP LỚN 02 – HỌC SÂU
## ĐỀ TÀI: NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG CNN (CNN FOR FACIAL EXPRESSION RECOGNITION – FER2013)

> **Lớp:** DCT123C4 | **Môn học:** Học sâu (Deep Learning) | **GVHD:** Đỗ Như Tài  
> **Quy mô nhóm:** 04 thành viên (25% khối lượng thực chất / thành viên)  
> **File nộp bài chuẩn (Nhóm trưởng nộp):** `BTLON02_Emotion_NguyenDinhCuong_3123411043.zip`  
> **Hồ sơ nộp bắt buộc:** Báo cáo (Report), Slide thuyết trình, Nhật ký thực nghiệm, Mã nguồn PyTorch (ghi rõ bảng phân công thành viên).

---

## I. CĂN CỨ TÀI LIỆU & MỤC TIÊU KỸ THUẬT

### 1. Căn cứ tài liệu giao bài
- **Yêu cầu bắt buộc (`Emotion_Recognition.pdf`):**
  - **Bộ dữ liệu:** Kaggle **FER2013** gồm **35.887** ảnh khuôn mặt mức xám kích thước **$48 \times 48 \times 1$**, chia thành **7 lớp cảm xúc**:
    `0 = Angry`, `1 = Disgust`, `2 = Fear`, `3 = Happy`, `4 = Sad`, `5 = Surprise`, `6 = Neutral`.
  - **Phân tập chuẩn của FER2013:**
    - `Training`: **28.709** ảnh (lưu ý mất cân bằng lớp nghiêm trọng: `Disgust` chỉ có 436 ảnh $\approx 1.52\%$, trong khi `Happy` có 7.215 ảnh $\approx 25.13\%$).
    - `PublicTest`: **3.589** ảnh (dùng làm tập **Validation**).
    - `PrivateTest`: **3.589** ảnh (khóa hoàn toàn làm tập **Final Test**).
  - **Mốc tham chiếu (Benchmark):** Kết quả tốt nhất cuộc thi gốc trên Kaggle là **69% (Public)** và **71% (Private)**; kết quả mẫu trong `demo_report.pdf` đạt **64% – 67%**.
  - **Yêu cầu công nghệ bắt buộc:** Toàn bộ mã nguồn triển khai bằng **Python + PyTorch** (tuyệt đối **không** dùng lại Keras/TensorFlow 1.8 cũ trong `demo_code.html`).
  - **4 hạng mục bắt buộc trong Báo cáo (Slide 5):**
    1. Tóm tắt ngắn gọn, rõ ràng về mã nguồn (Brief summary for source codes).
    2. Kết quả của nhiều thực nghiệm với đa dạng thiết lập tham số / kiến trúc / kỹ thuật huấn luyện.
    3. **Trực quan hóa các bộ lọc tầng tích chập cuối cùng (Visualize the final convolution filters)** *(nhấn mạnh đỏ trong đề bài)*.
    4. Phân tích và giải thích kết quả theo góc nhìn riêng của nhóm (Confusion Matrix, True/Wrong Cases, Fail Cases, mất cân bằng lớp).
- **Tài liệu tham khảo cấu trúc (`demo_report.pdf` & `demo_code.html`):**
  - Tham khảo cách tổ chức luồng khảo sát 4 kiến trúc CNN (`SMALL`, `MEDIUM`, `MEDIUM_01`, `LARGE`), phân tích ca thất bại (**Fail Case: Fine-tune VGG16** trên ảnh $48 \times 48$), hiển thị Confusion Matrix, True Cases và Wrong Cases.
  - Khắc phục các điểm còn thiếu của bản demo 2018: chuyển hoàn toàn sang **PyTorch**, bổ sung **BatchNorm / Residual / Global Average Pooling**, khảo sát hệ thống **Activation / Optimizer / Augmentation / Class Imbalance**, bổ sung **Nhật ký thực nghiệm** chuẩn hóa và **trực quan hóa Final Convolution Filters + Feature Maps + Grad-CAM**.

### 2. Mục tiêu định lượng của nhóm
- Xây dựng pipeline PyTorch sạch, tái lập được, không rò rỉ dữ liệu (**Zero Data Leakage**).
- Thực hiện tối thiểu **16–18 cấu hình thực nghiệm có kiểm soát** (chia 2 nhóm lớn: Kiến trúc & Kỹ thuật huấn luyện) và ghi chép đầy đủ vào Nhật ký thực nghiệm.
- Đạt độ chính xác trên tập `PrivateTest` **vượt mốc demo ($\ge 67\%$)** và hướng tới ngưỡng cạnh tranh **$68\% - 71\%$** (bằng kiến trúc CNN tối ưu cho ảnh $48 \times 48$, BatchNorm, Data Augmentation, LR Scheduler và tùy chọn Test-Time Augmentation).

---

## II. NGUYÊN TẮC KHOA HỌC & CHUẨN HÓA THỰC NGHIỆM

### 1. Chống rò rỉ dữ liệu (Zero Data Leakage) & Quy ước phân tập
- **Tập `Training` (28.709 ảnh):** Chỉ dùng để cập nhật trọng số mô hình, tính thống kê chuẩn hóa $(\mu_{\text{train}}, \sigma_{\text{train}})$, tính trọng số lớp (`class_weights`) và áp dụng Data Augmentation.
- **Tập `PublicTest` (3.589 ảnh – Validation):** Cố định làm tập thẩm định cho mọi thực nghiệm, kích hoạt `EarlyStopping`, `ReduceLROnPlateau`, lưu `ModelCheckpoint` tốt nhất và chọn cấu hình Best Model. **Không** áp dụng random augmentation khi evaluate.
- **Tập `PrivateTest` (3.589 ảnh – Final Unseen Test):** **Khóa tuyệt đối** trong suốt quá trình thiết kế và chọn siêu tham số. Chỉ được dùng **một lần duy nhất** ở giai đoạn cuối (`src/evaluate_best_model.py`) để đánh giá mô hình tốt nhất đã chốt từ `PublicTest`.
- **Kiểm tra dữ liệu lỗi (Data Hygiene):** Trong FER2013 có một số ảnh bị hỏng (toàn bộ pixel bằng 0). TV1 kiểm kê rõ số lượng ảnh bất thường trong EDA và ghi nhận cách xử lý nhất quán (giữ nguyên số lượng tập PublicTest/PrivateTest để so sánh công bằng với benchmark gốc).

### 2. Quy ước tái lập & Môi trường phần cứng
- **Seed cố định:** Cài đặt hàm `seed_everything(seed)` cho `random`, `numpy`, `torch`, `torch.cuda`.
  - Mọi thực nghiệm khảo sát đơn chạy cố định với `seed = 42`.
  - Riêng ứng viên **Best Model** chạy kiểm định ổn định trên **3 seeds `[42, 100, 2024]`** (báo cáo $\text{Mean} \pm \text{Std}$ trên `PublicTest`, chọn checkpoint gần Mean nhất để đánh giá cuối trên `PrivateTest`).
- **Phần cứng huấn luyện:** Ưu tiên chạy trên **GPU** (Google Colab T4 / Kaggle GPU T4–P100 / GPU CUDA cục bộ) có hỗ trợ **Mixed Precision (`torch.amp`)** để rút ngắn thời gian chạy hàng chục cấu hình CNN; mã nguồn tự động fallback về CPU nếu chạy kiểm thử cục bộ. Mọi run đều ghi rõ tên GPU/CPU và thời gian huấn luyện vào Nhật ký thực nghiệm.

### 3. Cấu hình Baseline chuẩn (Mốc đối chiếu chung)
Để cô lập biến khi làm Ablation Study, tất cả các thực nghiệm đều so sánh với Baseline thống nhất sau:
- **Đầu vào:** Tensor `(1, 48, 48)`, scale về `[0, 1]` và chuẩn hóa theo `mean/std` của tập Train.
- **Hàm mất mát:** `nn.CrossEntropyLoss()`.
- **Hàm kích hoạt mặc định:** `ReLU`.
- **Optimizer mặc định:** `Adam(lr=1e-3)`.
- **LR Scheduler mặc định:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=4, min_lr=1e-6)`.
- **Batch size:** `64`.
- **Số epoch tối đa:** `60` (kèm `EarlyStopping(patience=12)` theo `val_loss`).
- **Data Augmentation mặc định (Standard Aug):** `RandomHorizontalFlip(p=0.5)`, `RandomRotation(10)`, `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`.

### 4. Ma trận thiết kế thực nghiệm (Tổng cộng 18 thí nghiệm có kiểm soát)

| Nhóm | Mã TN | Nội dung khảo sát | Biến thay đổi (Các yếu tố khác giữ cố định Baseline) | Phụ trách |
|:---:|:---:|:---|:---|:---:|
| **Nhóm 1: Kiến trúc CNN** | `EXP-01` | `SmallCNN` (port từ demo) | 2 blocks Conv (64 → 128/256), FC 1024 | TV2 |
| | `EXP-02` | `MediumCNN` (port từ demo) | 3 blocks Conv (64 → 128 → 256), FC 1024 | TV2 |
| | `EXP-03` | `Medium1CNN` (port từ demo) | 3 blocks sâu hơn (3×64 → 4×128 → 4×256), FC 1024 | TV2 |
| | `EXP-04` | `LargeCNN` (port từ demo) | 4 blocks Conv (64 → 128 → 256 → 512), FC 2048 | TV2 |
| | `EXP-05` | `ImprovedFERNet` (Đề xuất nhóm) | 4 blocks Conv + **BatchNorm2d** + Residual/SE nhẹ + **GAP** | TV2 |
| | `EXP-06` | `VGG16Transfer` (**Fail Case**) | Fine-tune VGG16 trên ảnh $48\times 48$ (5 MaxPool → overfit mạnh) | TV2 |
| **Nhóm 2A: Activation** | `EXP-07` | Activation = `ReLU` | Trên kiến trúc thắng Nhóm 1 (Base reference) | TV3 |
| | `EXP-08` | Activation = `LeakyReLU(0.1)` | Thay thế toàn bộ activation bằng LeakyReLU | TV3 |
| | `EXP-09` | Activation = `GELU` (hoặc `ELU`) | Thay thế toàn bộ activation bằng GELU/ELU | TV3 |
| | `EXP-10` | Activation = `Sigmoid` | Kiểm chứng hiện tượng bão hòa/vanishing gradient | TV3 |
| **Nhóm 2B: Optimizer & LR** | `EXP-11` | `AdamW` + `CosineAnnealingLR` | `lr=1e-3, weight_decay=1e-2` | TV3 |
| | `EXP-12` | `SGD + Momentum(0.9)` + Cosine | `lr=0.01, weight_decay=5e-4, nesterov=True` | TV3 |
| **Nhóm 2C: Augmentation** | `EXP-13` | `No Augmentation` | Chỉ chuẩn hóa, tắt toàn bộ augmentation | TV3 |
| | `EXP-14` | `Strong Augmentation` | Standard Aug + `RandomErasing(p=0.25)` (Cutout) | TV3 |
| **Nhóm 2D: Regularization & Imbalance** | `EXP-15` | Không dùng `BatchNorm` | So sánh trực tiếp hiệu quả ổn định hội tụ của BN | TV3 |
| | `EXP-16` | `Weighted CrossEntropyLoss` | Gán trọng số nghịch đảo tần suất lớp (cứu lớp `Disgust`) | TV3 |
| | `EXP-17` | `Label Smoothing (0.1)` | Giảm overconfidence do nhiễu nhãn trong FER2013 | TV3 |
| **Chốt Best Model** | `EXP-18` | **Best Config (3 seeds + TTA)** | Kết hợp cấu hình thắng từ Nhóm 1 & 2, kiểm định 3 seeds `[42, 100, 2024]` trên `PublicTest` (có Fallback), đánh giá cuối trên `PrivateTest` | TV4 |

---

## III. DANH SÁCH THÀNH VIÊN & PHÂN CÔNG CHI TIẾT (25% × 4)

| STT | Họ và tên                            |     MSSV     | Vai trò chính                                                                                                        | Tỷ lệ |
| :-: | :----------------------------------- | :----------: | :------------------------------------------------------------------------------------------------------------------- | :---: |
|  1  | **Nguyễn Đình Cường** *(nhóm trưởng) | `3123411043` | **TV1:** Data Pipeline, EDA, Transforms & Kho kiến trúc CNN (`models.py`)                                            |  25%  |
|  2  | **Châu Quốc Bảo**                    | `3123411024` | **TV2:** Core Training Pipeline, Metrics, Callbacks & Thực nghiệm Nhóm 1 (Kiến trúc + Fail Case)                     |  25%  |
|  3  | **Nguyễn Hoàng Long**                | `3123411179` | **TV3:** Thực nghiệm Nhóm 2 (Ablation kỹ thuật huấn luyện) & Chủ trì Nhật ký thực nghiệm                             |  25%  |
|  4  | **Võ Huỳnh Minh Sang**               | `3123411256` | **TV4:** Trực quan hóa Final Conv Filters/Grad-CAM, Đánh giá PrivateTest, Tổng hợp Báo cáo, Slide & Đóng gói nộp bài |  25%  |

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         PHÂN BỔ KHỐI LƯỢNG CÂN BẰNG (100%)                               │
├────────────────────┬────────────────────┬────────────────────┬───────────────────────────┤
│   TV1: Bảo (25%)   │  TV2: Long (25%)   │  TV3: Sang (25%)   │  TV4: Cường - NT (25%)    │
│  Data, EDA, Models │ Train Core & Exp 1 │ Exp 2 & Exp Logger │ Filter Viz, Eval & Lead   │
└────────────────────┴────────────────────┴────────────────────┴───────────────────────────┘
```

### 1. THÀNH VIÊN 1 – Nguyễn Đình Cường (`3123411043- Nhóm Trưởng`) – 25%
- **Mã nguồn phụ trách:**
  - `src/dataset.py`:
    - Hỗ trợ đọc trực tiếp `fer2013.csv` (tự động cache sang `.npz` để nạp nhanh) hoặc cấu trúc thư mục ảnh `Training/PublicTest/PrivateTest`.
    - Cài đặt `FER2013Dataset(torch.utils.data.Dataset)` trả về `(image_tensor, label_int)`.
    - Tính `mean/std` **chỉ trên tập `Training`**, cung cấp hàm `get_dataloaders(...)` và `compute_class_weights(train_labels)`.
  - `src/transforms.py`:
    - Cung cấp hàm `get_transforms(mode: str)` với 3 chế độ: `"none"`, `"standard"`, `"strong"` (và `eval_transform` cố định không augment).
  - `src/eda.py`:
    - Thống kê số lượng từng lớp ở 3 tập, phát hiện ảnh toàn đen/lỗi.
    - Xuất biểu đồ phân bố 7 lớp (`figures/eda_class_distribution.png`), lưới 32 ảnh mẫu Train/Val (`figures/eda_sample_images.png`), và lưới minh họa ảnh trước/sau Augmentation (`figures/eda_augmented_samples.png`).
  - `src/models.py`:
    - Xây dựng factory `build_model(name, num_classes=7, activation='relu', use_bn=True, dropout=0.2)` cho 6 mô hình:
      `SmallCNN`, `MediumCNN`, `Medium1CNN`, `LargeCNN`, `ImprovedFERNet` (VGG-BN / Residual + GAP), `VGG16Transfer` (Fail Case).
    - Đảm bảo **mọi mô hình đều expose thuộc tính hoặc phương thức `get_final_conv_layer()`** (bàn giao cho TV4 làm trực quan hóa tầng tích chập cuối cùng!).
    - Viết unit test kiểm tra forward pass với tensor `(4, 1, 48, 48)` và in bảng số tham số từng mô hình.
- **Báo cáo, Slide & Nhật ký:**
  - Viết **Mục 1** (Đặt vấn đề & Giới thiệu FER2013), **Mục 2** (Khám phá dữ liệu EDA, phân tích mất cân bằng lớp, tiền xử lý & Augmentation) và **Mục 3.1** (Thiết kế các kiến trúc CNN).
  - Thiết kế **Slide 1 – 4** (Trang bìa + phân công, Tổng quan bài toán & Dataset, EDA & Data Augmentation, Các kiến trúc CNN).
  - Ghi nhật ký phần kiểm tra dữ liệu và benchmark số tham số kiến trúc.

---

### 2. THÀNH VIÊN 2 – Châu Quốc Bảo (`3123411024`) – 25%
- **Mã nguồn phụ trách:**
  - `src/utils.py`: `seed_everything`, `get_device`, đếm số tham số trainable, định dạng thời gian chạy.
  - `src/metrics.py`: Tính `accuracy`, `macro_f1`, `weighted_f1`, bảng per-class Precision/Recall/F1, `confusion_matrix`.
  - `src/trainer.py`:
    - Cài đặt vòng lặp `train_model(...)` bằng PyTorch chuẩn (`train()`, `eval()`, `no_grad()`, AMP tùy chọn).
    - Cài đặt `EarlyStopping`, `ModelCheckpoint` (lưu state_dict + metadata đầy đủ), hỗ trợ đa dạng Optimizer & LR Scheduler.
    - Tự động xuất log từng epoch ra `logs/history_<exp_id>.csv` và ghi nhận dòng tóm tắt chuẩn schema vào `logs/experiment_runs.csv`.
  - `src/run_exp1_architectures.py`:
    - Chạy trọn bộ **Thực nghiệm Nhóm 1 (`EXP-01` đến `EXP-06`)**: `SmallCNN`, `MediumCNN`, `Medium1CNN`, `LargeCNN`, `ImprovedFERNet`, và Fail Case `VGG16Transfer`.
    - Xuất bảng tổng hợp `logs/exp1_architectures_summary.csv`.
    - Gọi hàm vẽ từ TV4 để xuất `figures/exp1_arch_comparison.png`, `figures/exp1_learning_curves.png` và `figures/exp1_failcase_vgg16.png`.
- **Báo cáo, Slide & Nhật ký:**
  - Viết **Mục 3.2** (Tóm tắt cấu trúc mã nguồn PyTorch, pipeline huấn luyện, Callbacks & thiết lập Baseline) và **Mục 4.1** (Kết quả Thực nghiệm Nhóm 1: So sánh kiến trúc & Giải thích vì sao Fine-tune VGG16 thất bại trên ảnh $48 \times 48$).
  - Thiết kế **Slide 5 – 7** (Pipeline PyTorch, Kết quả so sánh kiến trúc CNN, Phân tích Fail Case VGG16).
  - Ghi chi tiết các mục `EXP-01` → `EXP-06` vào Nhật ký thực nghiệm (kèm phân tích hiện tượng overfit/tốc độ hội tụ).

---

### 3. THÀNH VIÊN 3 – Nguyễn Hoàng Long (`3123411179`) – 25%
- **Mã nguồn phụ trách:**
  - `src/run_exp2_training_techniques.py`:
    - Nhận kiến trúc tốt nhất từ Nhóm 1 làm mốc, chạy trọn bộ **Thực nghiệm Nhóm 2 (`EXP-07` đến `EXP-17`)** theo nguyên tắc cô lập từng biến:
      - **2A (Activation):** `ReLU` vs `LeakyReLU` vs `GELU/ELU` vs `Sigmoid`.
      - **2B (Optimizer & LR):** `Adam` vs `AdamW + Cosine` vs `SGD + Momentum + Cosine`.
      - **2C (Augmentation):** `No-Aug` vs `Standard Aug` vs `Strong Aug (+ RandomErasing)`.
      - **2D (Regularization & Class Imbalance):** `No-BN` vs `With-BN`, `Weighted CrossEntropyLoss`, `Label Smoothing (0.1)`.
    - Xuất các bảng tổng hợp `logs/exp2_ablation_summary.csv` và 4 biểu đồ đối sánh:
      - `figures/exp2_activations.png`
      - `figures/exp2_optimizers.png`
      - `figures/exp2_augmentation.png`
      - `figures/exp2_regularization_imbalance.png`
  - **Quản trị Nhật ký thực nghiệm (`logs/NHAT_KY_THUC_NGHIEM.md` & `logs/experiment_runs.csv`):**
    - Thiết lập biểu mẫu Nhật ký thực nghiệm ngay từ Giai đoạn 1.
    - Theo dõi, hợp nhất, đối chiếu 100% số liệu giữa file CSV log máy sinh ra với văn bản nhật ký của cả 4 thành viên (ghi rõ ngày chạy, người chạy, phần cứng, giả thuyết trước khi chạy, kết quả đạt được, lỗi phát sinh & quyết định điều chỉnh cho bước sau).
- **Báo cáo, Slide & Nhật ký:**
  - Viết **Mục 4.2** (Kết quả Thực nghiệm Nhóm 2: phân tích chuyên sâu ảnh hưởng từng kỹ thuật) và **Mục 4.3** (Bảng tổng hợp toàn bộ thực nghiệm & cơ sở chọn ứng viên Best Model).
  - Thiết kế **Slide 8 – 10** (Ablation Study: Activation, Optimizer, Augmentation, xử lý mất cân bằng lớp `Disgust`).
  - Hoàn thiện file bàn giao chính thức `logs/NHAT_KY_THUC_NGHIEM.md` (và xuất kèm bản PDF nếu cần in/nộp).

---

### 4. THÀNH VIÊN 4 –  (`3123411256) – 25%
- **Mã nguồn phụ trách:**
  - **Bàn giao sớm ngay Giai đoạn 1 – `src/visualize.py`:**
    - Bộ hàm vẽ chuẩn 300 DPI dùng chung cho cả nhóm: `plot_learning_curves`, `plot_experiment_comparison`, `plot_confusion_matrix`, `plot_prediction_grid`.
  - **`src/visualize_filters.py` (Trọng tâm yêu cầu ĐỎ Slide 5 – "Visualize the final convolution filters"):**
    - **Phần A – Trọng số bộ lọc (Filter Kernels):**
      - Trực quan hóa lưới trọng số tầng tích chập đầu tiên (`first_conv`) để thấy các bộ lọc phát hiện biên/cạnh cơ bản (`figures/viz_first_conv_filters.png`).
      - Trực quan hóa các bộ lọc (channel weights $3\times 3$) có L2-norm cao nhất và biểu đồ phân bố năng lượng bộ lọc của **tầng tích chập cuối cùng (`final_conv`)** (`figures/viz_final_conv_filters_weights.png`).
    - **Phần B – Bản đồ đặc trưng (Final Conv Feature Maps):**
      - Dùng PyTorch `forward_hook` trích xuất activation maps tại tầng tích chập cuối cùng cho các ảnh khuôn mặt mẫu thuộc 7 cảm xúc (`figures/viz_final_conv_feature_maps.png`), chỉ rõ các kênh chuyên biệt bắt vùng miệng cười (`Happy`), nhíu mày (`Angry`), mở rộng mắt/miệng (`Surprise`).
    - **Phần C – Grad-CAM trên tầng tích chập cuối cùng:**
      - Tính Grad-CAM dựa trên gradient theo tầng `final_conv` chồng lên ảnh gốc $48\times 48$ cho cả 7 lớp cảm xúc và một số ca dự đoán sai (`figures/viz_gradcam_7emotions.png`), chứng minh trực quan mô hình có đang học đúng đặc trưng khuôn mặt hay bị đánh lừa bởi nền/watermark.
  - **`src/evaluate_best_model.py` (`EXP-18`):**
    - Chạy thẩm định ứng viên tổ hợp tốt nhất qua **3 seeds `[42, 100, 2024]`** trên `PublicTest` (có **cơ chế Fallback** nếu tổ hợp mới không vượt cấu hình tốt nhất đã ghi nhận).
    - Thử nghiệm so sánh suy luận thường vs **Test-Time Augmentation (TTA: ảnh gốc + lật ngang + crop nhẹ)** trên `PublicTest` để quyết định có bật TTA hay không.
    - Nạp checkpoint đại diện `checkpoints/best_model.pth` và đánh giá **một lần duy nhất trên `PrivateTest` (3.589 ảnh)**.
    - Xuất: `logs/best_model_private_metrics.csv`, `logs/best_model_per_class_metrics.csv`, `figures/best_confusion_matrix.png`, `figures/best_learning_curve.png`, `figures/best_true_cases.png` (32 ảnh đúng), `figures/best_wrong_cases.png` (32 ảnh sai kèm nhãn `Pred / True`).
  - **`main.ipynb`, `README.md`, `requirements.txt`:** Notebook chạy tổng hợp end-to-end và hướng dẫn tái lập chi tiết.
- **Báo cáo, Slide & Đóng gói:**
  - Viết **Mục 5** (Đánh giá Best Model trên PrivateTest, Trực quan hóa & diễn giải Final Convolution Filters/Grad-CAM, Phân tích True/Wrong Cases & bài toán nhiễu nhãn FER2013) và **Mục 6** (Kết luận, Hạn chế, Hướng phát triển, Bảng phân công & Tài liệu tham khảo).
  - Biên tập, thống nhất định dạng toàn bộ Báo cáo (`reports/BTLON02_BaoCao_FER2013.docx` & `.pdf`) và bộ Slide (`slides/BTLON02_Slide_FER2013.pptx` & `.pdf`, 14–16 slide, kèm kịch bản chia lượt thuyết trình 4 người).
  - Rà soát chéo toàn bộ sản phẩm và đóng gói file nén chuẩn nộp hệ thống: **`BTLON02_Emotion_NguyenDinhCuong_3123411043.zip`**.

---

## IV. CẤU TRÚC THƯ MỤC & ĐƯỜNG DẪN CHUẨN (CANONICAL PATHS)

Toàn nhóm thống nhất cấu trúc dự án đặt trong thư mục dự án (trước khi nén thành `BTLON02_Emotion_NguyenDinhCuong_3123411043.zip`):

```text
BTLON02_Emotion_NguyenDinhCuong_3123411043/
├── README.md                                  # Hướng dẫn cài đặt, tải FER2013, chạy tái lập & bảng phân công TV
├── requirements.txt                           # Danh sách thư viện PyTorch & phụ thuộc
├── plan.md                                    # File kế hoạch & phân công nhiệm vụ 4 thành viên
├── main.ipynb                                 # Notebook chạy tổng hợp (hỗ trợ Google Colab GPU & Local)
├── data/
│   └── README_DATA.md                         # Link tải FER2013 & hướng dẫn đặt fer2013.csv (không nén file CSV nặng nếu giới hạn dung lượng)
├── src/
│   ├── __init__.py
│   ├── utils.py                               # [TV2] Seed, device, param counter, timer
│   ├── dataset.py                             # [TV1] FER2013Dataset, DataLoader, class weights, normalization
│   ├── transforms.py                          # [TV1] Pipeline Augmentation (none / standard / strong / TTA)
│   ├── eda.py                                 # [TV1] Khảo sát dữ liệu, vẽ histogram & ảnh mẫu
│   ├── models.py                              # [TV1] Small/Medium/Medium1/Large/ImprovedFERNet/VGG16Transfer
│   ├── metrics.py                             # [TV2] Accuracy, Macro-F1, Per-class metrics, Confusion Matrix
│   ├── trainer.py                             # [TV2] PyTorch training loop, EarlyStopping, Checkpoint, CSVLogger
│   ├── visualize.py                           # [TV4] Hàm vẽ dùng chung (300 DPI)
│   ├── run_exp1_architectures.py              # [TV2] Thực nghiệm Nhóm 1 (EXP-01..EXP-06)
│   ├── run_exp2_training_techniques.py        # [TV3] Thực nghiệm Nhóm 2 (EXP-07..EXP-17)
│   ├── visualize_filters.py                   # [TV4] Trực quan hóa Final Conv Filters, Feature Maps & Grad-CAM
│   └── evaluate_best_model.py                 # [TV4] Thẩm định 3 seeds (PublicTest) & Đánh giá cuối (PrivateTest)
├── checkpoints/
│   └── best_model.pth                         # Trọng số mô hình tối ưu cuối cùng (kèm metadata kiến trúc)
├── logs/
│   ├── NHAT_KY_THUC_NGHIEM.md                 # [TV3 chủ trì + Cả nhóm] Nhật ký thực nghiệm chi tiết (có thể xuất thêm .pdf)
│   ├── experiment_runs.csv                    # Bảng log tự chứa toàn bộ EXP-01..EXP-18
│   ├── exp1_architectures_summary.csv         # [TV2] Tổng hợp Nhóm 1
│   ├── exp2_ablation_summary.csv              # [TV3] Tổng hợp Nhóm 2
│   ├── best_model_private_metrics.csv         # [TV4] Kết quả PrivateTest chính thức
│   ├── best_model_per_class_metrics.csv       # [TV4] Precision/Recall/F1 từng cảm xúc
│   └── history_EXP-*.csv                      # Lịch sử train_loss, train_acc, val_loss, val_acc, lr theo epoch
├── figures/                                   # Toàn bộ biểu đồ 300 DPI nhúng vào Báo cáo & Slide
│   ├── eda_class_distribution.png             # [TV1]
│   ├── eda_sample_images.png                  # [TV1]
│   ├── eda_augmented_samples.png              # [TV1]
│   ├── exp1_arch_comparison.png               # [TV2]
│   ├── exp1_learning_curves.png               # [TV2]
│   ├── exp1_failcase_vgg16.png                # [TV2]
│   ├── exp2_activations.png                   # [TV3]
│   ├── exp2_optimizers.png                    # [TV3]
│   ├── exp2_augmentation.png                  # [TV3]
│   ├── exp2_regularization_imbalance.png      # [TV3]
│   ├── best_learning_curve.png                # [TV4]
│   ├── best_confusion_matrix.png              # [TV4]
│   ├── best_true_cases.png                    # [TV4]
│   ├── best_wrong_cases.png                   # [TV4]
│   ├── viz_first_conv_filters.png             # [TV4]
│   ├── viz_final_conv_filters_weights.png     # [TV4]
│   ├── viz_final_conv_feature_maps.png        # [TV4]
│   └── viz_gradcam_7emotions.png              # [TV4]
├── reports/
│   ├── BTLON02_BaoCao_FER2013.docx            # [TV4 tổng hợp + Cả nhóm viết]
│   └── BTLON02_BaoCao_FER2013.pdf
└── slides/
    ├── BTLON02_Slide_FER2013.pptx             # [TV4 tổng hợp + Cả nhóm thiết kế]
    └── BTLON02_Slide_FER2013.pdf
```

---

## V. GIAO THỨC BÀN GIAO & SCHEMA NHẬT KÝ THỰC NGHIỆM

### 1. Giao ước API giữa các module (Tránh xung đột khi ghép code)
- **`models.py` (TV1 → TV2, TV3, TV4):**
  - `build_model(name: str, num_classes: int = 7, activation: str = "relu", use_bn: bool = True, dropout: float = 0.2) -> nn.Module`
  - Mọi lớp mô hình cài đặt phương thức `model.get_final_conv_layer() -> nn.Conv2d` và `model.get_first_conv_layer() -> nn.Conv2d` để `visualize_filters.py` trích xuất trực tiếp mà không phụ thuộc cứng tên biến.
- **`dataset.py` (TV1 → TV2, TV3, TV4):**
  - `get_dataloaders(data_path: str, aug_mode: str = "standard", batch_size: int = 64, num_workers: int = 2, seed: int = 42) -> Dict[str, DataLoader]` trả về `{"train": ..., "val": ..., "test": ...}`.
- **`trainer.py` (TV2 → TV3, TV4):**
  - `train_model(model, train_loader, val_loader, config: dict) -> dict` trả về:
    ```python
    {
        "exp_id": str,
        "best_epoch": int,
        "total_epochs": int,
        "best_val_loss": float,
        "best_val_acc": float,
        "best_val_macro_f1": float,
        "train_time_sec": float,
        "num_params": int,
        "checkpoint_path": str,
        "history_path": str,
    }
    ```

### 2. Schema chuẩn của Nhật ký thực nghiệm (`logs/experiment_runs.csv` & `logs/NHAT_KY_THUC_NGHIEM.md`)
- **File máy ghi (`logs/experiment_runs.csv`):**
  ```text
  exp_id, date, owner, group_name, model_name, num_params, activation, use_bn, dropout, optimizer, lr, weight_decay, scheduler, loss_type, aug_mode, batch_size, max_epochs, seed, device, best_epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, val_macro_f1, train_time_sec, checkpoint_path, history_path, notes
  ```
- **File tường thuật học thuật (`logs/NHAT_KY_THUC_NGHIEM.md`):** Với mỗi đợt thí nghiệm (Session), thành viên thực hiện ghi theo mẫu thống nhất:
  1. **Mã thí nghiệm & Người thực hiện & Thời gian & Môi trường (GPU/PyTorch version).**
  2. **Mục tiêu & Giả thuyết trước khi chạy** (Ví dụ: *"Kỳ vọng thêm BatchNorm và Residual giúp tăng độ sâu mà không bị giảm hiệu năng như Medium1CNN"*).
  3. **Cấu hình siêu tham số thay đổi so với Baseline.**
  4. **Kết quả định lượng** (`val_loss`, `val_acc`, `val_macro_f1`, `best_epoch`, thời gian).
  5. **Quan sát, sự cố gặp phải & Quyết định cho vòng tiếp theo** (Ví dụ: *"VGG16 overfit ngay epoch 8 khi train_acc đạt 98% nhưng val_acc dừng ở 60% → xác nhận Fail Case do 5 tầng MaxPool làm feature map còn 1×1"*).

---

## VI. BỐ CỤC BÁO CÁO & SLIDE THUYẾT TRÌNH

### 1. Cấu trúc Báo cáo khoa học (Dự kiến 18 – 25 trang)
1. **Trang bìa, Mục lục, Danh mục Hình/Bảng & Bảng phân công nhiệm vụ 4 thành viên** *(TV4 tổng hợp)*
2. **Chương 1: Giới thiệu bài toán & Tổng quan FER2013** *(TV1)*
3. **Chương 2: Khám phá dữ liệu (EDA), Tiền xử lý & Data Augmentation** *(TV1)*
   - Phân bố 7 lớp ở 3 tập, phân tích mất cân bằng lớp (`Disgust`), ảnh mẫu & ảnh sau Augmentation.
4. **Chương 3: Phương pháp, Kiến trúc CNN & Tóm tắt mã nguồn PyTorch** *(TV1: mục 3.1; TV2: mục 3.2)*
   - Sơ đồ kiến trúc 6 mô hình (`Small` → `ImprovedFERNet` & `VGG16Transfer`), tóm tắt cấu trúc module PyTorch, vòng lặp huấn luyện & Callbacks.
5. **Chương 4: Kết quả thực nghiệm & Nghiên cứu cắt bỏ (Ablation Study)** *(TV2: mục 4.1; TV3: mục 4.2, 4.3)*
   - 4.1. Thực nghiệm Nhóm 1: So sánh kiến trúc CNN & Phân tích Fail Case VGG16.
   - 4.2. Thực nghiệm Nhóm 2: Ảnh hưởng Activation, Optimizer/Scheduler, Data Augmentation, BatchNorm & xử lý mất cân bằng lớp.
   - 4.3. Bảng tổng hợp 17 cấu hình & lập luận chọn ứng viên Best Model.
6. **Chương 5: Đánh giá Best Model & Trực quan hóa Final Convolution Filters** *(TV4)*
   - 5.1. Thẩm định đa-seed trên `PublicTest` (so sánh có/không TTA) và kết quả ngoại kiểm cuối cùng trên `PrivateTest`.
   - 5.2. **Trực quan hóa Final Convolution Filters:** trọng số bộ lọc tầng đầu & tầng cuối, Feature Maps tầng cuối, và bản đồ nhiệt **Grad-CAM** trên 7 cảm xúc kèm giải thích cơ chế nhận diện.
   - 5.3. Confusion Matrix, Precision/Recall/F1 từng lớp, phân tích sâu True Cases & Wrong Cases (nhầm lẫn điển hình: `Fear ↔ Sad`, `Angry ↔ Disgust`, nhãn nhiễu/mơ hồ trong FER2013).
7. **Chương 6: Kết luận, Hạn chế, Hướng phát triển & Tài liệu tham khảo** *(TV4)*
8. **Phụ lục:** Trích dẫn Nhật ký thực nghiệm & hướng dẫn chạy lại mã nguồn.

### 2. Cấu trúc Slide thuyết trình (14 – 16 slide, thời lượng 12 – 15 phút)
- **Slide 1 – 4 (TV1 trình bày ~3 phút):** Giới thiệu nhóm & phân công → Bài toán FER2013 → EDA & mất cân bằng lớp + Augmentation → Thiết kế các kiến trúc CNN.
- **Slide 5 – 7 (TV2 trình bày ~3 phút):** Pipeline huấn luyện PyTorch & cấu hình Baseline → Kết quả Nhóm 1 (So sánh 5 kiến trúc CNN) → Phân tích Fail Case Fine-tune VGG16.
- **Slide 8 – 10 (TV3 trình bày ~3 phút):** Kết quả Nhóm 2 (Activation & Optimizer) → Kết quả Nhóm 2 (Augmentation, BatchNorm & xử lý lớp `Disgust`) → Tổng quan Nhật ký thực nghiệm & lựa chọn ứng viên tối ưu.
- **Slide 11 – 15 (TV4 trình bày ~4 phút):** Kết quả Best Model trên `PrivateTest` (so sánh benchmark demo & Kaggle) → **Trực quan hóa Final Convolution Filters, Feature Maps & Grad-CAM** → Confusion Matrix & Phân tích Wrong/True Cases → Kết luận, Hạn chế & Demo/Q&A.

---

## VII. TIẾN ĐỘ THỰC HIỆN THEO 4 GIAI ĐOẠN (TIMELINE 12 NGÀY)

| Giai đoạn | Thời gian | Nội dung công việc chính | Sản phẩm bàn giao (Milestone) |
|:---|:---:|:---|:---|
| **GĐ 1: Khởi tạo Core & EDA** | Ngày 1 – 3 | - **TV1:** Hoàn thiện `dataset.py`, `transforms.py`, `eda.py`, `models.py` (có `get_final_conv_layer()`) và xuất các hình EDA.<br>- **TV2:** Hoàn thiện `utils.py`, `metrics.py`, `trainer.py`, chạy smoke-test 2 epoch với `SmallCNN`.<br>- **TV3:** Tạo khung `logs/NHAT_KY_THUC_NGHIEM.md` & `logs/experiment_runs.csv`, viết khung `run_exp2_training_techniques.py`.<br>- **TV4:** Hoàn thiện `src/visualize.py` (bàn giao sớm cho TV1/TV2/TV3) và dựng khung Báo cáo/Slide. | Code Core chạy thông suốt 100%, unit test forward pass qua toàn bộ 6 model, có đủ hình EDA. |
| **GĐ 2: Thực thi Thực nghiệm Nhóm 1 & Nhóm 2** | Ngày 4 – 7 | - **TV2 (Ngày 4–5):** Chạy trọn bộ `EXP-01..EXP-06` trên GPU, chốt kiến trúc tốt nhất và bàn giao kết quả/checkpoint cho TV3.<br>- **TV3 (Ngày 5–7):** Chạy trọn bộ `EXP-07..EXP-17` (Ablation), xuất đủ bảng CSV & biểu đồ Nhóm 2; tổng hợp Nhật ký thực nghiệm.<br>- **TV1:** Hoàn thiện nháp Chương 1, 2, 3.1 Báo cáo & Slide 1–4. | Đầy đủ `logs/history_EXP-*.csv`, `exp1/exp2` summary, biểu đồ Nhóm 1 & 2, chốt cấu hình ứng viên Best Model. |
| **GĐ 3: Best Model, Filter Viz & Tổng hợp Tài liệu** | Ngày 8 – 10 | - **TV4:** Chạy `evaluate_best_model.py` (3 seeds `PublicTest` + đánh giá cuối `PrivateTest`), chạy `visualize_filters.py` (Final Conv Filters + Feature Maps + Grad-CAM), xuất Confusion Matrix & True/Wrong Cases.<br>- **TV2 & TV3:** Hoàn thiện Chương 3.2, 4.1, 4.2, 4.3 Báo cáo, Slide 5–10 & chốt `NHAT_KY_THUC_NGHIEM.md`.<br>- **TV4:** Viết Chương 5, 6, ghép hoàn chỉnh Báo cáo Word/PDF & Slide PPTX/PDF. | Hoàn thiện 100% số liệu, toàn bộ 18 hình ảnh 300 DPI, bản thảo đầy đủ Báo cáo, Slide và Nhật ký thực nghiệm. |
| **GĐ 4: Kiểm tra chéo, Tái lập & Đóng gói nộp bài** | Ngày 11 – 12 | - **Cả 4 thành viên:** Kiểm tra chéo (Peer Review) đối khớp mọi số liệu giữa Báo cáo – Slide – Nhật ký – CSV logs; chạy thử lại `main.ipynb` / script đánh giá từ đầu theo `README.md`.<br>- **TV4 (Nhóm trưởng):** Dọn dẹp file rác (`__pycache__`, checkpoint phụ quá nặng), giữ lại `checkpoints/best_model.pth`, kiểm tra bảng phân công thành viên xuất hiện rõ ở cả `README.md`, Báo cáo và Slide; nén toàn bộ thành **`BTLON02_Emotion_NguyenDinhCuong_3123411043.zip`**.<br>- **Cả nhóm:** Tổng duyệt thuyết trình 12–15 phút & chuẩn bị câu hỏi phản biện. | File nén `BTLON02_Emotion_NguyenDinhCuong_3123411043.zip` sẵn sàng nộp đúng hạn. |

---

## VIII. DANH MỤC KIỂM TRA TRƯỚC KHI NỘP (PRE-SUBMISSION CHECKLIST)

- [ ] **Tên file nén chính xác:** `BTLON02_Emotion_NguyenDinhCuong_3123411043.zip` (nhóm trưởng Nguyễn Đình Cường – `3123411043` nộp).
- [ ] **Đủ 4 thành phần bắt buộc bên trong file nén:**
  - [ ] **Report:** `reports/BTLON02_BaoCao_FER2013.docx` và `.pdf` (có tóm tắt code, kết quả nhiều thực nghiệm, **trực quan hóa final convolution filters**, phân tích riêng của nhóm, bảng phân công thành viên).
  - [ ] **Slide:** `slides/BTLON02_Slide_FER2013.pptx` và `.pdf` (rõ trục/nhãn biểu đồ, có phân công thành viên).
  - [ ] **Nhật ký thực nghiệm:** `logs/NHAT_KY_THUC_NGHIEM.md` (kèm `logs/experiment_runs.csv` và các file `history_EXP-*.csv`).
  - [ ] **Code PyTorch:** Thư mục `src/`, `main.ipynb`, `README.md`, `requirements.txt`, `checkpoints/best_model.pth` (thuần PyTorch, chạy lại được, có ghi rõ phân công từng file).
- [ ] **Kiểm định khoa học:** Không dùng `PrivateTest` để tune; mọi bảng biểu trong Báo cáo & Slide truy vết chính xác 100% từ `logs/*.csv`.
- [ ] **Dung lượng file zip hợp lý:** Không đóng gói file raw `fer2013.csv` (~300MB) hay hàng chục file checkpoint trung gian nếu hệ thống nộp bài giới hạn dung lượng (ghi rõ hướng dẫn tải `fer2013.csv` trong `data/README_DATA.md` và `README.md`).
