# KẾ HOẠCH & PHÂN CÔNG NHIỆM VỤ CHUYỂN ĐỔI MÃ NGUỒN TỪ KERAS SANG PYTORCH
## ĐỀ TÀI: CHUYỂN ĐỔI BỘ VÍ DỤ HỌC SÂU (MLP, CNN, CNN+LSTM, SEGMENTATION) TỪ KERAS 2.x/TF 1.x SANG PYTORCH 2.x

> **Lớp:** DCT123C4 | **Môn học:** Học sâu (Deep Learning) | **GVHD:** Đỗ Như Tài  
> **Quy mô nhóm:** 04 thành viên (25% khối lượng thực chất / thành viên)  
> **Nguồn tài liệu gốc trong thư mục `KerasToPytorch/`:**  
> - `01_Introduction-to-Python-TensorFlow-and-Keras.pdf` (37 trang – Cú pháp cơ bản Python, TensorFlow/Keras, MLP MNIST)  
> - `02_KerasExamples.pdf` (57 trang – 6 bài toán thực hành: MLP, CNN #1/#2/#3, CNN+LSTM, SegNet Segmentation)  
> - `README (1).md` (Hướng dẫn cài đặt môi trường gốc Python 3.5 / TF 1.1 / Keras 2.0.9 / CUDA 8.0)  
> **Sản phẩm nộp bắt buộc:** File Báo cáo (`.docx` & `.pdf`) + Toàn bộ Code đã chuyển sang PyTorch (`.ipynb` & `.py`) + Các file liên quan (`README.md` cập nhật cho PyTorch, `requirements.txt`, `logs/`, `figures/`, bảng phân công thành viên).

---

## I. PHẠM VI CHUYỂN ĐỔI & MỐC ĐỐI CHIẾU TỪ TÀI LIỆU GỐC

Toàn bộ bộ tài liệu Keras gốc gồm **1 phần cú pháp nền tảng + 6 bài toán thực hành** trải dài từ MLP, CNN, kỹ thuật chống Overfitting, chuỗi thời gian (TimeDistributed + LSTM) đến phân đoạn ảnh (SegNet):

| Mã bài | Tên bài toán gốc (trong Slide) | Kiến trúc & Kỹ thuật Keras cần chuyển sang PyTorch | Input / Output | Mốc Keras gốc (Slide) | Phụ trách |
|:---:|:---|:---|:---|:---|:---:|
| **Bài 0** | Cú pháp cơ bản & Kiểm tra GPU (`01_Intro`, tr. 18–35) | `Input`, `Dense`, `Flatten`, `Conv2D`, `MaxPooling2D`, `Dropout`, `LSTM`, `summary()`, `compile/fit/evaluate`, kiểm tra GPU | Ví dụ minh họa từng tầng | Khớp shape & số tham số | **TV1** |
| **Bài 1** | **MLP – MNIST Classification** (`02_Examples`, tr. 11–20) | `Flatten(784) → FC(512, ReLU) → FC(512, ReLU) → FC(10)`, `SGD(lr=0.01)`, `batch=128`, `epochs=20` | `(1, 28, 28)` → 10 lớp | Loss `0.1128` \| Acc **`96.57%`** | **TV1** |
| **Bài 2** | **CNN #1 – MNIST Classification** (`02_Examples`, tr. 22–26) | 3 khối `Conv2D(3×3, same, ReLU) + MaxPool(2×2)` + `Dropout` → `Flatten → FC(512) → FC(512) → FC(10)` | `(1, 28, 28)` → 10 lớp | Loss `0.0319` \| Acc **`98.89%`** | **TV1** |
| **Bài 3** | **CNN #2 – Human Gender Prediction** (`02_Examples`, tr. 27–36) | Khối `Conv2D(16→...) + MaxPool(2×2)` → `FC(1024) → FC(1024) → FC(2)`<br>• **Bản A:** Baseline không kỹ thuật<br>• **Bản B:** Thêm `EarlyStopping` + `ReduceLROnPlateau` + demo ảnh thật (tr. 31) | `(3, 64, 64)` → 2 lớp (Nam/Nữ) | Bản A: **`~82.64%`** (Loss `0.4602`)<br>Bản B: **`~87.73%`** (Loss `0.3237`) | **TV2** |
| **Bài 4** | **CNN #3 – Facial Expression Recognition** (`02_Examples`, tr. 37–44) | Khối `Conv2D(16→...) + MaxPool(2×2)` → `FC(1024) → FC(7)`<br>• **Bản A:** Không Augmentation<br>• **Bản B:** Thay `ImageDataGenerator` bằng `torchvision.transforms` | `(1, 48, 48)` → 7 cảm xúc (FER2013) | Bản A: **`59.54%`** (Loss `1.4070`)<br>Bản B: **`61.63%`** (Loss `1.0158`, +2%) | **TV2** |
| **Bài 5** | **CNN + LSTM – MNIST Summation** (`02_Examples`, tr. 46–50) | `TimeDistributed(Conv2D + MaxPool + Flatten)` → `LSTM(128)` → `Linear(1)`<br>Hàm mất mát hồi quy: **MAE (`L1Loss`)** / **MSE (`MSELoss`)** | `(7, 1, 28, 28)` → 1 số thực (tổng 7 chữ số) | Test MAE **`0.6502`** (10.000 chuỗi test) | **TV3** |
| **Bài 6** | **Segmentation – Hand Depth (SegNet)** (`02_Examples`, tr. 52–56) | Encoder (`Conv + MaxPool` × 3) → Decoder (`Conv + UpSampling2D` × 3) → Phân loại từng pixel (`Conv 1×1`) | `(1, 96, 96)` → Mask `(C, 96, 96)` | Loss `0.6860` \| Pixel Acc **`77.32%`** (1.050 ảnh) | **TV4** |

> **Lưu ý về nguồn dữ liệu công khai (Reproducibility):**
> - **Bài 1, 2, 5 (MNIST):** Tải tự động qua `torchvision.datasets.MNIST` (không cần tải tay).
> - **Bài 3 (Gender 64×64):** Sử dụng bộ dữ liệu khuôn mặt công khai có nhãn giới tính (ví dụ: UTKFace / Kaggle Gender Classification Dataset), resize chuẩn về $64 \times 64$, chia tập Train/Val/Test cố định (`seed=42`) và ghi rõ nguồn trong `data/README_DATA.md` + Báo cáo.
> - **Bài 4 (FER2013 48×48):** Dùng đúng bộ Kaggle FER2013 (`Training=28.709`, `PublicTest=3.589`, `PrivateTest=3.589`) – có thể dùng chung nguồn dữ liệu với BTL02 nhưng giữ thư mục code độc lập.
> - **Bài 6 (Hand Depth Segmentation 96×96):** Ưu tiên tải đúng bộ dữ liệu Hand Depth Segmentation từ repo nghiên cứu của tác giả (`nhduong/multitask_segnet_hand_segmentation_fingertip_detection`); nếu đường dẫn dữ liệu cũ (2018) không còn hoạt động, sử dụng bộ dữ liệu phân đoạn bàn tay công khai tương đương (ví dụ EgoHands / Kaggle Hand Segmentation / NYU Hand Depth resize $96 \times 96$) và ghi chú minh bạch trong Báo cáo.

---

## II. QUY CHUẨN ÁNH XẠ KỸ THUẬT KERAS ↔ PYTORCH (MAPPING STANDARD)

Toàn bộ 4 thành viên thống nhất áp dụng đúng bảng quy đổi kỹ thuật sau để tránh sai sót phổ biến khi port từ Keras sang PyTorch:

| Thành phần | Keras 2.x (Mã gốc) | PyTorch 2.x (Quy chuẩn nhóm) | Lưu ý tránh lỗi (Pitfalls) |
|:---|:---|:---|:---|
| **Thứ tự chiều Tensor** | `NHWC`: `(B, H, W, C)` | `NCHW`: `(B, C, H, W)` | Luôn kiểm tra shape đầu vào Conv2d; dùng `ToTensor()` tự chuyển `HWC → CHW` và chia `255.0`. |
| **Định nghĩa mô hình** | `Input()` + Functional API + `Model(in, out)` | Kế thừa `nn.Module` (`__init__`, `forward`) | Đặt tên layer tường minh tương ứng tên trong slide (`conv1`, `pool1`, `fc1`...). |
| **Tầng tích chập** | `Conv2D(f, (3,3), padding='same', activation='relu')` | `nn.Conv2d(in_c, f, kernel_size=3, padding=1)` + `nn.ReLU(inplace=True)` | PyTorch tách rời `Conv2d` và `ReLU`; phải truyền rõ `in_channels`. |
| **Tầng gộp & Dropout** | `MaxPooling2D((2,2))`, `Dropout(0.2)` | `nn.MaxPool2d(2, 2)`, `nn.Dropout(p=0.2)` | Nhớ gọi `model.train()` khi huấn luyện và `model.eval()` khi đánh giá để tắt Dropout. |
| **Tầng kết nối đầy đủ** | `Flatten()`, `Dense(512, activation='relu')` | `nn.Flatten()`, `nn.Linear(in_feat, 512)` + `nn.ReLU()` | Tính chính xác `in_feat` sau `Flatten` (hoặc dùng `nn.LazyLinear` kiểm thử rồi khóa số cố định). |
| **TimeDistributed** | `TimeDistributed(layer)(x)` | Class `TimeDistributed(nn.Module)` tự cài đặt: reshape `(B, T, ...)` → `(B*T, ...)`, gọi `layer`, reshape ngược `(B, T, ...)` | PyTorch không có sẵn `TimeDistributed`; bắt buộc viết module gọn sạch, có unit test shape. |
| **LSTM** | `LSTM(units=128)` (mặc định trả bước cuối) | `nn.LSTM(input_size, 128, batch_first=True)` → lấy `out[:, -1, :]` | **Cực kỳ quan trọng:** phải bật `batch_first=True` nếu tensor là `(B, T, F)` và nhớ `nn.LSTM` trả về tuple `(output, (h_n, c_n))`. |
| **UpSampling2D** | `UpSampling2D((2,2))` | `nn.Upsample(scale_factor=2, mode='nearest')` | Giữ đúng `mode='nearest'` để khớp hành vi mặc định của Keras `UpSampling2D`. |
| **Loss phân loại** | `Dense(C, 'softmax')` + `'categorical_crossentropy'` (nhãn one-hot) | **Raw logits** (`Linear` cuối, **KHÔNG** có `Softmax`) + `nn.CrossEntropyLoss()` (nhãn `LongTensor` chỉ số `0..C-1`) | Nếu thêm `Softmax` trước `CrossEntropyLoss` sẽ bị áp softmax 2 lần làm tụt mạnh độ chính xác! |
| **Loss hồi quy** | `'mean_absolute_error'` / `'mean_squared_error'` | `nn.L1Loss()` / `nn.MSELoss()` | Đảm bảo shape `pred` và `target` khớp nhau `(B, 1)` tránh lỗi broadcasting ngầm. |
| **Loss phân đoạn** | Softmax theo kênh + Pixel Categorical CE | Raw logits `(B, C, H, W)` + `nn.CrossEntropyLoss()` với mask `LongTensor` `(B, H, W)` | Không cần flatten thủ công; `CrossEntropyLoss` hỗ trợ trực tiếp đầu vào 4D `(B, C, H, W)`. |
| **Model Summary** | `model.summary()` | `torchinfo.summary(model, input_size=...)` | In rõ Output Shape và tổng số tham số để đối chiếu 1-1 với slide Keras. |
| **Compile / Fit / Eval** | `model.compile()`, `model.fit()`, `model.evaluate()` | Viết hàm dùng chung `fit()` & `evaluate()` trong `src/pytorch_core.py` | Đảm bảo thứ tự: `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`. |
| **Callbacks** | `EarlyStopping`, `ReduceLROnPlateau` | Class `EarlyStopping` tự viết + `torch.optim.lr_scheduler.ReduceLROnPlateau` | Gọi `scheduler.step(val_loss)` sau mỗi epoch validation; khôi phục `best_state_dict` khi dừng. |
| **Data Augmentation** | `ImageDataGenerator` + `fit_generator()` | `torchvision.transforms` gắn trong `Dataset.__getitem__` + `DataLoader(shuffle=True)` | Chỉ áp dụng augmentation cho tập Train; tuyệt đối không augment tập Val/Test. |
| **Khởi tạo trọng số** | Mặc định Keras dùng **Glorot/Xavier Uniform** | Mặc định PyTorch dùng **Kaiming Uniform** | Ghi nhận sự khác biệt nhỏ về tốc độ hội tụ/kết quả trong phần phân tích của Báo cáo! |

---

## III. PHÂN CÔNG NHIỆM VỤ CHI TIẾT CHO 4 THÀNH VIÊN (25% × 4)

| STT | Họ và tên | MSSV | Nhiệm vụ kỹ thuật & Code PyTorch | Nhiệm vụ Báo cáo & Tài liệu | Tỷ lệ |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | **Châu Quốc Bảo** | `3123411024` | **TV1:** Core `pytorch_core.py`, `00_keras_vs_pytorch_basics.ipynb`, **Bài 1 (MLP MNIST)** & **Bài 2 (CNN MNIST)** | Viết Chương 1 (Bảng ánh xạ API) & Chương 2 (Bài 1, 2) | 25% |
| 2 | **Nguyễn Hoàng Long** | `3123411179` | **TV2:** **Bài 3 (Gender CNN: Baseline vs EarlyStopping+ReduceLR)** & **Bài 4 (FER CNN: No-Aug vs Augmentation)** | Viết Chương 3 (Bài 3, 4 & minh chứng 3 kỹ thuật huấn luyện) | 25% |
| 3 | **Võ Huỳnh Minh Sang** | `3123411256` | **TV3:** **Bài 5 (CNN + TimeDistributed + LSTM MNIST Summation)**, bộ unit test kiểm tra shape/tái lập & bảng log tổng hợp đối chiếu Keras vs PyTorch | Viết Chương 4 (Bài 5 CNN-LSTM) & mục phân tích khác biệt cơ chế Keras vs PyTorch | 25% |
| 4 | **Nguyễn Đình Cường** *(Nhóm trưởng)* | `3123411043` | **TV4:** **Bài 6 (Hand Depth Segmentation – SegNet)**, cập nhật `README.md` (PyTorch 2.x + Colab), `requirements.txt`, `visualize.py` | Viết Chương 5 (Bài 6 Segmentation) & Chương 6 (Tổng kết, phân công), biên tập Báo cáo & đóng gói nộp bài | 25% |

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                     PHÂN BỔ KHỐI LƯỢNG CHUYỂN ĐỔI KERAS → PYTORCH                        │
├────────────────────┬────────────────────┬────────────────────┬───────────────────────────┤
│   TV1: Bảo (25%)   │  TV2: Long (25%)   │  TV3: Sang (25%)   │  TV4: Cường - NT (25%)    │
│ Core + Bài 0, 1, 2 │ Bài 3 & Bài 4 CNN  │ Bài 5 LSTM + Tests │ Bài 6 SegNet + Lead/Pack  │
└────────────────────┴────────────────────┴────────────────────┴───────────────────────────┘
```

### Chi tiết đầu việc từng thành viên

#### 1. TV1 – Châu Quốc Bảo (`3123411024`) – 25%
- **Code bàn giao sớm (Giai đoạn 1):**
  - `src/pytorch_core.py`: `seed_everything(seed=42)`, `get_device()`, `EarlyStopping` (tự động lưu/khôi phục `best_model_state`), `train_one_epoch`, `evaluate`, `fit` (hỗ trợ cả bài toán **Classification**, **Regression**, **Segmentation**, tự lưu lịch sử loss/metric ra CSV).
- **Chuyển đổi code & Notebook:**
  - `notebooks/00_keras_vs_pytorch_basics.ipynb`: Chuyển toàn bộ đoạn code minh họa trong `01_Introduction-to-Python-TensorFlow-and-Keras.pdf` (kiểm tra PyTorch/CUDA GPU, so sánh định nghĩa tầng Conv2d/Linear/Dropout/LSTM/Flatten, `torchinfo.summary`).
  - `src/models/mnist_mlp.py` & `notebooks/01_mnist_mlp.ipynb`: Chuyển **Bài 1 (MLP MNIST)** – khớp kiến trúc `784→512→512→10`, `SGD(lr=0.01)`, `batch=128`, `epochs=20`, đối chiếu mốc **`96.57%`**.
  - `src/models/mnist_cnn.py` & `notebooks/02_mnist_cnn.ipynb`: Chuyển **Bài 2 (CNN #1 MNIST)** – đối chiếu mốc **`98.89%`** và so sánh trực tiếp với MLP (đường cong học tập + thời gian chạy + số tham số).
- **Báo cáo:** Viết **Chương 1** (Cơ sở lý thuyết & Bảng ánh xạ cú pháp Keras ↔ PyTorch đầy đủ kèm code đối chiếu cạnh nhau) và **Chương 2** (Kết quả Bài 1 & Bài 2).

#### 2. TV2 – Nguyễn Hoàng Long (`3123411179`) – 25%
- **Chuyển đổi code & Notebook:**
  - `src/models/gender_cnn.py` & `notebooks/03_gender_cnn.ipynb`: Chuyển **Bài 3 (CNN #2 – Human Gender Prediction, $64\times 64$ RGB, 2 lớp)**:
    - Chạy và lưu kết quả **2 cấu hình**:
      1. *Baseline* (không callback): đối chiếu mốc Keras `~82.6%`.
      2. *Có `EarlyStopping` + `ReduceLROnPlateau`*: đối chiếu mốc Keras `~87.7%`, vẽ đồ thị learning rate giảm theo epoch và đường cong train/val loss chứng minh hết overfit.
    - Chạy suy luận demo trên vài ảnh mẫu kèm xác suất `Softmax` dự đoán (tương ứng Slide 31).
  - `src/models/emotion_cnn.py` & `notebooks/04_emotion_cnn.ipynb`: Chuyển **Bài 4 (CNN #3 – FER2013, $48\times 48\times 1$, 7 lớp)**:
    - Chạy và lưu kết quả **2 cấu hình**:
      1. *Không Data Augmentation*: đối chiếu mốc Keras `~59.5%`.
      2. *Có Data Augmentation (`torchvision.transforms`: Flip, Rotation, Affine)* thay cho `ImageDataGenerator`: đối chiếu mức tăng accuracy (`~61.6%+`) và giảm khoảng cách train–val gap.
- **Báo cáo:** Viết **Chương 3** (Trình bày chi tiết chuyển đổi Bài 3 & Bài 4, phân tích cách chuyển 3 kỹ thuật huấn luyện `EarlyStopping`, `ReduceLROnPlateau`, `ImageDataGenerator → torchvision.transforms`).

#### 3. TV3 – Võ Huỳnh Minh Sang (`3123411256`) – 25%
- **Chuyển đổi code & Notebook:**
  - `src/models/mnist_lstm.py` & `notebooks/05_mnist_lstm_sum.ipynb`: Chuyển **Bài 5 (CNN + LSTM – MNIST Sequence Summation)**:
    - Cài đặt module **`TimeDistributed(nn.Module)`** tái sử dụng được trong PyTorch.
    - Cài đặt `MNISTSequenceSumDataset`: tạo chuỗi cố định $T=7$ ảnh `(7, 1, 28, 28)` từ tập Train MNIST (50.000–60.000 chuỗi) và tập Test MNIST độc lập (đúng **10.000 chuỗi test** như Slide 50).
    - Huấn luyện với **`nn.L1Loss()` (MAE)**, đối chiếu mốc Keras Test MAE **`0.6502`**, đồng thời thử nghiệm phiên bản cải thiện có `ReduceLROnPlateau` + `EarlyStopping` theo gợi ý Slide 50.
    - Xuất biểu đồ hội tụ MAE và lưới minh họa 8–10 chuỗi mẫu kèm `True Sum` vs `Predicted Sum`.
  - **Kiểm thử tự động & Bảng tổng hợp đối chiếu:**
    - `tests/test_models_contract.py`: Unit test tự động kiểm tra forward pass, output shape, số tham số và khả năng backprop 1 bước cho **toàn bộ 6 mô hình** của cả 4 thành viên.
    - Tổng hợp file `logs/comparison_keras_vs_pytorch.csv` đối chiếu đầy đủ Loss / Accuracy / MAE giữa Slide Keras gốc và kết quả PyTorch chạy thực tế của nhóm.
- **Báo cáo:** Viết **Chương 4** (Chuyển đổi Bài 5 CNN-LSTM, giải thích kỹ thuật `TimeDistributed` trong PyTorch & bài toán hồi quy MAE) và **Mục 6.1** (Phân tích các khác biệt cốt lõi giữa Keras và PyTorch ảnh hưởng đến kết quả thực nghiệm).

#### 4. TV4 – Nguyễn Đình Cường (`3123411043` – Nhóm trưởng) – 25%
- **Code bàn giao sớm (Giai đoạn 1):**
  - `src/visualize.py`: Bộ hàm vẽ đồ thị thống nhất style (300 DPI) cho cả nhóm: đường cong Train/Val Loss & Accuracy, biểu đồ so sánh Before/After kỹ thuật huấn luyện, lưới ảnh dự đoán và lưới Segmentation Mask.
- **Chuyển đổi code & Notebook:**
  - `src/models/segnet.py` & `notebooks/06_hand_segmentation_segnet.ipynb`: Chuyển **Bài 6 (Segmentation – Hand Depth Image Segmentation / SegNet)**:
    - Cài đặt kiến trúc Encoder–Decoder đối xứng (`Conv2d + ReLU + MaxPool2d` × 3 và `Conv2d + ReLU + nn.Upsample(scale_factor=2, mode='nearest')` × 3) cho ảnh đầu vào `(1, 96, 96)`.
    - Đánh giá bằng **Pixel Accuracy** (đối chiếu mốc Keras **`77.32%`**) kèm **Mean IoU (mIoU)** để đánh giá khách quan hơn với nền ảnh.
    - Trực quan hóa lưới 3 cột: *Ảnh đầu vào ($96\times 96$) | Ground Truth Mask | Predicted Mask*.
- **Tài liệu hóa, Biên tập & Đóng gói:**
  - Cập nhật `README.md` (thay thế hoàn toàn `README (1).md` cũ của TF 1.1/Keras 2.0.9/CUDA 8 bằng hướng dẫn cài đặt **PyTorch 2.x + CUDA 12.x + Google Colab + cách chạy từng notebook** và bảng phân công 4 thành viên).
  - Tạo `requirements.txt`.
  - Viết **Chương 5** (Chuyển đổi Bài 6 Segmentation – SegNet) và **Chương 6** (Bảng tổng kết đối chiếu 6 bài, kết luận & phân công đóng góp).
  - Tổng hợp, định dạng thống nhất **Báo cáo (`reports/BaoCao_KerasToPyTorch.docx` & `.pdf`)**, chạy kiểm tra chéo tất cả notebook và đóng gói toàn bộ sản phẩm nộp bài.

---

## IV. CẤU TRÚC THƯ MỤC SẢN PHẨM CHUẨN

```text
KerasToPytorch/
├── README.md                                  # [TV4] Hướng dẫn môi trường PyTorch 2.x (thay thế README cũ) + phân công TV
├── requirements.txt                           # [TV4] torch, torchvision, torchinfo, numpy, pandas, matplotlib, scikit-learn, tqdm
├── plan.md                                    # File kế hoạch & phân công nhiệm vụ 4 thành viên
├── reference_original/                        # Lưu giữ tài liệu gốc giảng viên cung cấp để đối chiếu
│   ├── 01_Introduction-to-Python-TensorFlow-and-Keras.pdf
│   ├── 02_KerasExamples.pdf
│   └── README (1).md
├── data/
│   └── README_DATA.md                         # Hướng dẫn tự động tải MNIST & link/cách đặt dữ liệu Gender, FER2013, Hand Depth
├── src/
│   ├── __init__.py
│   ├── pytorch_core.py                        # [TV1] seed_everything, device, EarlyStopping, fit(), evaluate()
│   ├── visualize.py                           # [TV4] Hàm vẽ learning curves, so sánh before/after, segmentation masks
│   └── models/
│       ├── __init__.py
│       ├── mnist_mlp.py                       # [TV1] Bài 1: MLP MNIST
│       ├── mnist_cnn.py                       # [TV1] Bài 2: CNN #1 MNIST
│       ├── gender_cnn.py                      # [TV2] Bài 3: CNN #2 Gender
│       ├── emotion_cnn.py                     # [TV2] Bài 4: CNN #3 FER2013
│       ├── mnist_lstm.py                      # [TV3] Bài 5: TimeDistributed + CNN + LSTM
│       └── segnet.py                          # [TV4] Bài 6: SegNet Segmentation
├── notebooks/                                 # Toàn bộ notebook đã chuyển sang PyTorch (đã chạy sẵn output rõ ràng)
│   ├── 00_keras_vs_pytorch_basics.ipynb       # [TV1]
│   ├── 01_mnist_mlp.ipynb                     # [TV1]
│   ├── 02_mnist_cnn.ipynb                     # [TV1]
│   ├── 03_gender_cnn.ipynb                    # [TV2]
│   ├── 04_emotion_cnn.ipynb                   # [TV2]
│   ├── 05_mnist_lstm_sum.ipynb                # [TV3]
│   └── 06_hand_segmentation_segnet.ipynb      # [TV4]
├── tests/
│   └── test_models_contract.py                # [TV3] Kiểm thử tự động shape & forward/backward cho cả 6 model
├── logs/
│   ├── comparison_keras_vs_pytorch.csv        # [TV3] Bảng tổng hợp kết quả Keras (slide) vs PyTorch (nhóm chạy)
│   └── history_01..06_*.csv                   # Lịch sử train/val loss & metric theo epoch của 6 bài
├── figures/                                   # Toàn bộ biểu đồ 300 DPI nhúng vào Báo cáo
│   ├── 01_mnist_mlp_curves.png                # [TV1]
│   ├── 02_mnist_cnn_vs_mlp.png                # [TV1]
│   ├── 03_gender_before_after_callbacks.png   # [TV2]
│   ├── 03_gender_demo_predictions.png         # [TV2]
│   ├── 04_emotion_before_after_aug.png        # [TV2]
│   ├── 05_mnist_lstm_mae_curve.png            # [TV3]
│   ├── 05_mnist_lstm_samples.png              # [TV3]
│   ├── 06_segnet_loss_acc.png                 # [TV4]
│   └── 06_segnet_mask_predictions.png         # [TV4]
└── reports/
    ├── BaoCao_KerasToPyTorch.docx             # [TV4 tổng hợp + Cả nhóm viết]
    └── BaoCao_KerasToPyTorch.pdf
```

---

## V. BỐ CỤC BÁO CÁO CHUYỂN ĐỔI (15 – 20 TRANG)

1. **Trang bìa, Mục lục & Bảng phân công nhiệm vụ 4 thành viên** *(TV4)*
2. **Chương 1: Tổng quan & Bảng đối chiếu cú pháp Keras ↔ PyTorch** *(TV1)*
   - Giới thiệu mục tiêu nâng cấp từ môi trường cũ (Python 3.5, TF 1.1, Keras 2.0.9) sang PyTorch 2.x.
   - Bảng ánh xạ chi tiết từng thành phần (Tensor NCHW, `nn.Module`, Layers, `TimeDistributed`, `Upsample`, Loss, Callbacks, Data Augmentation) kèm đoạn mã đối sánh 2 cột Keras vs PyTorch.
3. **Chương 2: Chuyển đổi Bài 1 (MLP MNIST) & Bài 2 (CNN #1 MNIST)** *(TV1)*
   - Kiến trúc `torchinfo.summary`, đường cong huấn luyện, đối chiếu kết quả (`96.57%` và `98.89%`) và so sánh hiệu quả CNN vs MLP.
4. **Chương 3: Chuyển đổi Bài 3 (Gender CNN) & Bài 4 (FER2013 CNN)** *(TV2)*
   - Chứng minh hiệu quả chuyển đổi `EarlyStopping` + `ReduceLROnPlateau` (Bài 3) và `ImageDataGenerator → torchvision.transforms` (Bài 4) qua biểu đồ Before/After.
5. **Chương 4: Chuyển đổi Bài 5 (CNN + TimeDistributed + LSTM – MNIST Summation)** *(TV3)*
   - Cách xây dựng wrapper `TimeDistributed` trong PyTorch, thiết kế dataset chuỗi 7 ảnh, huấn luyện hồi quy MAE (`L1Loss`) và đối chiếu mốc `0.6502`.
6. **Chương 5: Chuyển đổi Bài 6 (Hand Depth Image Segmentation – SegNet)** *(TV4)*
   - Chuyển đổi `UpSampling2D` sang `nn.Upsample`, hàm mất mát phân đoạn từng pixel, kết quả Pixel Accuracy / mIoU và trực quan hóa mặt nạ phân đoạn.
7. **Chương 6: Tổng hợp đối chiếu toàn bộ 6 bài, Phân tích khác biệt Keras–PyTorch & Kết luận** *(TV3: mục 6.1; TV4: tổng hợp & 6.2)*
   - Bảng tổng hợp duy nhất đối chiếu toàn bộ 6 bài (Keras gốc vs PyTorch mới: Loss, Accuracy/MAE, Thời gian chạy, Nhận xét).
   - Giải thích các điểm khác biệt kỹ thuật (Weight initialization, Softmax/CrossEntropyLoss, DataLoader parallelism) & hướng phát triển.

---

## VI. TIẾN ĐỘ THỰC HIỆN (TIMELINE 8 NGÀY)

| Giai đoạn | Thời gian | Nội dung công việc chính | Mốc bàn giao (Milestone) |
|:---|:---:|:---|:---|
| **GĐ 1: Core & Khung chung** | Ngày 1 – 2 | - **TV1:** Hoàn thiện `src/pytorch_core.py`, `00_keras_vs_pytorch_basics.ipynb` và chạy xong Bài 1 (MLP MNIST).<br>- **TV4:** Hoàn thiện `src/visualize.py`, `requirements.txt` và chuẩn bị nguồn dữ liệu cho Bài 6.<br>- **TV2 & TV3:** Chuẩn bị bộ dữ liệu Bài 3, 4, 5 và dựng khung class model PyTorch tương ứng. | `pytorch_core.py` & `visualize.py` chạy thông suốt; 4 thành viên thống nhất API và chuẩn dữ liệu. |
| **GĐ 2: Chuyển đổi & Huấn luyện 6 Bài** | Ngày 3 – 5 | - **TV1:** Hoàn thiện Bài 2 (`02_mnist_cnn.ipynb`), xuất logs & figures.<br>- **TV2:** Hoàn thiện Bài 3 (`03_gender_cnn.ipynb`, 2 bản) & Bài 4 (`04_emotion_cnn.ipynb`, 2 bản), xuất logs & figures.<br>- **TV3:** Hoàn thiện Bài 5 (`05_mnist_lstm_sum.ipynb` + `TimeDistributed`), viết `tests/test_models_contract.py`.<br>- **TV4:** Hoàn thiện Bài 6 (`06_hand_segmentation_segnet.ipynb`), xuất mask dự đoán & logs. | Toàn bộ 7 notebook (`00` → `06`) chạy hoàn chỉnh từ đầu đến cuối, đạt hoặc vượt mốc Keras gốc. |
| **GĐ 3: Đối chiếu & Viết Báo cáo** | Ngày 6 – 7 | - **TV3:** Chạy `pytest/unittest` kiểm tra toàn bộ 6 model, tổng hợp `logs/comparison_keras_vs_pytorch.csv`.<br>- **TV1, TV2, TV3, TV4:** Hoàn thiện các chương báo cáo được phân công (Chương 1 → 6).<br>- **TV4:** Cập nhật `README.md` bản PyTorch 2.x, ghép hoàn chỉnh `reports/BaoCao_KerasToPyTorch.docx` & `.pdf`. | Báo cáo đầy đủ 6 chương, toàn bộ số liệu khớp 100% với `logs/*.csv` và output notebook. |
| **GĐ 4: Rà soát & Đóng gói** | Ngày 8 | - **Cả nhóm:** Kiểm tra chéo (Peer Review) từng notebook đảm bảo không còn sót dòng `import keras/tensorflow`, ảnh biểu đồ rõ nét, output cell đã lưu sẵn.<br>- **TV4 (Nhóm trưởng):** Dọn dẹp cache rác, không nén tập dữ liệu thô quá nặng, kiểm tra đủ **Báo cáo + Code đã chuyển + File liên quan** và đóng gói file nộp bài. | Bộ sản phẩm hoàn chỉnh sẵn sàng nộp đúng hạn. |

---

## VII. DANH MỤC KIỂM TRA TRƯỚC KHI NỘP (PRE-SUBMISSION CHECKLIST)

- [ ] **Hoàn toàn thuần PyTorch:** Không còn lệnh `import keras` hay `import tensorflow` trong toàn bộ code đã chuyển (`src/` và `notebooks/`).
- [ ] **Đủ 7 Notebook đã chạy có lưu Output rõ ràng:** `00` (Basics), `01` (MLP MNIST), `02` (CNN MNIST), `03` (Gender CNN – 2 bản), `04` (FER CNN – 2 bản), `05` (CNN+LSTM Summation), `06` (SegNet Segmentation).
- [ ] **Minh chứng đầy đủ 3 kỹ thuật huấn luyện từ Slide:** `EarlyStopping`, `ReduceLROnPlateau` (Bài 3) và `Data Augmentation` (Bài 4) đều có đối sánh Before/After.
- [ ] **Báo cáo hoàn chỉnh (`.docx` & `.pdf`):** Có bảng ánh xạ cú pháp Keras ↔ PyTorch, code đối chiếu, kết quả thực nghiệm khớp với `logs/comparison_keras_vs_pytorch.csv` và bảng phân công 4 thành viên.
- [ ] **File liên quan đầy đủ:** `README.md` (đã cập nhật hướng dẫn PyTorch 2.x/Colab thay cho `README (1).md` cũ), `requirements.txt`, `src/`, `tests/`, `logs/`, `figures/`.
