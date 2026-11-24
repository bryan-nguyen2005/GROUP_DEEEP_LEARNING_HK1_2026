# Nhật ký thực nghiệm Giai đoạn 3 (EXP-18 & Trực quan hóa) – Thành viên 4 (TV4)

**Thời gian thực hiện:** 24/11/2026 · **Phụ trách:** Võ Huỳnh Minh Sang (`3123411256`)  
**Phạm vi:** Xây dựng thư viện vẽ chuẩn 300 DPI (`src/visualize.py`), trực quan hóa bộ lọc tầng tích chập đầu/cuối, Feature Maps và Grad-CAM (`src/visualize_filters.py`), thực thi `EXP-18` (`src/evaluate_best_model.py`) thẩm định trên `PublicTest` (Standard vs 3-view TTA) và mở khóa đánh giá một lần duy nhất trên `PrivateTest` (3.589 ảnh), đồng thời hoàn thiện `main.ipynb`, `README.md`, `requirements.txt`, Báo cáo tổng hợp, Slide tổng hợp và đóng gói nộp bài.

---

## 1. Môi trường nghiệm thu & Kiểm chứng Checkpoint

- **Phần cứng nghiệm thu:** 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 luồng CPU, không có GPU CUDA cục bộ).
- **Phần mềm:** Python `3.12`, PyTorch `2.15.0+cpu`, TorchVision `0.29.2+cpu`, NumPy `2.5.3`, scikit-learn, Matplotlib `3.11.2`.
- **Dữ liệu kiểm chứng:** `data/fer2013.csv` trùng khớp tuyệt đối SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`.
- **Kiểm chứng trọng số bàn giao:** Đánh giá độc lập trên `PublicTest` xác nhận `checkpoints/EXP-05_ImprovedFERNet.pth` lưu giữ trọng số đã huấn luyện thật (`val_acc = 68.63%`, `val_loss = 0.8867`), trong khi các tệp `.pth` trung gian của Nhóm 2 chưa đồng bộ đúng trọng số sau huấn luyện. Theo đúng cơ chế **Fallback** trong `plan.md`, TV4 chốt `EXP-05_ImprovedFERNet.pth` làm checkpoint đại diện (`checkpoints/best_model.pth`), đồng thời cài đặt sẵn cờ `--train-seeds 42 100 2024` trong `src/evaluate_best_model.py` để tái huấn luyện đa seed trên GPU.

---

## 2. Kết quả `EXP-18` (PublicTest Standard vs 3-View TTA → PrivateTest Final)

Nguồn truy vết 100%: [best_model_private_metrics.csv](best_model_private_metrics.csv) và [best_model_per_class_metrics.csv](best_model_per_class_metrics.csv).

| Tập đánh giá & Chế độ | Mô hình | Tham số | Loss | **Accuracy** | **Macro-F1** | Weighted-F1 | Macro-Precision | Macro-Recall |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `PublicTest` (Standard) | `ImprovedFERNet` | 4.863.751 | 0.8867 | **68.63%** | 66.36% | 68.49% | 68.10% | 65.21% |
| `PublicTest` (3-View TTA) | `ImprovedFERNet` | 4.863.751 | 0.8672 | 68.51% | **66.41%** | 68.36% | 68.08% | 65.25% |
| **`PrivateTest` (FINAL — Standard)** | **`ImprovedFERNet`** | **4.863.751** | **0.8364** | **70.41%** | **68.47%** | **70.29%** | **70.36%** | **67.18%** |

- **Quyết định TTA trên `PublicTest`:** Dù 3-View TTA giảm `val_loss` từ `0.8867` xuống `0.8672` và nhích nhẹ Macro-F1 (+0.05%), Accuracy giảm nhẹ `-0.12%` do phép crop tâm `44×44` có thể cắt sát viền khuôn mặt ở ảnh cận cảnh. Hệ thống tự động khóa suy luận **Standard** trước khi mở khóa `PrivateTest`.
- **Hiệu năng từng lớp trên `PrivateTest`:** `Happy` dẫn đầu (`F1 = 89.39%`, Recall `90.56%`), kế đến `Surprise` (`79.70%`), `Neutral` (`69.81%`), `Disgust` (`66.67%`, Precision `78.05%`), `Angry` (`64.44%`), `Sad` (`57.92%`) và `Fear` (`51.37%`).

---

## 3. Trực quan hóa Final Convolution Filters, Feature Maps & Grad-CAM

- `figures/viz_first_conv_filters.png`: 64 bộ lọc `3×3` tầng đầu phát hiện biên ngang, biên dọc, biên chéo và tương phản sáng–tối.
- `figures/viz_final_conv_filters_weights.png`: 36 bộ lọc mạnh nhất cùng phân bố chuẩn L2 của toàn bộ 512 bộ lọc tầng cuối (`Mean = 7.188 ± 0.380`, không có bộ lọc chết).
- `figures/viz_final_conv_feature_maps.png`: Top-6 bản đồ đặc trưng kích hoạt tại `final_conv` cho 7 cảm xúc trên `PrivateTest`.
- `figures/viz_gradcam_7emotions.png`: Bản đồ nhiệt Grad-CAM trên 7 ca đúng và 7 ca sai, chứng minh mô hình bám đúng vùng mắt–mũi–miệng (không bị nhiễu bởi watermark) và làm rõ yếu tố nhiễu nhãn trong FER2013.

---
*Xác nhận hoàn thành nhiệm vụ TV4 · Võ Huỳnh Minh Sang (3123411256) · Ngày 24/11/2026*
