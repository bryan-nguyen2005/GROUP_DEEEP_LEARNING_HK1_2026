# CHUYỂN ĐỔI MÃ NGUỒN HỌC SÂU TỪ KERAS 2.x/TF 1.x SANG PYTORCH 2.x

- **Môn học:** Học sâu (Deep Learning) · **Lớp:** DCT123C4 · **GVHD:** ThS. Đỗ Như Tài
- **Mục tiêu:** Chuyển đổi toàn bộ bộ ví dụ thực hành từ tài liệu gốc (`01_Introduction-to-Python-TensorFlow-and-Keras.pdf` và `02_KerasExamples.pdf` — Python 3.5 / TensorFlow 1.1 / Keras 2.0.9 / CUDA 8.0) sang **Python 3.12 + PyTorch 2.x + CUDA 12.x** (tương thích hoàn toàn Google Colab GPU và máy cục bộ).

---

## 1. Bảng phân công nhiệm vụ thành viên (4 × 25%)

| STT | Họ và tên | MSSV | Nhiệm vụ kỹ thuật & Code PyTorch | Nhiệm vụ Báo cáo & Tài liệu | Tỷ lệ |
| :-: | :--- | :---: | :--- | :--- | :---: |
| 1 | **Nguyễn Đình Cường** *(Nhóm trưởng)* | `3123411043` | **TV1:** Core `pytorch_core.py`, `00_keras_vs_pytorch_basics.ipynb`, Bài 1 (`mnist_mlp.py`), Bài 2 (`mnist_cnn.py`) | Chương 1 (Bảng ánh xạ API) & Chương 2 (Bài 1, 2) | 25% |
| 2 | **Châu Quốc Bảo** | `3123411024` | **TV2:** Bài 3 (`gender_cnn.py`: Baseline vs EarlyStopping+ReduceLR) & Bài 4 (`emotion_cnn.py`: No-Aug vs Augmentation) | Chương 3 (Bài 3, 4 & 3 kỹ thuật huấn luyện) | 25% |
| 3 | **Nguyễn Hoàng Long** | `3123411179` | **TV3:** Bài 5 (`mnist_lstm.py` + `TimeDistributed`), `tests/test_models_contract.py`, `logs/comparison_keras_vs_pytorch.csv` | Chương 4 (Bài 5 CNN-LSTM) & Mục 6.1 (Khác biệt Keras vs PyTorch) | 25% |
| 4 | **Võ Huỳnh Minh Sang** | `3123411256` | **TV4:** Bài 6 (`segnet.py`, `segnet_experiment.py`, `06_hand_segmentation_segnet.ipynb`), `visualize.py`, `README.md`, `requirements.txt` | Chương 5 (Bài 6 SegNet), Chương 6 (Tổng kết), biên tập Báo cáo & đóng gói | 25% |

---

## 2. Cài đặt môi trường PyTorch 2.x (Local & Google Colab)

### Trên máy cục bộ (Linux / Windows / macOS)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Cài PyTorch 2.x hỗ trợ CUDA 12.x (hoặc thay cu124 bằng cpu nếu máy không có GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Trên Google Colab (GPU T4 miễn phí)
Chọn `Runtime → Change runtime type → T4 GPU`, clone/upload thư mục `KerasToPytorch/`, cài thêm `torchinfo` (`!pip install -q torchinfo`) và chạy trực tiếp từng notebook trong `notebooks/`.

---

## 3. Cấu trúc dự án & Cách chạy tái lập từng bài

```text
KerasToPytorch/
├── README.md                                  # Hướng dẫn môi trường PyTorch 2.x (thay thế README (1).md cũ)
├── requirements.txt                           # Danh sách thư viện PyTorch 2.x & phụ thuộc
├── plan.md                                    # Kế hoạch & phân công 4 thành viên
├── data/README_DATA.md                        # Hướng dẫn dữ liệu cho cả 6 bài
├── src/
│   ├── pytorch_core.py                        # [TV1] seed_everything, EarlyStopping, fit(), evaluate()
│   ├── visualize.py                           # [TV4] Bộ hàm vẽ 300 DPI dùng chung
│   ├── mnist_experiment.py                    # [TV1] Chạy Bài 1 & Bài 2
│   ├── face_experiment.py                     # [TV2] Chạy Bài 3 & Bài 4
│   ├── lstm_experiment.py                     # [TV3] Chạy Bài 5
│   ├── segnet_experiment.py                   # [TV4] Chạy Bài 6
│   └── models/                                # 6 kiến trúc PyTorch thuần (không dùng Keras/TF)
│       ├── mnist_mlp.py, mnist_cnn.py, gender_cnn.py, emotion_cnn.py, mnist_lstm.py, segnet.py
├── notebooks/                                 # 7 notebook đã chạy sẵn output (00 → 06)
├── tests/test_models_contract.py              # Kiểm thử shape & backprop cho cả 6 mô hình
├── logs/comparison_keras_vs_pytorch.csv       # Bảng đối chiếu tổng hợp 6 bài (Keras vs PyTorch)
├── figures/                                   # Toàn bộ biểu đồ 300 DPI
└── reports/
    ├── BaoCao_KerasToPyTorch.docx             # Báo cáo tổng hợp hoàn chỉnh (bìa SGU + mục lục tự động)
    └── BaoCao_KerasToPyTorch.pdf
```

### Lệnh chạy nhanh
```bash
# 1. Chạy bộ kiểm thử hợp đồng cho toàn bộ 6 mô hình
python -m unittest discover -s tests -v

# 2. Chạy từng bài thực hành qua CLI (hoặc mở trực tiếp notebook tương ứng trong notebooks/)
python -m src.mnist_experiment                 # Bài 1 (MLP) & Bài 2 (CNN #1 MNIST)
python -m src.face_experiment gender --data data/UTKFace --epochs 30
python -m src.face_experiment emotion --data ../BTLON02_Emotion_HoTen_MaSV.zip/data/fer2013.csv --epochs 30
python -m src.lstm_experiment                  # Bài 5 (CNN + TimeDistributed + LSTM)
python -m src.segnet_experiment --epochs 18    # Bài 6 (SegNet Hand Depth 96x96)
```
