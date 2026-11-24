# BÀI TẬP LỚN 02 – HỌC SÂU (DEEP LEARNING)
## Nhận diện cảm xúc khuôn mặt bằng mạng CNN trên bộ dữ liệu FER2013

- **Lớp:** DCT123C4
- **Giảng viên hướng dẫn:** ThS. Đỗ Như Tài
- **Công nghệ:** Python 3.12 · PyTorch · TorchVision · scikit-learn · Matplotlib
- **Bộ dữ liệu:** Kaggle FER2013 (35.887 ảnh xám `48 × 48`, 7 lớp cảm xúc: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`)

---

## 1. Bảng phân công nhiệm vụ thành viên (4 × 25%)

| STT | Họ và tên | MSSV | Vai trò & Phụ trách chính | Tỷ lệ |
| :-: | :--- | :---: | :--- | :---: |
| 1 | **Nguyễn Đình Cường** *(Nhóm trưởng)* | `3123411043` | **TV1:** Data Pipeline (`dataset.py`, `transforms.py`), EDA (`eda.py`), kho 6 kiến trúc CNN (`models.py`), Chương 1–2–3.1 Báo cáo & Slide 1–4. | 25% |
| 2 | **Châu Quốc Bảo** | `3123411024` | **TV2:** Core Training Pipeline (`utils.py`, `metrics.py`, `trainer.py`), Thực nghiệm Nhóm 1 (`run_exp1_architectures.py`, `EXP-01..EXP-06` + Fail Case VGG16), Chương 3.2–4.1 Báo cáo & Slide 5–7. | 25% |
| 3 | **Nguyễn Hoàng Long** | `3123411179` | **TV3:** Thực nghiệm Nhóm 2 Ablation Study (`run_exp2_training_techniques.py`, `EXP-07..EXP-17`), chủ trì Nhật ký thực nghiệm (`NHAT_KY_THUC_NGHIEM.md`), Chương 4.2–4.3 Báo cáo & Slide 8–10. | 25% |
| 4 | **Võ Huỳnh Minh Sang** | `3123411256` | **TV4:** Thư viện vẽ 300 DPI (`visualize.py`), Trực quan hóa Final Conv Filters / Feature Maps / Grad-CAM (`visualize_filters.py`), Đánh giá Best Model & PrivateTest (`evaluate_best_model.py`, `EXP-18`), `main.ipynb`, `README.md`, `requirements.txt`, Chương 5–6, tổng hợp Báo cáo & Slide cuối cùng, đóng gói nộp bài. | 25% |

---

## 2. Cấu trúc thư mục dự án

```text
├── README.md
├── requirements.txt
├── plan.md
├── main.ipynb                                 # Notebook tổng hợp end-to-end (Colab GPU & Local)
├── data/
│   └── README_DATA.md                         # Hướng dẫn tải fer2013.csv (SHA-256 chuẩn)
├── src/
│   ├── utils.py                               # [TV2] Seed, device, param counter, timer
│   ├── dataset.py                             # [TV1] FER2013Dataset, DataLoader, class weights
│   ├── transforms.py                          # [TV1] Augmentation (none/standard/strong) & 3-view TTA
│   ├── eda.py                                 # [TV1] Thống kê EDA & biểu đồ dữ liệu
│   ├── models.py                              # [TV1] 6 kiến trúc CNN + get_first/final_conv_layer()
│   ├── metrics.py                             # [TV2] Accuracy, Macro-F1, Per-class, Confusion Matrix
│   ├── trainer.py                             # [TV2] Vòng lặp PyTorch, EarlyStopping, Checkpoint, CSVLogger
│   ├── visualize.py                           # [TV4] Bộ hàm vẽ 300 DPI dùng chung
│   ├── run_exp1_architectures.py              # [TV2] Thực nghiệm Nhóm 1 (EXP-01..EXP-06)
│   ├── run_exp2_training_techniques.py        # [TV3] Thực nghiệm Nhóm 2 (EXP-07..EXP-17)
│   ├── visualize_filters.py                   # [TV4] Trực quan hóa First/Final Conv Filters, Feature Maps & Grad-CAM
│   └── evaluate_best_model.py                 # [TV4] Thẩm định PublicTest (Standard vs TTA) & Đánh giá PrivateTest
├── checkpoints/
│   └── best_model.pth                         # Trọng số mô hình tối ưu cuối cùng
├── logs/
│   ├── NHAT_KY_THUC_NGHIEM.md                 # Nhật ký thực nghiệm tổng hợp (EXP-01..EXP-18)
│   ├── experiment_runs.csv                    # Bảng log tự chứa 18 thực nghiệm
│   ├── best_model_private_metrics.csv         # Kết quả PublicTest & PrivateTest chính thức
│   ├── best_model_per_class_metrics.csv       # Precision/Recall/F1 từng lớp trên PrivateTest
│   └── history_EXP-*.csv                      # Lịch sử huấn luyện theo từng epoch
├── figures/                                   # 18 biểu đồ chất lượng 300 DPI
├── reports/
│   ├── BTLON02_BaoCao_FER2013.docx
│   └── BTLON02_BaoCao_FER2013.pdf
├── slides/
│   ├── BTLON02_Slide_FER2013.pptx
│   └── BTLON02_Slide_FER2013.pdf
└── tests/                                     # Unit tests cho toàn bộ pipeline
```

---

## 3. Cài đặt môi trường và chuẩn bị dữ liệu

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải tập dữ liệu `fer2013.csv` theo hướng dẫn tại `data/README_DATA.md` (SHA-256: `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`) và đặt tại `data/fer2013.csv`.

---

## 4. Hướng dẫn chạy tái lập từng giai đoạn

```bash
# 1. Chạy toàn bộ unit tests
python -m unittest discover -s tests -v

# 2. Khám phá dữ liệu EDA và kiểm tra số tham số 6 mô hình (TV1)
python -m src.eda data/fer2013.csv
python -m src.models

# 3. Thực nghiệm Nhóm 1: So sánh 6 kiến trúc CNN + Fail Case VGG16 (TV2)
python -m src.run_exp1_architectures --data-path data/fer2013.csv --max-epochs 60 --patience 12 --batch-size 64 --seed 42

# 4. Thực nghiệm Nhóm 2: Ablation Study kỹ thuật huấn luyện (TV3)
python -m src.run_exp2_training_techniques --data-path data/fer2013.csv --max-epochs 60 --patience 12 --batch-size 64 --seed 42

# 5. Đánh giá Best Model trên PublicTest (Standard vs TTA) và PrivateTest (TV4)
python -m src.evaluate_best_model --data-path data/fer2013.csv --checkpoint checkpoints/EXP-05_ImprovedFERNet.pth
# (Tùy chọn huấn luyện lại 3 seeds trên GPU: thêm --train-seeds 42 100 2024)

# 6. Trực quan hóa First/Final Conv Filters, Feature Maps và Grad-CAM (TV4)
python -m src.visualize_filters --checkpoint checkpoints/best_model.pth --data-path data/fer2013.csv --split PrivateTest
```

Hoặc mở trực tiếp `main.ipynb` để chạy toàn bộ quy trình trên Google Colab GPU hoặc máy cục bộ.
