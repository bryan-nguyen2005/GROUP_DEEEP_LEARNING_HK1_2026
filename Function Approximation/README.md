# Bài Tập Lớn Học Sâu: Xấp Xỉ Hàm Mật Độ Gaussian Hai Chiều Bằng Mạng MLP

**Môn học:** Học sâu (Deep Learning)  
**Giảng viên hướng dẫn:** Đỗ Như Tài  
**Đề tài:** Triển khai mạng Perceptron đa tầng (MLP) bằng PyTorch để xấp xỉ hàm mật độ xác suất Gaussian 2 chiều và khảo sát ảnh hưởng của quy mô dữ liệu huấn luyện (Thực nghiệm A), độ rộng tầng ẩn (Thực nghiệm B) và độ sâu mạng (Thực nghiệm C).

---

## 1. Thông tin nhóm thực hiện

| STT | Họ và tên | MSSV | Nhiệm vụ chính | Đóng góp |
|:---:|:---|:---:|:---|:---:|
| 1 | Châu Quốc Bảo | 3123411024 | Sinh dữ liệu Gaussian 2D, thiết kế kiến trúc mạng `GaussianMLP`, vẽ biểu đồ 3D; viết Mục 1–2 báo cáo | 25% |
| 2 | Nguyễn Hoàng Long | 3123411179 | Xây dựng pipeline huấn luyện, Early Stopping, các chỉ số đánh giá và Thực nghiệm A; viết Mục 3 & 4.1 | 25% |
| 3 | Võ Huỳnh Minh Sang | 3123411256 | Thực hiện Thực nghiệm B (độ rộng tầng ẩn) và Thực nghiệm C (độ sâu mạng & thời gian); viết Mục 4.2 & 4.3 | 25% |
| 4 | Nguyễn Đình Cường (Nhóm trưởng) | 3123411043 | Thẩm định và đánh giá Best Model, trực quan hóa kết quả, tổng hợp báo cáo Word và slide thuyết trình | 25% |

---

## 2. Cấu trúc thư mục

```text
.
├── data/                               # Bộ dữ liệu CSV chuẩn
│   ├── train_val_pool.csv              # Pool 5.000 mẫu (seed=2024): 400 mẫu Validation [0:400] + 4.600 mẫu Train [400:5000]
│   └── test_dataset_100.csv            # Tập Test độc lập 100 mẫu (seed=42)
├── src/                                # Mã nguồn chính của dự án
│   ├── __init__.py
│   ├── data_generator.py               # Hàm mật độ Gaussian 2D, sinh/lưu dữ liệu CSV và chia tập Train/Validation
│   ├── models.py                       # Định nghĩa lớp mạng GaussianMLP (tùy chỉnh số tầng ẩn C và số nơ-ron B)
│   ├── metrics.py                      # Hàm tính các chỉ số hồi quy: MSE, MAE, RMSE, R2
│   ├── trainer.py                      # DataLoader, vòng lặp huấn luyện, EarlyStopping và lưu checkpoint
│   ├── visualize.py                    # Các hàm vẽ biểu đồ 3D, đường cong hội tụ, xu hướng sai số và Parity/Residual
│   ├── run_experiment_A.py             # Thực nghiệm A: Khảo sát số lượng mẫu huấn luyện A (200 -> 2000)
│   ├── run_experiment_B.py             # Thực nghiệm B: Khảo sát số nơ-ron tầng ẩn B (1 -> 64)
│   ├── run_experiment_C.py             # Thực nghiệm C: Khảo sát số tầng ẩn C (1 -> 5, 32 nơ-ron/tầng) và thời gian huấn luyện
│   └── evaluate_best_model.py          # Thẩm định ứng viên Best Model qua 3 seed và đánh giá trên tập Test 100 mẫu
├── logs/                               # Các file CSV lưu kết quả thực nghiệm và lịch sử huấn luyện theo epoch
├── checkpoints/                        # Các file trọng số mô hình (.pth), bao gồm checkpoints/best_model.pth
├── figures/                            # Hình ảnh biểu đồ xuất ra từ thực nghiệm (300 DPI)
├── reports/
│   └── BTL01_BaoCao_MLP.docx           # Báo cáo tổng hợp Bài tập lớn
├── slides/
│   └── BTL01_ThuyetTrinh_MLP.pptx      # Slide thuyết trình của nhóm
├── tests/
│   └── test_training_contract.py       # Unit test kiểm tra lưu checkpoint và xuất chỉ số thực nghiệm
├── main.ipynb                          # Jupyter Notebook chạy tổng hợp toàn bộ quy trình
├── requirements.txt                    # Danh sách thư viện Python cần thiết
└── README.md                           # Tài liệu hướng dẫn chạy chương trình
```

---

## 3. Hướng dẫn cài đặt và chạy chương trình

### 3.1. Cài đặt môi trường

Yêu cầu Python 3.10+ và các thư viện trong `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2. Chạy từng bước từ dòng lệnh

Các lệnh dưới đây được thực thi tại thư mục gốc của dự án:

1. **Sinh bộ dữ liệu chuẩn (`data/train_val_pool.csv` và `data/test_dataset_100.csv`):**
   ```bash
   python src/data_generator.py
   ```

2. **Chạy 3 nhóm thực nghiệm A, B, C (mỗi cấu hình chạy trên 3 random seed `42, 100, 2024`):**
   ```bash
   python src/run_experiment_A.py
   python src/run_experiment_B.py
   python src/run_experiment_C.py
   ```

3. **Thẩm định và đánh giá mô hình tốt nhất (`Best Model`) trên tập Test 100 mẫu:**
   ```bash
   python src/evaluate_best_model.py
   ```

4. **Chạy thông qua Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```

5. **Chạy kiểm thử tự động (Unit Tests):**
   ```bash
   python -m unittest discover -s tests -v
   ```

---

## 4. Tóm tắt kết quả chính

- **Hàm mục tiêu:** Hàm mật độ Gaussian 2 chiều trên miền $[-2, 2] \times [-4, 4]$ với $\mu = [0, 0]^T$ và ma trận hiệp phương sai $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$.
- **Thiết lập huấn luyện chuẩn:** Hàm kích hoạt ReLU ở tầng ẩn, đầu ra tuyến tính, tối ưu bằng Adam (`lr = 0.005`, `batch_size = 32`, tối đa 150 epochs, Early Stopping `patience = 15` trên tập Validation 400 mẫu).
- **Mô hình tốt nhất (`Best Model`):** Cấu hình 2 tầng ẩn $\times$ 64 nơ-ron/tầng, huấn luyện trên $A = 2.000$ mẫu (`checkpoints/best_model.pth`, seed đại diện `2024`):
  - **Validation MSE (trung bình $\pm$ độ lệch chuẩn qua 3 seed):** `2.255170e-05 ± 2.051022e-05`
  - **Kết quả trên tập Test độc lập 100 mẫu:**
    - `MSE = 1.770885e-05`
    - `MAE = 3.031825e-03`
    - `RMSE = 4.208188e-03`
    - `R2 = 0.99427`
