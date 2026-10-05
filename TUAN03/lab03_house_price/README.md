# Lab 03 — House Price Prediction

Bài làm nhóm về dự đoán giá nhà từ bộ dữ liệu Ames Housing của cuộc thi **House Prices: Advanced Regression Techniques**. Nhóm thực hiện phân tích dữ liệu, xây dựng pipeline tiền xử lý, huấn luyện mô hình scikit-learn và MLP bằng PyTorch, đánh giá feature engineering và nộp dự đoán lên Kaggle.

Toàn bộ dữ liệu, notebook, mã nguồn, mô hình, kết quả và báo cáo nằm trong thư mục [`lab03_house_price/`](lab03_house_price/).

## 1. Hướng dẫn xem bài nhanh

1. **Báo cáo:** [lab03_house_price_NguyenDinhCuong_3123411043.docx](lab03_house_price/report/lab03_house_price_NguyenDinhCuong_3123411043.docx).
2. **Quy trình thực hiện:** mở các notebook trong [`notebooks/`](lab03_house_price/notebooks/) theo thứ tự `01 → 02 → 03 → 04`. Các notebook có kết quả đã lưu để đối chiếu với báo cáo.
3. **Kết quả tổng hợp:** [experiment_results.csv](lab03_house_price/outputs/results/experiment_results.csv); bảng và nhận xét feature engineering tại [experiment_report.md](lab03_house_price/outputs/results/experiment_report.md).
4. **Minh chứng Kaggle:** [kaggle_submission_best.png](lab03_house_price/outputs/figures/kaggle_submission_best.png).
5. **File đã nộp Kaggle:** [submission_best.csv](lab03_house_price/outputs/submissions/submission_best.csv).

## 2. Thành viên và phân công

**Lớp:** DCT123C4.

| Thành viên | MSSV | Phần thực hiện |
|---|---|---|
| Nguyễn Đình Cường — Nhóm trưởng | 3123411043 | Các mô hình scikit-learn, cross-validation, tuning và tích hợp bài làm |
| Châu Quốc Bảo | 3123411024 | MLP PyTorch, Dataset/DataLoader, huấn luyện, validation và checkpoint |
| Nguyễn Hoàng Long | 3123411179 | Feature engineering, ablation E0–E5, tổng hợp kết quả và submission Kaggle |
| Võ Huỳnh Minh Sang | 3123411256 | Phân tích dữ liệu, EDA và pipeline tiền xử lý dùng chung |

## 3. Cấu trúc và ý nghĩa các thư mục

```text
TUAN03/
├── README.md                         # Hướng dẫn xem và chạy bài làm
└── lab03_house_price/
    ├── data/                         # Dữ liệu đầu vào của Kaggle
    ├── notebooks/                    # 4 notebook trình bày quy trình thực nghiệm
    ├── src/                          # Mã nguồn xử lý dữ liệu, huấn luyện, dự đoán
    ├── models/                       # Mô hình đã huấn luyện và preprocessor
    │   └── ablation/                 # Checkpoint MLP của từng bộ feature E0–E5
    ├── outputs/
    │   ├── results/                  # Bảng số liệu, cấu hình và lịch sử huấn luyện
    │   ├── figures/                  # Biểu đồ và ảnh minh chứng Kaggle
    │   └── submissions/              # Các CSV dự đoán giá nhà cho tập test
    ├── report/                       # Báo cáo Word của nhóm
    └── _smoke_test.py                # Kiểm tra nhanh việc tạo engineered features
```

| Thư mục | Nội dung và cách sử dụng |
|---|---|
| [`data/`](lab03_house_price/data/) | `train.csv`: 1.460 mẫu có nhãn `SalePrice`; `test.csv`: 1.459 mẫu cần dự đoán; `sample_submission.csv`: mẫu định dạng nộp; `data_description.txt`: mô tả các thuộc tính. Dữ liệu đã được kèm trong bài làm. |
| [`notebooks/`](lab03_house_price/notebooks/) | Trình bày từng bước thực hiện, mã chạy, bảng kết quả, biểu đồ và nhận xét. Đây là nơi phù hợp để đọc quy trình và chạy lại bài. |
| [`src/`](lab03_house_price/src/) | Các module dùng chung cho notebook và các lệnh chạy từ terminal. Chi tiết từng module ở mục 6. |
| [`models/`](lab03_house_price/models/) | Pipeline sklearn `.joblib`, checkpoint PyTorch `.pt` và các preprocessor `.joblib`. Checkpoint holdout dùng để đánh giá; checkpoint refit dùng để dự đoán tập test. |
| [`models/ablation/`](lab03_house_price/models/ablation/) | Checkpoint và preprocessor MLP tương ứng với E0–E5 để đối chiếu từng thực nghiệm feature engineering. |
| [`outputs/results/`](lab03_house_price/outputs/results/) | Kết quả sklearn, MLP, đánh giá từng feature, ablation; JSON cấu hình và CSV lịch sử huấn luyện giúp truy vết số liệu trong báo cáo. |
| [`outputs/figures/`](lab03_house_price/outputs/figures/) | Các ảnh EDA, so sánh mô hình, learning curve MLP, ablation và ảnh điểm Kaggle. |
| [`outputs/submissions/`](lab03_house_price/outputs/submissions/) | Ba file `submission_sklearn.csv`, `submission_mlp.csv`, `submission_best.csv`. Mỗi file có hai cột `Id,SalePrice` và 1.459 dòng dự đoán. |
| [`report/`](lab03_house_price/report/) | Báo cáo Word trình bày bài toán, phương pháp, thực nghiệm, kết quả và kết luận. |

## 4. Chuẩn bị môi trường

Cần Python, pip và các thư viện bên dưới. Bài làm có thể chạy bằng **CPU**, không bắt buộc GPU hay tài khoản Kaggle để chạy lại thực nghiệm nội bộ.

### Windows — Command Prompt (CMD)

Mở terminal tại thư mục gốc repository rồi chạy:

```bat
cd TUAN03\lab03_house_price
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install "pandas>=3,<4" numpy scipy matplotlib seaborn scikit-learn torch joblib jupyterlab ipykernel
```

Nếu terminal đã mở tại thư mục `TUAN03`, chỉ cần `cd lab03_house_price` ở bước đầu. Nếu dùng PowerShell, lệnh kích hoạt môi trường là `.\.venv\Scripts\Activate.ps1`; cũng có thể dùng trực tiếp `.\.venv\Scripts\python.exe` thay cho `python` mà không cần kích hoạt.

### Linux hoặc macOS

```bash
cd TUAN03/lab03_house_price
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pandas>=3,<4" numpy scipy matplotlib seaborn scikit-learn torch joblib jupyterlab ipykernel
```

Các lệnh ở mục 5–6 đều chạy khi thư mục làm việc là **`lab03_house_price/`** và môi trường trên đã được kích hoạt.

Thông tin phiên bản của lần chạy MLP đã lưu tại [mlp_environment.txt](lab03_house_price/outputs/results/mlp_environment.txt) và [mlp_run_summary.json](lab03_house_price/outputs/results/mlp_run_summary.json). Khi sử dụng lại mô hình đã lưu, nên dùng các phiên bản tương ứng; nếu môi trường khác gây lỗi tải mô hình, có thể huấn luyện lại trong môi trường mới.

## 5. Chạy bằng notebook

Khởi động JupyterLab từ thư mục `lab03_house_price/`:

```bash
python -m jupyterlab
```

Mở thư mục `notebooks/`, chọn kernel của môi trường vừa cài và chạy từng notebook bằng **Restart Kernel and Run All Cells** theo thứ tự:

| Thứ tự | Notebook | Nội dung chính |
|---|---|---|
| 01 | [01_eda_preprocessing.ipynb](lab03_house_price/notebooks/01_eda_preprocessing.ipynb) | Kiểm tra dữ liệu, missing values, phân phối giá, tương quan, outlier và tiền xử lý. |
| 02 | [02_sklearn_models.ipynb](lab03_house_price/notebooks/02_sklearn_models.ipynb) | So sánh các mô hình hồi quy bằng 5-fold CV, tuning và lưu mô hình sklearn tốt nhất trên feature gốc. |
| 03 | [03_pytorch_mlp.ipynb](lab03_house_price/notebooks/03_pytorch_mlp.ipynb) | So sánh 8 cấu hình MLP, early stopping, learning curve, chọn checkpoint và refit để dự đoán test. |
| 04 | [04_feature_engineering.ipynb](lab03_house_price/notebooks/04_feature_engineering.ipynb) | Tạo 11 feature, đánh giá từng feature, ablation E0–E5, tạo submission và bảng tổng hợp. |

Cũng có thể mở notebook bằng VS Code và chọn Python interpreter của `.venv`. Notebook cần chạy với thư mục làm việc là `lab03_house_price/` hoặc `lab03_house_price/notebooks/` để nhận đúng đường dẫn dữ liệu.

## 6. Chạy bằng terminal và vai trò từng module

Notebook 01 là bước đọc EDA; các bước huấn luyện và dự đoán còn lại có thể chạy bằng terminal. Không cần chạy cả notebook và script cho cùng một bước nếu chỉ muốn thực hiện lại bước đó một lần.

| Module | Vai trò |
|---|---|
| `src/preprocessing.py` | Đọc dữ liệu, tách X/y, bỏ `Id`, biến đổi target bằng `log1p`, điền missing, one-hot encoding và scaling. |
| `src/train_sklearn.py` | Chạy các baseline, 5-fold CV, GridSearchCV và lưu pipeline sklearn tốt nhất trên feature gốc. |
| `src/mlp_model.py` | Định nghĩa kiến trúc MLP và các cấu hình mạng. |
| `src/train_mlp.py` | Chia holdout, tạo DataLoader, huấn luyện MLP, early stopping, lưu checkpoint, biểu đồ và refit mô hình dự đoán test. |
| `src/features.py` | Tạo 11 engineered features và định nghĩa các bộ feature E0–E5. |
| `src/run_ablation.py` | So sánh E0–E5, đánh giá từng feature, xuất bảng và biểu đồ feature engineering. |
| `src/predict.py` | Tạo submission sklearn từ pipeline đã lưu hoặc refit mô hình/bộ feature tốt nhất trong ablation. |
| `src/predict_mlp.py` | Nạp checkpoint MLP cùng preprocessor và tạo submission MLP. |

### Chạy lại toàn bộ phần mô hình

Thực hiện lần lượt:

```bash
python -m src.train_sklearn
python -m src.train_mlp
python -m src.run_ablation
python -m src.predict --only all
```

- Bước sklearn lưu `models/best_sklearn_model.joblib` và `outputs/results/sklearn_results.csv`.
- Bước MLP mặc định chạy 8 cấu hình trên CPU, tối đa 400 epoch, patience 40; lưu `models/best_mlp.pt`, mô hình refit `models/mlp_submission.pt` và tạo `submission_mlp.csv`.
- Bước ablation lưu kết quả E0–E5, bảng từng feature, hình so sánh và `experiment_results.csv`.
- Bước prediction tạo `submission_sklearn.csv` và `submission_best.csv`. File best được chọn từ kết quả ablation rồi huấn luyện lại trên toàn bộ `train.csv`.

GridSearchCV thử nhiều cấu hình qua 5 fold nên thời gian chạy phụ thuộc cấu hình máy. Chạy lại các bước trên sẽ cập nhật các file kết quả và mô hình. Điểm Kaggle không được tính tự động khi chạy trên máy; bảng tổng hợp được sinh lại có thể để trống cột điểm Kaggle. Nếu nộp một submission mới, cần ghi điểm của đúng lần nộp đó.

### Tạo lại submission từ các kết quả đã có

Không cần chạy lại toàn bộ thực nghiệm nếu chỉ muốn tạo dự đoán:

```bash
python -m src.predict --only sklearn
python -m src.predict_mlp
python -m src.predict --only best
```

Hai lệnh đầu sử dụng pipeline/checkpoint đã lưu. Lệnh `--only best` đọc `ablation_results.csv`, chọn mô hình sklearn và bộ feature có CV RMSE thấp nhất rồi **refit** trước khi dự đoán. Với kết quả kèm trong bài, lựa chọn này là **Gradient Boosting + E5**.

### Kiểm tra nhanh và xem tùy chọn

```bash
python _smoke_test.py
python -m src.train_mlp --help
python -m src.run_ablation --help
python -m src.predict --help
python -m src.predict_mlp --help
```

`_smoke_test.py` kiểm tra số cột và các feature tạo thêm, không phải phép đánh giá chất lượng mô hình. Ví dụ, `python -m src.run_ablation --skip-mlp` chỉ chạy Ridge và Gradient Boosting; kết quả của lần chạy này không có phần ablation MLP.

## 7. Kết quả chính và cách đọc số liệu

| Mô hình | Bộ feature | Đánh giá nội bộ | RMSE | Kaggle score |
|---|---|---|---:|---:|
| Gradient Boosting tuned | Gốc (E0) | 5-fold CV | 0.12915 | Chưa ghi nhận |
| Gradient Boosting tuned | Toàn bộ feature engineering (E5) | 5-fold CV | **0.12645** | **0.13320** |
| MLP tốt nhất, `MLP_2_lr_3e-4` | Gốc (E0) | Holdout 80/20 | 0.12940 | Chưa ghi nhận |
| MLP với cấu hình được chọn | E5 | Holdout 80/20 | 0.13189 | Chưa ghi nhận |

- Metric nội bộ là **RMSE trên `log1p(SalePrice)`**, seed **42**; giá trị càng nhỏ càng tốt.
- Sklearn sử dụng 5-fold CV; MLP sử dụng holdout 80/20. Hai cách đánh giá cần được phân biệt khi so sánh kết quả.
- E0 là feature gốc; E1 thêm `TotalSF`; E2 thêm `TotalSF` và `TotalBathrooms`; E3 thêm `HouseAge` và `RemodAge`; E4 thêm các biến nhị phân; E5 thêm toàn bộ 11 feature. Mỗi bộ được xây dựng từ dữ liệu gốc.
- Feature engineering cải thiện Gradient Boosting từ **0.12915 → 0.12645**, khoảng **2,09%**; MLP vẫn đạt kết quả tốt nhất với feature gốc.
- File đã nộp là **`submission_best.csv`**, dùng Gradient Boosting tuned + E5; Kaggle score thực tế **0.13320** được minh chứng bằng ảnh trong `outputs/figures/`.
- Trước khi lưu submission, dự đoán được đưa từ thang log về giá bán bằng `expm1`. `test.csv` không có nhãn nên không được dùng để tính RMSE nội bộ.

Các nguồn đối chiếu chi tiết:

| File kết quả | Nội dung |
|---|---|
| [sklearn_results.csv](lab03_house_price/outputs/results/sklearn_results.csv) | 10 baseline và 3 kết quả tuning, CV RMSE, độ lệch chuẩn và tham số. |
| [mlp_results.csv](lab03_house_price/outputs/results/mlp_results.csv) | 8 cấu hình MLP, kiến trúc, learning rate, dropout, validation RMSE và best epoch. |
| [feature_evaluation.csv](lab03_house_price/outputs/results/feature_evaluation.csv) | Ảnh hưởng khi thêm riêng từng engineered feature. |
| [ablation_results.csv](lab03_house_price/outputs/results/ablation_results.csv) | So sánh Ridge, Gradient Boosting và MLP trên E0–E5. |
| [experiment_results.csv](lab03_house_price/outputs/results/experiment_results.csv) | Tổng hợp mô hình, bộ feature, cách đánh giá và Kaggle score đã ghi nhận. |
| [mlp_training_history.csv](lab03_house_price/outputs/results/mlp_training_history.csv) | Lịch sử train/validation theo epoch để đối chiếu learning curve. |

## 8. Khi gặp lỗi chạy

| Hiện tượng | Cách xử lý |
|---|---|
| Không tìm thấy `data/train.csv` hoặc module `src` | Chuyển terminal về `lab03_house_price/`. Với notebook, mở JupyterLab từ thư mục này và chạy các cell theo thứ tự. |
| `ModuleNotFoundError` cho một thư viện | Cài thư viện bằng `python -m pip` trong môi trường đang dùng và chọn đúng kernel notebook. |
| Không tải được `.joblib` hoặc `.pt` | Đối chiếu phiên bản với `outputs/results/mlp_environment.txt`; có thể chạy lại bước huấn luyện tương ứng để tạo mô hình trong môi trường hiện tại. |
| Chưa có mô hình hoặc bảng ablation | Chạy bước sklearn/MLP/ablation tương ứng trước khi tạo submission. |
| Cột Kaggle score trống sau khi chạy lại | Script chỉ tạo kết quả nội bộ và CSV submission; điểm Kaggle cần được lấy từ trang kết quả của lần nộp tương ứng. |
