# LAB 03 - HOUSE PRICE PREDICTION

## 1. Mục tiêu bài làm

Tham gia bài toán Kaggle:

- **House Prices: Advanced Regression Techniques**
- Link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Paper Ames Housing: https://jse.amstat.org/v19n3/decock.pdf

### Yêu cầu chính

- [ ] Đọc paper để xác định bài toán.
- [ ] Xây dựng mô hình Machine Learning chuẩn bằng **scikit-learn**.
- [ ] Xây dựng **MLP bằng PyTorch**.
- [ ] Chạy thực nghiệm để tìm đặc trưng tốt giúp cải tiến mô hình.
- [ ] So sánh kết quả các mô hình.
- [ ] Tạo file submission Kaggle.
- [ ] Viết báo cáo.
- [ ] Nộp toàn bộ source code + kết quả kiểm tra + báo cáo.
- [ ] Nén file theo tên: `lab03_house_price_hoten_masv.zip`.

---

# 2. Xác định bài toán

Đây là bài toán **Supervised Learning - Regression**.

- **Input X:** các đặc trưng mô tả căn nhà.
- **Target y:** `SalePrice`.
- **Mục tiêu:** dự đoán giá bán của căn nhà.
- **Dataset:** Ames Housing.

Trong bộ dữ liệu Kaggle:

- `train.csv`: có dữ liệu đầu vào + `SalePrice`.
- `test.csv`: chỉ có dữ liệu đầu vào, không có `SalePrice`.
- `sample_submission.csv`: mẫu file nộp kết quả.

Nên train với:

```python
y = np.log1p(train["SalePrice"])
```

Khi dự đoán giá để nộp Kaggle:

```python
pred_price = np.expm1(pred_log)
```

Metric sử dụng trong quá trình thực nghiệm:

```text
RMSE trên log(SalePrice)
```

---

# 3. Phân công nhóm 4 người

## Thành viên 1 - Data + EDA + Preprocessing

### Nhiệm vụ

- [ ] Đọc paper Ames Housing.
- [ ] Xác định bài toán.
- [ ] Load `train.csv`, `test.csv`.
- [ ] Kiểm tra shape.
- [ ] Kiểm tra kiểu dữ liệu.
- [ ] Kiểm tra missing values.
- [ ] Phân tích target `SalePrice`.
- [ ] Phân tích numerical features.
- [ ] Phân tích categorical features.
- [ ] Phân tích correlation.
- [ ] Kiểm tra outlier.
- [ ] Xây dựng preprocessing pipeline dùng sklearn.

### File phụ trách

```text
notebooks/01_eda.ipynb
src/preprocessing.py
```

### Các bước EDA tối thiểu

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(train.shape)
print(test.shape)
print(train.head())
print(train.info())
print(train.describe())
```

### Kiểm tra missing value

```python
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)
```

### Các biểu đồ nên có

- [ ] Distribution của `SalePrice`.
- [ ] Distribution của `log1p(SalePrice)`.
- [ ] `OverallQual` vs `SalePrice`.
- [ ] `GrLivArea` vs `SalePrice`.
- [ ] `YearBuilt` vs `SalePrice`.
- [ ] `GarageCars` vs `SalePrice`.
- [ ] `TotalBsmtSF` vs `SalePrice`.
- [ ] Correlation heatmap cho numerical features.

### Preprocessing đề xuất

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])
```

### Kết quả phải bàn giao cho nhóm

- [ ] Danh sách numerical columns.
- [ ] Danh sách categorical columns.
- [ ] Preprocessing pipeline.
- [ ] Notebook EDA chạy được từ đầu tới cuối.

---

# 4. Thành viên 2 - Machine Learning bằng scikit-learn

## Nhiệm vụ

Dùng preprocessing của thành viên 1.

### Baseline models

- [ ] `DummyRegressor`
- [ ] `LinearRegression`
- [ ] `Ridge`
- [ ] `Lasso`
- [ ] `ElasticNet`

### Tree / Ensemble models

- [ ] `DecisionTreeRegressor`
- [ ] `RandomForestRegressor`
- [ ] `ExtraTreesRegressor`
- [ ] `GradientBoostingRegressor`
- [ ] `HistGradientBoostingRegressor` nếu phù hợp pipeline.

> Lưu ý: đây là bài toán Regression, không dùng `DecisionTreeClassifier`.

### File phụ trách

```text
notebooks/02_sklearn_models.ipynb
src/train_sklearn.py
```

### Pipeline mẫu

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=10.0))
])
```

### Cross Validation

```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### RMSE

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

rmse = -scores.mean()
print(rmse)
```

### Hyperparameter tuning

Có thể dùng:

```text
GridSearchCV
RandomizedSearchCV
```

Ví dụ Ridge:

```python
params = {
    "regressor__alpha": [0.1, 1, 5, 10, 20, 50, 100]
}
```

Random Forest:

```python
params = {
    "regressor__n_estimators": [200, 500, 800],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5, 10]
}
```

### Bảng kết quả cần tạo

| Model | CV RMSE | Best Params |
|---|---:|---|
| DummyRegressor | ... | ... |
| LinearRegression | ... | ... |
| Ridge | ... | ... |
| Lasso | ... | ... |
| RandomForest | ... | ... |
| GradientBoosting | ... | ... |

### Kết quả phải bàn giao

- [ ] File kết quả CV.
- [ ] Best hyperparameters.
- [ ] Best sklearn model.
- [ ] Bảng so sánh model.

---

# 5. Thành viên 3 - MLP bằng PyTorch

## Nhiệm vụ

- [ ] Nhận dữ liệu sau preprocessing.
- [ ] Chuyển dữ liệu sang Tensor.
- [ ] Tạo Dataset.
- [ ] Tạo DataLoader.
- [ ] Xây dựng MLP.
- [ ] Viết train loop.
- [ ] Viết validation loop.
- [ ] Early stopping.
- [ ] Save best model.
- [ ] Vẽ train loss và validation loss.
- [ ] So sánh nhiều cấu hình MLP.

### File phụ trách

```text
notebooks/03_pytorch_mlp.ipynb
src/model.py
src/train_mlp.py
```

### Kiến trúc MLP mẫu

```python
import torch
import torch.nn as nn

class HouseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
```

### Loss

```python
criterion = nn.MSELoss()
```

### Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

### Cấu hình thực nghiệm

#### MLP 1

```text
128 -> 64 -> 1
```

#### MLP 2

```text
256 -> 128 -> 64 -> 1
Dropout = 0.2
```

#### MLP 3

```text
512 -> 256 -> 128 -> 64 -> 1
BatchNorm
Dropout = 0.2
```

### Learning rate cần thử

```text
1e-2
1e-3
3e-4
```

### Dropout cần thử

```text
0.0
0.1
0.2
0.3
```

### Bảng kết quả

| Model | Hidden Layers | LR | Dropout | Val RMSE |
|---|---|---:|---:|---:|
| MLP 1 | 128-64 | 0.001 | 0.0 | ... |
| MLP 2 | 256-128-64 | 0.001 | 0.2 | ... |
| MLP 3 | 512-256-128-64 | 0.0003 | 0.2 | ... |

### Kết quả phải bàn giao

- [ ] Model PyTorch.
- [ ] Best checkpoint.
- [ ] Train/validation loss curve.
- [ ] Validation RMSE.
- [ ] File dự đoán test.

---

# 6. Thành viên 4 - Feature Engineering + Experiments + Kaggle

## Nhiệm vụ

- [ ] Tạo feature mới.
- [ ] Đánh giá từng feature.
- [ ] Chạy ablation experiments.
- [ ] So sánh trước/sau feature engineering.
- [ ] Tạo submission Kaggle.
- [ ] Tổng hợp bảng kết quả cuối.

### File phụ trách

```text
notebooks/04_feature_engineering.ipynb
src/features.py
src/predict.py
```

---

# 7. Feature Engineering cần thử

## TotalSF

```python
df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)
```

## TotalBathrooms

```python
df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)
```

## TotalPorchSF

```python
df["TotalPorchSF"] = (
    df["OpenPorchSF"]
    + df["EnclosedPorch"]
    + df["3SsnPorch"]
    + df["ScreenPorch"]
    + df["WoodDeckSF"]
)
```

## HouseAge

```python
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
```

## RemodAge

```python
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
```

## GarageAge

```python
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
```

## Binary features

```python
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
```

---

# 8. Ablation Experiment

Không thêm tất cả feature cùng lúc ngay từ đầu.

Cần chứng minh feature nào thực sự giúp model tốt hơn.

### Experiment E0

```text
Original Features
```

### Experiment E1

```text
Original + TotalSF
```

### Experiment E2

```text
Original + TotalSF + TotalBathrooms
```

### Experiment E3

```text
Original + HouseAge + RemodAge
```

### Experiment E4

```text
Original + Binary Features
```

### Experiment E5

```text
All Engineered Features
```

### Bảng kết quả

| Experiment | Features | Ridge RMSE | Gradient Boosting RMSE | MLP RMSE |
|---|---|---:|---:|---:|
| E0 | Original | ... | ... | ... |
| E1 | + TotalSF | ... | ... | ... |
| E2 | + TotalBathrooms | ... | ... | ... |
| E3 | + Age features | ... | ... | ... |
| E4 | + Binary features | ... | ... | ... |
| E5 | All engineered | ... | ... | ... |

### Cách nhận xét

Không viết:

```text
TotalSF là feature tốt.
```

Nên viết:

```text
Baseline RMSE: 0.xxx
Sau khi thêm TotalSF: 0.xxx
Mức cải thiện: ...
```

---

# 9. Kaggle Submission

Format:

```csv
Id,SalePrice
1461,208500
1462,181500
...
```

Nếu model dự đoán `log1p(SalePrice)`:

```python
pred_log = model.predict(X_test)
pred_price = np.expm1(pred_log)
```

Tạo submission:

```python
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": pred_price
})

submission.to_csv(
    "outputs/submission_best.csv",
    index=False
)
```

### File nên tạo

```text
outputs/submission_sklearn.csv
outputs/submission_mlp.csv
outputs/submission_best.csv
```

---

# 10. Cấu trúc source code

```text
lab03_house_price/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_sklearn_models.ipynb
│   ├── 03_pytorch_mlp.ipynb
│   └── 04_feature_engineering.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_sklearn.py
│   ├── model.py
│   ├── train_mlp.py
│   ├── evaluate.py
│   └── predict.py
│
├── outputs/
│   ├── figures/
│   ├── experiment_results.csv
│   ├── submission_sklearn.csv
│   ├── submission_mlp.csv
│   └── submission_best.csv
│
└── report/
    └── Lab03_House_Price_Report.pdf
```

---

# 11. requirements.txt

Có thể dùng:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
torch
```

Nếu nhóm dùng thêm thư viện thì bổ sung vào file.

---

# 12. Nội dung README.md

README nên ghi:

```text
1. Tên bài
2. Thành viên nhóm
3. Cách cài môi trường
4. Cách chạy EDA
5. Cách chạy sklearn
6. Cách chạy PyTorch
7. Cách chạy prediction
8. Kết quả tốt nhất
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

---

# 13. Cấu trúc báo cáo

## Trang bìa

```text
LAB 03
HOUSE PRICE PREDICTION
House Prices: Advanced Regression Techniques

Giảng viên: ...................
Lớp: .........................

THÀNH VIÊN
1. Họ tên - MSSV
2. Họ tên - MSSV
3. Họ tên - MSSV
4. Họ tên - MSSV
```

## Nội dung

```text
1. Phân công công việc

2. Giới thiệu bài toán
   2.1 Ames Housing Dataset
   2.2 Mục tiêu bài toán
   2.3 Evaluation metric

3. Phân tích dữ liệu
   3.1 Dataset
   3.2 Missing values
   3.3 SalePrice
   3.4 Correlation
   3.5 Outliers

4. Tiền xử lý dữ liệu
   4.1 Missing data
   4.2 Categorical encoding
   4.3 Feature scaling
   4.4 Log transformation

5. Machine Learning với sklearn
   5.1 Baseline
   5.2 Ridge
   5.3 Random Forest
   5.4 Gradient Boosting
   5.5 Hyperparameter tuning

6. MLP bằng PyTorch
   6.1 Architecture
   6.2 Loss function
   6.3 Optimizer
   6.4 Training
   6.5 Validation

7. Feature Engineering
   7.1 Feature creation
   7.2 Ablation experiments

8. Kết quả thực nghiệm
   8.1 So sánh sklearn models
   8.2 So sánh MLP
   8.3 Ảnh hưởng của Feature Engineering
   8.4 Kaggle score

9. Kết luận

10. Tài liệu tham khảo
```

---

# 14. Bảng phân công đưa vào báo cáo

| STT | Thành viên | Công việc |
|---|---|---|
| 1 | Họ tên 1 | Nghiên cứu paper, phân tích dữ liệu, EDA, xử lý missing data, xây dựng preprocessing pipeline |
| 2 | Họ tên 2 | Xây dựng các mô hình Machine Learning bằng scikit-learn, cross-validation, hyperparameter tuning |
| 3 | Họ tên 3 | Xây dựng MLP bằng PyTorch, DataLoader, training loop, validation, early stopping |
| 4 | Họ tên 4 | Feature Engineering, ablation experiments, Kaggle submission, tổng hợp kết quả |
| Cả nhóm | | Kiểm tra kết quả, viết và rà soát báo cáo |
| Nhóm trưởng | | Tích hợp source code, chuẩn hóa báo cáo, đóng gói file nộp |

---

# 15. Tiến độ thực hiện đề xuất

## Ngày 1

### Thành viên 1

- [ ] Đọc paper.
- [ ] EDA.
- [ ] Missing values.
- [ ] Preprocessing.
- [ ] Hoàn thành `01_eda.ipynb`.

### Cả nhóm

- [ ] Chốt preprocessing chung.
- [ ] Chốt metric chung.
- [ ] Chốt random seed = 42.

---

## Ngày 2

### Thành viên 2

- [ ] Baseline sklearn.
- [ ] Ridge/Lasso.
- [ ] Random Forest.
- [ ] Gradient Boosting.

### Thành viên 3

- [ ] Dataset/DataLoader.
- [ ] MLP baseline.
- [ ] Train/validation loop.

### Thành viên 4

- [ ] Viết `features.py`.
- [ ] Tạo feature engineering.
- [ ] Chuẩn bị ablation experiments.

---

## Ngày 3

### Thành viên 2

- [ ] Hyperparameter tuning.
- [ ] Chốt best sklearn model.

### Thành viên 3

- [ ] Chạy 3 MLP configurations.
- [ ] Chốt best MLP.

### Thành viên 4

- [ ] Chạy E0-E5.
- [ ] Tổng hợp experiment results.

---

## Ngày 4

### Cả nhóm

- [ ] Tạo Kaggle submission.
- [ ] Ghi Kaggle score.
- [ ] Chụp ảnh kết quả nếu cần đưa vào report.
- [ ] Mỗi người viết phần báo cáo mình phụ trách.

---

## Ngày 5

### Nhóm trưởng

- [ ] Ghép báo cáo.
- [ ] Kiểm tra lại source code.
- [ ] Chạy lại notebook từ đầu đến cuối.
- [ ] Xóa file thừa.
- [ ] Kiểm tra requirements.
- [ ] Kiểm tra submission.
- [ ] Nén file.

---

# 16. Quy tắc làm chung để tránh lỗi

## Dùng chung random seed

```python
RANDOM_STATE = 42
```

PyTorch:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## Không đánh giá bằng test.csv

`test.csv` không có `SalePrice`.

Dùng:

```text
Cross Validation
hoặc
Train / Validation split
```

để đánh giá model.

## Không xử lý train và test theo hai cách khác nhau

Phải fit preprocessing trên train và transform validation/test.

Ví dụ:

```python
preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)
```

## Không data leakage

Không được fit scaler, imputer, encoder trên toàn bộ dữ liệu trước khi cross-validation.

Nên đặt preprocessing trong sklearn `Pipeline`.

---

# 17. Bảng kết quả cuối cùng cần có

| Model | Feature Set | CV / Val RMSE | Kaggle Score |
|---|---|---:|---:|
| Ridge | Original | ... | ... |
| Ridge | Engineered | ... | ... |
| Random Forest | Original | ... | ... |
| Gradient Boosting | Engineered | ... | ... |
| MLP | Original | ... | ... |
| MLP | Engineered | ... | ... |

Sau đó kết luận dựa trên kết quả thực nghiệm.

---

# 18. Checklist trước khi nộp

## Source code

- [ ] `01_eda.ipynb` chạy được.
- [ ] `02_sklearn_models.ipynb` chạy được.
- [ ] `03_pytorch_mlp.ipynb` chạy được.
- [ ] `04_feature_engineering.ipynb` chạy được.
- [ ] Source trong `src/` không lỗi import.
- [ ] `requirements.txt` đầy đủ.

## Experiments

- [ ] Có baseline.
- [ ] Có sklearn model.
- [ ] Có PyTorch MLP.
- [ ] Có feature engineering.
- [ ] Có ablation study.
- [ ] Có RMSE.
- [ ] Có bảng so sánh.

## Kaggle

- [ ] Có file submission đúng format.
- [ ] Có Kaggle score.
- [ ] Ghi rõ model nào được dùng để submit.

## Report

- [ ] Trang bìa.
- [ ] Họ tên + MSSV đủ 4 thành viên.
- [ ] Phân công công việc.
- [ ] Mô tả bài toán.
- [ ] EDA.
- [ ] Preprocessing.
- [ ] sklearn models.
- [ ] PyTorch MLP.
- [ ] Feature Engineering.
- [ ] Kết quả thực nghiệm.
- [ ] Kaggle result.
- [ ] Kết luận.
- [ ] Tài liệu tham khảo.

## File nộp

- [ ] Tên file đúng format:

```text
lab03_house_price_hoten_masv.zip
```

- [ ] Mở ZIP kiểm tra lại sau khi nén.

---

# 19. Thứ tự ưu tiên nếu thời gian ít

Nếu gần deadline, làm theo thứ tự:

1. [ ] Preprocessing chạy đúng.
2. [ ] Ridge baseline.
3. [ ] GradientBoosting/RandomForest.
4. [ ] MLP PyTorch baseline.
5. [ ] `TotalSF`, `TotalBathrooms`, `HouseAge`.
6. [ ] Ablation experiment.
7. [ ] Kaggle submission.
8. [ ] Báo cáo.
9. [ ] Tuning sâu hơn nếu còn thời gian.

---

# 20. Nguồn tham khảo

1. Kaggle - House Prices: Advanced Regression Techniques  
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

2. Dean De Cock - Ames Housing Data  
   https://jse.amstat.org/v19n3/decock.pdf

3. File code tham khảo đã được cung cấp trong đề/bài làm trước đó. Nên dùng để tham khảo workflow, không sao chép nguyên xi; đặc biệt cần sửa các phần chưa phù hợp với bài toán regression và triển khai MLP bằng PyTorch theo đúng yêu cầu.

---

# 21. Definition of Done cho từng người

## Người 1 hoàn thành khi

- [ ] Notebook EDA chạy toàn bộ không lỗi.
- [ ] Có preprocessing pipeline dùng chung.
- [ ] Có ít nhất 5 biểu đồ EDA.
- [ ] Có phần viết báo cáo Data + EDA + Preprocessing.

## Người 2 hoàn thành khi

- [ ] Có ít nhất 4 sklearn regression models.
- [ ] Có 5-fold CV.
- [ ] Có tuning ít nhất 1 model.
- [ ] Có bảng RMSE.
- [ ] Có best sklearn model.

## Người 3 hoàn thành khi

- [ ] MLP viết bằng PyTorch.
- [ ] Có DataLoader.
- [ ] Có train + validation loop.
- [ ] Có early stopping hoặc lưu best checkpoint.
- [ ] Có loss curve.
- [ ] Có bảng so sánh tối thiểu 3 cấu hình.

## Người 4 hoàn thành khi

- [ ] Có `features.py`.
- [ ] Có ít nhất 5 engineered features.
- [ ] Có ablation study.
- [ ] Có bảng trước/sau feature engineering.
- [ ] Có ít nhất 1 submission Kaggle hợp lệ.

## Cả nhóm hoàn thành khi

- [ ] Có báo cáo PDF.
- [ ] Có source code.
- [ ] Có output thí nghiệm.
- [ ] Có submission.
- [ ] ZIP đúng tên.
- [ ] Một thành viên khác nhóm trưởng đã thử giải nén và chạy lại project.
