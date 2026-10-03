"""Tiền xử lý dữ liệu dùng chung cho nhóm — Lab 03 House Price Prediction.

Module này là nguồn sự thật duy nhất về cách xử lý missing values, chia cột
numeric/categorical và tạo pipeline. Thành viên 2 và 3 import từ đây để mọi
mô hình dùng cùng một cách tiền xử lý.

Nguyên tắc quan trọng (plan mục 16):
    Pipeline phải được ``fit`` trên tập train của mỗi lần đánh giá rồi mới
    ``transform`` validation/test. Tuyệt đối không fit trên toàn bộ dữ liệu
    trước khi cross-validation, nếu không sẽ rò rỉ thông tin (data leakage).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Thư mục gốc của project: src/ -> lab03_house_price/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_STATE = 42

# "Id" là mã dòng, không mang thông tin về căn nhà -> loại khỏi feature.
ID_COLUMN = "Id"

# Target gốc và target dùng để train (log1p theo plan mục 2).
TARGET_COLUMN = "SalePrice"
TARGET_LOG_COLUMN = "SalePrice_log"

# Cột categorical đã bị pandas đọc thành chuỗi, khai báo tường minh để
# select_dtypes không báo deprecated trên pandas 3.x.
STRING_DTYPES = ["object", "str", "string"]


def load_data(data_dir=DATA_DIR):
    """Đọc train.csv và test.csv.

    Parameters
    ----------
    data_dir : Path or str
        Thư mục chứa train.csv và test.csv.

    Returns
    -------
    train, test : pandas.DataFrame
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def add_target_log(df):
    """Thêm cột target dạng log: ``SalePrice_log = log1p(SalePrice)``.

    Theo plan mục 2, toàn bộ thực nghiệm train trên ``log1p(SalePrice)`` và
    chấm điểm bằng RMSE trên log. ``SalePrice`` lệch phải mạnh (skew ~ 1.88)
    nên biến đổi log làm phân phối gần đối xứng hơn nhiều.
    """
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        df[TARGET_LOG_COLUMN] = np.log1p(df[TARGET_COLUMN])
    return df


def split_feature_target(df, drop_id=True):
    """Tách X (đặc trưng) và y (target dạng log).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame train đã đi qua :func:`add_target_log`.
    drop_id : bool
        Bỏ cột ``Id`` khỏi X vì đây chỉ là mã dòng.

    Returns
    -------
    X, y : pandas.DataFrame, pandas.Series
    """
    X = df.drop(columns=[TARGET_COLUMN, TARGET_LOG_COLUMN])
    y = df[TARGET_LOG_COLUMN]
    if drop_id and ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])
    return X, y


def select_column_groups(X, cast_mssubclass=False):
    """Chia cột thành hai nhóm numeric và categorical.

    Mặc định giữ nguyên dtype gốc của Kaggle (``MSSubClass`` vẫn là int64) để
    baseline trung thực với dữ liệu gốc. ``MSSubClass`` thực chất là mã loại
    nhà chứ không phải số đo, nên có thể đưa sang nhóm categorical bằng
    ``cast_mssubclass=True`` để chạy thực nghiệm so sánh — xem notebook EDA.

    Parameters
    ----------
    X : pandas.DataFrame
    cast_mssubclass : bool
        Nếu True, ép ``MSSubClass`` sang chuỗi trước khi phân nhóm.

    Returns
    -------
    numeric_cols, categorical_cols : list of str
    """
    X = X.copy()
    if cast_mssubclass:
        X["MSSubClass"] = X["MSSubClass"].astype(str)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=STRING_DTYPES).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True,
                       sparse_threshold=0.3):
    """Tạo ColumnTransformer xử lý missing values và encode categorical.

    - Numeric: điền median rồi chuẩn hoá (StandardScaler). Median được dùng vì
      các cột kiểu LotFrontage, MasVnrArea có phân phối lệch.
    - Categorical: điền giá trị phổ biến nhất rồi one-hot encode.
      ``handle_unknown="ignore"`` để transform được tập test có thể chứa
      category chưa từng xuất hiện trong train.
    - Cột không nằm trong hai nhóm trên sẽ bị bỏ qua (remainder="drop").

    Parameters
    ----------
    numeric_cols, categorical_cols : list of str
        Danh sách cột, lấy từ :func:`select_column_groups`.
    scale_numeric : bool
        Có chuẩn hoá numeric bằng StandardScaler hay không. Ridge/MLP cần bật;
        mô hình cây không bị ảnh hưởng.
    sparse_threshold : float
        Ngưỡng mật độ để quyết định trả về sparse hay dense. 0.3 (mặc định) cho
        sparse; đặt 0 để luôn trả về ma trận dense.

    Notes
    -----
    Về kiểu dữ liệu đầu ra (quan trọng cho thành viên 2 và 3):
        Với 43 cột categorical, one-hot tạo khoảng 287 cột nhưng mật độ chỉ
        ~28%, nên ColumnTransformer mặc định trả về ma trận sparse. Hầu hết mô
        hình sklearn chấp nhận sparse, NHƯNG:
          - ``HistGradientBoostingRegressor`` KHÔNG nhận sparse, cần đặt
            ``sparse_threshold=0``.
          - ``torch`` cũng không nhận sparse, nên thành viên 3 cần
            ``.toarray()``. Ma trận 1460x287 chỉ khoảng 3 MB nên dense vẫn rất nhẹ.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        # Nếu mật độ thấp hơn sparse_threshold thì trả về CSR sparse (tiết kiệm
        # RAM). Đặt sparse_threshold=0 để ép dense khi dùng HistGradientBoosting
        # hoặc PyTorch.
        sparse_threshold=sparse_threshold,
    )


def prepare_features(train, test=None, cast_mssubclass=False):
    """Tiền xử lý train (và test) thành ma trận đã encode.

    Dùng cho bước khám phá hoặc khi chỉ cần xem ma trận đầu vào. Với
    cross-validation KHÔNG dùng hàm này — hãy đặt ``build_preprocessor``
    vào trong ``Pipeline`` để tránh rò rỉ thông tin.

    Parameters
    ----------
    train, test : pandas.DataFrame
    cast_mssubclass : bool

    Returns
    -------
    X_train_processed, y, X_test_processed
        ``y`` là np.ndarray của ``log1p(SalePrice)``.
        ``X_test_processed`` là None nếu không truyền test.
    """
    feature_cols = [
        c for c in train.columns
        if c not in (TARGET_COLUMN, TARGET_LOG_COLUMN, ID_COLUMN)
    ]
    X_train = train[feature_cols]

    numeric_cols, categorical_cols = select_column_groups(
        X_train, cast_mssubclass=cast_mssubclass
    )
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = None
    if test is not None:
        X_test_processed = preprocessor.transform(test[feature_cols])

    return X_train_processed, np.log1p(train[TARGET_COLUMN]), X_test_processed
