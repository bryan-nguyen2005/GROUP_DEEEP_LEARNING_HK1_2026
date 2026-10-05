"""Feature engineering cho Lab 03 House Price Prediction — Thành viên 4.

Module này tạo các feature mới theo plan mục 7 và khai báo các tổ hợp
feature dùng cho ablation study E0–E5 (plan mục 8).

Thiết kế quan trọng:

    - Công thức chỉ dựa trên cột gốc, không đụng tới target nên thêm feature
      không gây rò rỉ thông tin (leakage). Preprocessing vẫn được ``fit`` trên
      tập train của từng lần đánh giá như thường lệ.
    - ``add_features`` mặc định thêm TOÀN BỘ feature (giữ tương thích ngược với
      ``train_mlp.py`` / ``predict_mlp.py`` vốn gọi ``add_features(X)``).
    - Muốn thêm đúng một nhóm feature (cho ablation), truyền ``features=...``
      với tên feature hoặc tên nhóm trong :data:`FEATURE_GROUPS`.

Ví dụ::

    from src.features import add_features, ABLATION_EXPERIMENTS

    X_e1 = add_features(X, features=ABLATION_EXPERIMENTS["E1"][1])
    X_e5 = add_features(X)                      # tất cả feature
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Định nghĩa feature
# ---------------------------------------------------------------------------

# Thứ tự cố định, dùng để sinh cột theo đúng một trình tự nhất định.
ENGINEERED_FEATURES = (
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF",
    "HouseAge",
    "RemodAge",
    "GarageAge",
    "HasGarage",
    "HasBsmt",
    "HasFireplace",
    "HasPool",
    "Has2ndFloor",
)

# Nhóm feature theo plan mục 7 — dùng cho ablation và báo cáo.
FEATURE_GROUPS = {
    "totalsf": ("TotalSF",),
    "totalbathrooms": ("TotalBathrooms",),
    "totalporchsf": ("TotalPorchSF",),
    "age": ("HouseAge", "RemodAge", "GarageAge"),
    "binary": ("HasGarage", "HasBsmt", "HasFireplace", "HasPool", "Has2ndFloor"),
}

# Ablation study E0–E5 theo plan mục 8:
#   E0 Original | E1 +TotalSF | E2 +TotalSF+TotalBathrooms
#   E3 +HouseAge+RemodAge | E4 +binary | E5 all engineered
ABLATION_EXPERIMENTS = {
    "E0": ("Original", ()),
    "E1": ("+ TotalSF", FEATURE_GROUPS["totalsf"]),
    "E2": ("+ TotalSF + TotalBathrooms",
           FEATURE_GROUPS["totalsf"] + FEATURE_GROUPS["totalbathrooms"]),
    # E3 theo plan chỉ dùng HouseAge + RemodAge (chưa có GarageAge).
    "E3": ("+ HouseAge + RemodAge", ("HouseAge", "RemodAge")),
    "E4": ("+ Binary features", FEATURE_GROUPS["binary"]),
    "E5": ("All engineered features", ENGINEERED_FEATURES),
}


def _resolve_features(features):
    """Chuẩn hóa danh sách feature thành tuple tên feature hợp lệ.

    Chấp nhận tên feature trực tiếp (``"TotalSF"``) hoặc tên nhóm trong
    :data:`FEATURE_GROUPS` (``"binary"``). ``None`` nghĩa là toàn bộ feature.
    """
    if features is None:
        return ENGINEERED_FEATURES
    if isinstance(features, str):
        features = [features]

    resolved = []
    for item in features:
        key = str(item).lower()
        if key in FEATURE_GROUPS:
            resolved.extend(FEATURE_GROUPS[key])
        elif item in ENGINEERED_FEATURES:
            resolved.append(item)
        else:
            raise ValueError(
                f"Unknown feature or group: {item!r}. "
                f"Valid features={list(ENGINEERED_FEATURES)}, "
                f"valid groups={list(FEATURE_GROUPS)}."
            )

    # Giữ thứ tự chuẩn hóa, loại trùng (E2 ghép hai nhóm có thể trùng tên).
    known_order = {name: i for i, name in enumerate(ENGINEERED_FEATURES)}
    unique = set(resolved)
    return tuple(sorted(unique, key=known_order.__getitem__))


def compute_engineered(df):
    """Tính toàn bộ feature mới, trả về dict ``{tên cột: Series}``.

    Các cột nguồn có thể vắng mặt (ví dụ dữ liệu khác bộ Ames) thì feature
    tương ứng sẽ không được tính — caller chỉ lấy feature mình cần.
    Công thức khớp plan mục 7:

    - ``TotalSF``: tổng diện tích tầng hầm + tầng 1 + tầng 2.
    - ``TotalBathrooms``: số phòng tắm quy đổi (nửa điểm cho half bath).
    - ``TotalPorchSF``: tổng diện tích các loại porch + sàn gỗ.
    - ``HouseAge`` / ``RemodAge`` / ``GarageAge``: số năm đến thời điểm bán.
    - ``Has*``: cờ nhị phân 0/1 đánh dấu sự hiện diện của hạng mục.
    """
    out = {}

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        out["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0)
            + df["1stFlrSF"].fillna(0)
            + df["2ndFlrSF"].fillna(0)
        )

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(df.columns):
        out["TotalBathrooms"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                  "ScreenPorch", "WoodDeckSF"]
    if all(c in df.columns for c in porch_cols):
        total_porch = sum(df[c].fillna(0) for c in porch_cols)
        out["TotalPorchSF"] = total_porch

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        out["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        out["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # GarageYrBlt bị thiếu khi nhà không có garage -> GarageAge cũng thiếu,
    # để nguyên NaN để SimpleImputer điền median (giữ nhất quán với pipeline).
    if {"YrSold", "GarageYrBlt"}.issubset(df.columns):
        out["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

    if "GarageArea" in df.columns:
        out["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        out["HasBsmt"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    if "Fireplaces" in df.columns:
        out["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    if "PoolArea" in df.columns:
        out["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        out["Has2ndFloor"] = (df["2ndFlrSF"].fillna(0) > 0).astype(int)

    return out


def add_features(df, features=None):
    """Thêm feature mới vào DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Khung dữ liệu chứa các cột gốc của bộ Ames.
    features : None, str or iterable of str, optional
        Danh sách feature (hoặc tên nhóm trong :data:`FEATURE_GROUPS`) cần
        thêm. ``None`` (mặc định) thêm toàn bộ :data:`ENGINEERED_FEATURES`.

    Returns
    -------
    pandas.DataFrame
        Bản sao của ``df`` với các cột mới được thêm vào.

    Raises
    ------
    ValueError
        Nếu ``features`` chứa tên feature/ nhóm không hợp lệ, hoặc yêu cầu
        một feature mà cột nguồn tương ứng không tồn tại trong ``df``.
    """
    requested = _resolve_features(features)
    computed = compute_engineered(df)

    missing_source = [name for name in requested if name not in computed]
    if missing_source:
        raise ValueError(
            f"Cannot compute {missing_source}: source columns are missing."
        )

    result = df.copy()
    for name in requested:
        result[name] = computed[name]
    return result
