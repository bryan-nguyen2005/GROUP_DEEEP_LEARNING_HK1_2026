"""Smoke test nhanh cho src/features.py (thành viên 4). Xóa sau khi chạy."""

from src.features import (
    ABLATION_EXPERIMENTS,
    ENGINEERED_FEATURES,
    _resolve_features,
    add_features,
)
import pandas as pd

train = pd.read_csv("data/train.csv")
X = train.drop(columns=["Id", "SalePrice"])
Xe = add_features(X)
print("engineered cols:", [c for c in Xe.columns if c in ENGINEERED_FEATURES])
print("shape", X.shape, "->", Xe.shape)
print("E2 =", ABLATION_EXPERIMENTS["E2"][1])
print("resolve binary =", _resolve_features("binary"))
X1 = add_features(X, features=ABLATION_EXPERIMENTS["E1"][1])
print("E1 cols added:", [c for c in X1.columns if c not in X.columns])
print("nulls GarageAge:", int(Xe["GarageAge"].isna().sum()))
print(Xe[list(ENGINEERED_FEATURES)].head(3).to_string())
