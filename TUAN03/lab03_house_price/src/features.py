"""Feature engineering utilities."""


def add_features(df):
    df = df.copy()

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0)
            + df["1stFlrSF"].fillna(0)
            + df["2ndFlrSF"].fillna(0)
        )

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    return df
