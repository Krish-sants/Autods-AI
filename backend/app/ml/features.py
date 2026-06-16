from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

LOW_CARDINALITY_THRESHOLD = 15


def _decompose_datetime_columns(df: pd.DataFrame, datetime_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added: list[str] = []
    for col in datetime_cols:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], errors="coerce", format="mixed")
        for part, fn in (
            ("day", lambda s: s.dt.day),
            ("month", lambda s: s.dt.month),
            ("quarter", lambda s: s.dt.quarter),
            ("year", lambda s: s.dt.year),
            ("week", lambda s: s.dt.isocalendar().week.astype("int64")),
            ("is_weekend", lambda s: (s.dt.dayofweek >= 5).astype(int)),
        ):
            new_col = f"{col}_{part}"
            df[new_col] = fn(series)
            added.append(new_col)
        df = df.drop(columns=[col])
    return df, added


def engineer_features(
    df: pd.DataFrame,
    target_column: str | None,
    datetime_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, ColumnTransformer, dict[str, Any]]:
    datetime_columns = [c for c in (datetime_columns or []) if c != target_column]
    df, added_datetime_features = _decompose_datetime_columns(df, datetime_columns)

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        y = None
        X = df.copy()

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    low_card = [c for c in categorical_features if X[c].nunique(dropna=True) <= LOW_CARDINALITY_THRESHOLD]
    high_card = [c for c in categorical_features if c not in low_card]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))
    if low_card:
        low_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat_low", low_card_pipeline, low_card))
    if high_card:
        high_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        transformers.append(("cat_high", high_card_pipeline, high_card))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    report = {
        "datetime_columns_decomposed": datetime_columns,
        "new_features_added": added_datetime_features,
        "numeric_features": numeric_features,
        "low_cardinality_categorical": low_card,
        "high_cardinality_categorical": high_card,
        "n_features_before_encoding": X.shape[1],
    }

    return X, y, preprocessor, report


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def save_feature_map(preprocessor: ColumnTransformer, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_features": get_output_feature_names(preprocessor),
        **report,
    }
