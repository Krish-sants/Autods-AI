from typing import Any

import numpy as np
import pandas as pd


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # try to sniff datetime-looking string columns (pandas 3.x defaults string
    # columns to a "str" dtype rather than legacy "object", so check broadly)
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        sample = series.dropna().head(20)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                return "datetime"
    return "categorical"


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    columns: list[dict[str, Any]] = []

    numeric_cols, categorical_cols, datetime_cols, boolean_cols = [], [], [], []

    for col in df.columns:
        series = df[col]
        kind = _column_kind(series)
        missing = int(series.isna().sum())
        cardinality = int(series.nunique(dropna=True))

        if kind == "numeric":
            numeric_cols.append(col)
        elif kind == "categorical":
            categorical_cols.append(col)
        elif kind == "datetime":
            datetime_cols.append(col)
        elif kind == "boolean":
            boolean_cols.append(col)

        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "kind": kind,
                "missing_count": missing,
                "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
                "cardinality": cardinality,
            }
        )

    duplicate_count = int(df.duplicated().sum())

    potential_targets = [
        c["name"]
        for c in columns
        if c["kind"] in ("categorical", "numeric", "boolean")
        and 1 < c["cardinality"] < n_rows
        and c["missing_pct"] < 50
    ]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "duplicate_count": duplicate_count,
        "missing_total": int(df.isna().sum().sum()),
        "has_missing": bool(df.isna().sum().sum() > 0),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "columns": columns,
        "potential_targets": potential_targets[:10],
        "sample_rows": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }
