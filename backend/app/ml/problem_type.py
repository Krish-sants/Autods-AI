import pandas as pd

CLASSIFICATION = "classification"
REGRESSION = "regression"
CLUSTERING = "clustering"
FORECASTING = "forecasting"


def _has_usable_datetime(df: pd.DataFrame, exclude_col: str) -> bool:
    """True if df has a datetime index or at least one datetime column (other than target)."""
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return True
    return any(
        pd.api.types.is_datetime64_any_dtype(df[c])
        for c in df.columns
        if c != exclude_col
    )


def infer_problem_type(df: pd.DataFrame, target_column: str | None) -> str:
    if not target_column or target_column not in df.columns:
        return CLUSTERING

    series = df[target_column].dropna()
    if series.empty:
        return CLUSTERING

    cardinality = series.nunique()

    if pd.api.types.is_bool_dtype(series):
        return CLASSIFICATION

    if not pd.api.types.is_numeric_dtype(series):
        return CLASSIFICATION

    # Numeric target — check for time-series before classification/regression heuristics
    if _has_usable_datetime(df, target_column) and cardinality > 20:
        return FORECASTING

    # Numeric target: treat as classification if low cardinality relative to row count
    if cardinality <= 2:
        return CLASSIFICATION
    if cardinality <= 20 and cardinality / len(series) < 0.05:
        return CLASSIFICATION

    return REGRESSION
