import pandas as pd

CLASSIFICATION = "classification"
REGRESSION = "regression"
CLUSTERING = "clustering"


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

    # Numeric target: treat as classification if low cardinality relative to row count
    # (e.g. 0/1 flags, small category codes), otherwise regression.
    if cardinality <= 2:
        return CLASSIFICATION
    if cardinality <= 20 and cardinality / len(series) < 0.05:
        return CLASSIFICATION

    return REGRESSION
