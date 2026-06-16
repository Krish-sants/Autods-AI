from typing import Any

import pandas as pd

_NEGATIVE_INVALID_NAME_HINTS = ("age", "price", "cost", "quantity", "count", "amount", "salary", "revenue")
_DATE_NAME_HINTS = ("date", "_dt", "datetime", "timestamp")


def _looks_like_non_negative_domain(col_name: str) -> bool:
    lowered = col_name.lower()
    return any(hint in lowered for hint in _NEGATIVE_INVALID_NAME_HINTS)


def _looks_like_date_name(col_name: str) -> bool:
    lowered = col_name.lower()
    return any(hint in lowered for hint in _DATE_NAME_HINTS)


def clean_dataset(df: pd.DataFrame, summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    actions: list[dict[str, Any]] = []
    shape_before = df.shape

    # 1. Drop fully-empty rows/columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        actions.append({"action": "drop_empty_columns", "columns": empty_cols})

    empty_rows_before = len(df)
    df = df.dropna(how="all")
    if len(df) != empty_rows_before:
        actions.append({"action": "drop_empty_rows", "rows_dropped": empty_rows_before - len(df)})

    # 2. Remove duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates()
        actions.append({"action": "drop_duplicates", "rows_dropped": dup_count})

    numeric_cols = [c for c in summary["numeric_columns"] if c in df.columns]
    categorical_cols = [c for c in summary["categorical_columns"] if c in df.columns]
    datetime_cols = [c for c in summary["datetime_columns"] if c in df.columns]

    # 3. Invalid data fixes: negative values in non-negative-domain numeric columns
    for col in numeric_cols:
        if _looks_like_non_negative_domain(col):
            invalid_count = int((df[col] < 0).sum())
            if invalid_count > 0:
                median_val = df.loc[df[col] >= 0, col].median()
                df.loc[df[col] < 0, col] = median_val
                actions.append(
                    {
                        "action": "fix_invalid_negative",
                        "column": col,
                        "rows_fixed": invalid_count,
                        "replacement": "median_of_valid_rows",
                    }
                )

    # 4. Coerce likely date columns, drop unparsable as NaT (handled by missing-value step below)
    for col in categorical_cols[:]:
        if _looks_like_date_name(col):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.5:
                df[col] = parsed
                if col in categorical_cols:
                    categorical_cols.remove(col)
                if col not in datetime_cols:
                    datetime_cols.append(col)
                actions.append({"action": "coerce_to_datetime", "column": col})

    # 5. Missing value imputation
    for col in numeric_cols:
        missing = int(df[col].isna().sum())
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            actions.append({"action": "impute_numeric_median", "column": col, "rows_filled": missing})

    for col in categorical_cols:
        missing = int(df[col].isna().sum())
        if missing > 0:
            mode_series = df[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            actions.append({"action": "impute_categorical_mode", "column": col, "rows_filled": missing})

    for col in datetime_cols:
        missing = int(df[col].isna().sum())
        if missing > 0:
            df[col] = df[col].ffill().bfill()
            actions.append({"action": "impute_datetime_ffill_bfill", "column": col, "rows_filled": missing})

    # 6. Outlier capping (winsorization) for numeric columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        if capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            actions.append({"action": "winsorize_outliers", "column": col, "rows_capped": capped})

    report = {
        "shape_before": list(shape_before),
        "shape_after": list(df.shape),
        "actions": actions,
    }
    return df, report
