import pandas as pd

_TARGET_NAME_HINTS = [
    "churn", "target", "label", "class", "outcome", "y",
    "price", "sales", "revenue", "profit", "income",
    "fraud", "default", "disease", "diagnosis", "spam",
    "survived", "purchase", "converted", "response",
]


def guess_target_column(df: pd.DataFrame) -> tuple[str | None, float, str]:
    if df.shape[1] == 0:
        return None, 0.0, "Dataset has no columns."

    lowered = {col: col.lower() for col in df.columns}

    # 1. Exact name match
    for hint in _TARGET_NAME_HINTS:
        for col, low in lowered.items():
            if low == hint:
                return col, 0.95, f"Column name '{col}' is an exact match for a common target name ('{hint}')."

    # 2. Substring match
    for hint in _TARGET_NAME_HINTS:
        for col, low in lowered.items():
            if hint in low:
                return col, 0.75, f"Column name '{col}' contains the common target keyword '{hint}'."

    # 3. Heuristic: low-cardinality non-id-like column near the end, or last column fallback
    candidate_cols = [c for c in df.columns if not lowered[c].endswith("id") and lowered[c] != "index"]
    pool = candidate_cols or list(df.columns)

    for col in reversed(pool):
        cardinality = df[col].nunique(dropna=True)
        if 1 < cardinality <= max(20, int(len(df) * 0.05)):
            return col, 0.4, f"Column '{col}' is near the end of the dataset and has low cardinality ({cardinality})."

    last_col = pool[-1]
    return last_col, 0.2, f"No strong signal found; defaulting to the last column '{last_col}'."
