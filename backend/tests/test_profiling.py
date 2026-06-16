import pandas as pd

from app.ml.profiling import profile_dataset


def test_profile_dataset_basic(synthetic_classification_df):
    summary = profile_dataset(synthetic_classification_df)

    assert summary["n_rows"] == len(synthetic_classification_df)
    assert summary["n_cols"] == synthetic_classification_df.shape[1]
    assert "tenure" in summary["numeric_columns"]
    assert "contract" in summary["categorical_columns"]
    assert summary["has_missing"] is True
    assert summary["missing_total"] > 0


def test_profile_dataset_missingness_math():
    df = pd.DataFrame({"a": [1, 2, None, None], "b": [1, 2, 3, 4]})
    summary = profile_dataset(df)
    col_a = next(c for c in summary["columns"] if c["name"] == "a")
    assert col_a["missing_count"] == 2
    assert col_a["missing_pct"] == 50.0


def test_profile_dataset_detects_datetime_like_strings():
    df = pd.DataFrame({"order_date": ["2023-01-01", "2023-01-02", "2023-01-03"], "amount": [1, 2, 3]})
    summary = profile_dataset(df)
    assert "order_date" in summary["datetime_columns"]


def test_profile_dataset_duplicate_count():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    summary = profile_dataset(df)
    assert summary["duplicate_count"] == 1
