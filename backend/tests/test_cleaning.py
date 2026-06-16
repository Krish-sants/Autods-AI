import pandas as pd

from app.ml.cleaning import clean_dataset
from app.ml.profiling import profile_dataset


def test_clean_dataset_imputes_missing_numeric():
    df = pd.DataFrame({"age": [10, 20, None, 40], "city": ["a", "b", "c", "d"]})
    summary = profile_dataset(df)
    cleaned, report = clean_dataset(df, summary)

    assert cleaned["age"].isna().sum() == 0
    actions = [a["action"] for a in report["actions"]]
    assert "impute_numeric_median" in actions


def test_clean_dataset_imputes_missing_categorical():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "city": ["a", "a", None, "b"]})
    summary = profile_dataset(df)
    cleaned, report = clean_dataset(df, summary)

    assert cleaned["city"].isna().sum() == 0
    actions = [a["action"] for a in report["actions"]]
    assert "impute_categorical_mode" in actions


def test_clean_dataset_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    summary = profile_dataset(df)
    cleaned, report = clean_dataset(df, summary)

    assert len(cleaned) == 2
    assert report["shape_before"] == [3, 2]
    assert report["shape_after"] == [2, 2]


def test_clean_dataset_fixes_negative_age():
    df = pd.DataFrame({"age": [10, 20, -5, 40, 50, 60, 70, 80, 90, 25]})
    summary = profile_dataset(df)
    cleaned, report = clean_dataset(df, summary)

    assert (cleaned["age"] >= 0).all()
    actions = [a["action"] for a in report["actions"]]
    assert "fix_invalid_negative" in actions


def test_clean_dataset_caps_outliers():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    summary = profile_dataset(df)
    cleaned, report = clean_dataset(df, summary)

    assert cleaned["value"].max() < 1000
    actions = [a["action"] for a in report["actions"]]
    assert "winsorize_outliers" in actions
