import pandas as pd

from app.ml.target_detection import guess_target_column


def test_guess_target_exact_name_match():
    df = pd.DataFrame({"id": [1, 2, 3], "tenure": [1, 2, 3], "churn": [0, 1, 0]})
    column, confidence, reasoning = guess_target_column(df)
    assert column == "churn"
    assert confidence >= 0.9


def test_guess_target_substring_match():
    df = pd.DataFrame({"id": [1, 2, 3], "monthly_sales_amount": [10, 20, 30]})
    column, confidence, reasoning = guess_target_column(df)
    assert column == "monthly_sales_amount"
    assert confidence >= 0.5


def test_guess_target_last_column_fallback():
    df = pd.DataFrame({"foo": [1, 2, 3, 4, 5], "bar": [10.5, 20.1, 5.3, 8.8, 9.9]})
    column, confidence, reasoning = guess_target_column(df)
    assert column in df.columns
    assert confidence > 0
    assert reasoning


def test_confidence_ordering_exact_beats_fallback():
    df_exact = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df_fallback = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [9.9, 8.8, 7.7]})

    _, conf_exact, _ = guess_target_column(df_exact)
    _, conf_fallback, _ = guess_target_column(df_fallback)

    assert conf_exact > conf_fallback
