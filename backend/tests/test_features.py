import pickle

import pandas as pd

from app.ml.features import engineer_features, get_output_feature_names, save_feature_map


def test_engineer_features_basic_shapes(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, report = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    assert y.name == "churn"
    assert "signup_date" not in X.columns
    assert any(c.startswith("signup_date_") for c in X.columns)
    assert "monthly_charges" in report["numeric_features"]
    assert "contract" in report["low_cardinality_categorical"]


def test_engineer_features_preprocessor_is_picklable(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, report = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    fitted = preprocessor.fit(X, y)
    pickle.dumps(fitted)  # should not raise

    transformed = fitted.transform(X)
    assert transformed.shape[0] == len(X)


def test_get_output_feature_names_and_feature_map(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, report = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])
    preprocessor.fit(X, y)

    names = get_output_feature_names(preprocessor)
    assert len(names) > 0

    feature_map = save_feature_map(preprocessor, report)
    assert feature_map["output_features"] == names
    assert "numeric_features" in feature_map


def test_engineer_features_no_target_for_clustering(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id", "churn"])
    X, y, preprocessor, report = engineer_features(df, target_column=None, datetime_columns=["signup_date"])
    assert y is None
    assert X.shape[0] == len(df)
