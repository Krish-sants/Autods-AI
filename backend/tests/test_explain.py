import numpy as np

from app.ml.explain import compute_shap
from app.ml.features import engineer_features
from app.ml.train import train_and_tune_all


def test_compute_shap_for_random_forest(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(X, y, "classification", preprocessor, cv_folds=2, n_trials=2, timeout_s=30, random_state=42)
    rf_result = next(r for r in output.results if r.model_id == "random_forest")
    assert rf_result.error is None

    shap_results = compute_shap(rf_result.estimator, "random_forest", output.X_test, max_samples=50)

    assert shap_results["available"] is True
    assert len(shap_results["global_importance"]) > 0
    for item in shap_results["global_importance"]:
        assert np.isfinite(item["mean_abs_shap"])

    assert len(shap_results["local_examples"]) > 0
    for example in shap_results["local_examples"]:
        for contribution in example["contributions"]:
            assert np.isfinite(contribution["shap_value"])


def test_compute_shap_for_logistic_regression(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(X, y, "classification", preprocessor, cv_folds=2, n_trials=2, timeout_s=30, random_state=42)
    lr_result = next(r for r in output.results if r.model_id == "logistic_regression")
    assert lr_result.error is None

    shap_results = compute_shap(lr_result.estimator, "logistic_regression", output.X_test, max_samples=50)
    assert shap_results["available"] is True
    assert len(shap_results["global_importance"]) > 0
