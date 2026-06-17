from app.ml.features import engineer_features
from app.ml.registry import get_registry
from app.ml.train import train_and_tune_all

EXPECTED_CLASSIFICATION_MODELS = 8  # LR, RF, ET, GBM, XGB, LGBM, SVM, KNN
EXPECTED_REGRESSION_MODELS = 8      # Ridge, RF, ET, GBM, XGB, LGBM, SVR, KNN


def test_registry_has_expected_models_per_problem_type():
    classification_specs = get_registry("classification")
    regression_specs = get_registry("regression")
    assert len(classification_specs) == EXPECTED_CLASSIFICATION_MODELS
    assert len(regression_specs) == EXPECTED_REGRESSION_MODELS


def test_each_classification_model_has_optuna_space():
    for spec in get_registry("classification"):
        assert spec.optuna_space is not None, f"{spec.model_id} missing optuna_space"


def test_each_regression_model_has_optuna_space():
    for spec in get_registry("regression"):
        assert spec.optuna_space is not None, f"{spec.model_id} missing optuna_space"


def test_each_classification_model_fits_without_error(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(
        X, y, "classification", preprocessor, cv_folds=2, n_trials=2, timeout_s=30, random_state=42
    )

    assert len(output.results) == EXPECTED_CLASSIFICATION_MODELS
    failures = [r for r in output.results if r.error is not None]
    assert not failures, f"Some models failed to train: {[f.model_id + ': ' + str(f.error) for f in failures]}"
    for result in output.results:
        assert result.cv_score is not None
        assert result.cv_score > float("-inf")
        assert result.estimator is not None


def test_each_regression_model_fits_without_error(synthetic_regression_df):
    df = synthetic_regression_df
    X, y, preprocessor, _ = engineer_features(df, target_column="price", datetime_columns=[])

    output = train_and_tune_all(
        X, y, "regression", preprocessor, cv_folds=2, n_trials=2, timeout_s=30, random_state=42
    )

    assert len(output.results) == EXPECTED_REGRESSION_MODELS
    failures = [r for r in output.results if r.error is not None]
    assert not failures, f"Some models failed to train: {[f.model_id + ': ' + str(f.error) for f in failures]}"
    for result in output.results:
        assert result.cv_score is not None
        assert result.estimator is not None


def test_train_test_split_is_proper_holdout(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(
        X, y, "classification", preprocessor, cv_folds=2, n_trials=2, timeout_s=30, random_state=42
    )

    assert len(output.X_train) + len(output.X_test) == len(X)
    assert set(output.X_train.index).isdisjoint(set(output.X_test.index))
