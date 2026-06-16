from app.ml.features import engineer_features
from app.ml.registry import get_registry
from app.ml.train import train_and_tune_all


def test_registry_has_five_models_per_problem_type():
    classification_specs = get_registry("classification")
    regression_specs = get_registry("regression")
    assert len(classification_specs) == 5
    assert len(regression_specs) == 5


def test_each_classification_model_fits_without_error(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(X, y, "classification", preprocessor, cv_folds=2, n_iter=2, random_state=42)

    assert len(output.results) == 5
    failures = [r for r in output.results if r.error is not None]
    assert not failures, f"Some models failed to train: {failures}"
    for result in output.results:
        assert result.cv_score is not None
        assert result.cv_score > float("-inf")
        assert result.estimator is not None


def test_each_regression_model_fits_without_error(synthetic_regression_df):
    df = synthetic_regression_df
    X, y, preprocessor, _ = engineer_features(df, target_column="price", datetime_columns=[])

    output = train_and_tune_all(X, y, "regression", preprocessor, cv_folds=2, n_iter=2, random_state=42)

    assert len(output.results) == 5
    failures = [r for r in output.results if r.error is not None]
    assert not failures, f"Some models failed to train: {failures}"
    for result in output.results:
        assert result.cv_score is not None
        assert result.estimator is not None


def test_train_test_split_is_proper_holdout(synthetic_classification_df):
    df = synthetic_classification_df.drop(columns=["customer_id"])
    X, y, preprocessor, _ = engineer_features(df, target_column="churn", datetime_columns=["signup_date"])

    output = train_and_tune_all(X, y, "classification", preprocessor, cv_folds=2, n_iter=2, random_state=42)

    assert len(output.X_train) + len(output.X_test) == len(X)
    assert set(output.X_train.index).isdisjoint(set(output.X_test.index))
