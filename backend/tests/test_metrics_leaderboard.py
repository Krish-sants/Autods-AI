import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.ml.leaderboard import rank_models
from app.ml.metrics import compute_classification_metrics, compute_regression_metrics
from app.ml.train import ModelResult


def test_compute_classification_metrics_known_values():
    X = pd.DataFrame({"x": [0, 0, 1, 1]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)

    metrics = compute_classification_metrics(model, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_regression_metrics_known_values():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    model = LinearRegression().fit(X, y)

    metrics = compute_regression_metrics(model, X, y)
    assert metrics["mae"] < 1e-6
    assert metrics["rmse"] < 1e-6
    assert abs(metrics["r2"] - 1.0) < 1e-6


def _fake_result(model_id, score):
    return ModelResult(model_id=model_id, display_name=model_id, estimator=object(), cv_score=score, fit_time_s=1.0)


def test_rank_models_orders_by_score_descending():
    results = [_fake_result("a", 0.7), _fake_result("b", 0.9), _fake_result("c", 0.5)]
    rows, best_model_id = rank_models(results)

    assert [r["model_id"] for r in rows] == ["b", "a", "c"]
    assert best_model_id == "b"
    assert rows[0]["rank"] == 1


def test_rank_models_puts_failed_models_last_without_rank():
    ok = _fake_result("good", 0.8)
    failed = ModelResult(model_id="bad", display_name="bad", estimator=None, cv_score=float("-inf"), fit_time_s=0.1, error="boom")
    rows, best_model_id = rank_models([ok, failed])

    assert best_model_id == "good"
    assert rows[-1]["model_id"] == "bad"
    assert rows[-1]["rank"] is None
