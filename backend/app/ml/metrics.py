from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = estimator.predict(X_test)
    labels = sorted(y_test.unique().tolist())

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": [str(l) for l in labels],
    }

    if len(labels) == 2 and hasattr(estimator, "predict_proba"):
        try:
            y_proba = estimator.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_regression_metrics(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = estimator.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    n, p = len(y_test), X_test.shape[1]
    r2 = float(r2_score(y_test, y_pred))
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else None

    try:
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
    except Exception:
        mape = None

    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mape": mape,
    }


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray, silhouette: float) -> dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "silhouette_score": silhouette,
        "n_clusters": int(len(unique)),
        "cluster_sizes": {int(u): int(c) for u, c in zip(unique, counts)},
    }
