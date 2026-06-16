from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringOutput:
    best_k: int
    labels: np.ndarray
    silhouette_scores: dict[int, float]
    cluster_profile: list[dict[str, Any]]
    estimator: Any
    fitted_preprocessor: ColumnTransformer
    transformed_X: np.ndarray


def run_kmeans_scan(
    X: pd.DataFrame, preprocessor: ColumnTransformer, *, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> ClusteringOutput:
    transformed = preprocessor.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    n_samples = transformed.shape[0]
    k_max = min(k_max, max(k_min, n_samples - 1))

    silhouette_scores: dict[int, float] = {}
    best_k, best_score, best_model, best_labels = k_min, -1.0, None, None

    for k in range(k_min, k_max + 1):
        if k >= n_samples:
            break
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(transformed)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(transformed, labels))
        silhouette_scores[k] = score
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, model, labels

    if best_model is None:
        best_model = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
        best_labels = best_model.fit_predict(transformed)
        best_k = k_min

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    profile = []
    if numeric_cols:
        df_with_labels = X[numeric_cols].copy()
        df_with_labels["__cluster__"] = best_labels
        for cluster_id, group in df_with_labels.groupby("__cluster__"):
            profile.append(
                {
                    "cluster": int(cluster_id),
                    "size": int(len(group)),
                    "mean_values": group[numeric_cols].mean().round(3).to_dict(),
                }
            )

    return ClusteringOutput(
        best_k=best_k,
        labels=best_labels,
        silhouette_scores=silhouette_scores,
        cluster_profile=profile,
        estimator=best_model,
        fitted_preprocessor=preprocessor,
        transformed_X=transformed,
    )
