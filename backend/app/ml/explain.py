from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

TREE_MODEL_IDS = {"random_forest", "extra_trees", "xgboost", "lightgbm", "gradient_boosting"}
LINEAR_MODEL_IDS = {"logistic_regression", "ridge"}
# Models with no native feature importances — use KernelExplainer on predict_proba/predict
KERNEL_MODEL_IDS = {"svm", "knn", "svr"}


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None:
        return []
    try:
        return [str(n) for n in preprocess.get_feature_names_out()]
    except Exception:
        return []


def compute_shap(
    pipeline: Pipeline, model_id: str, X_sample: pd.DataFrame, *, max_samples: int = 200
) -> dict[str, Any]:
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(n=max_samples, random_state=42)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocess.transform(X_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = _transformed_feature_names(pipeline) or [f"feature_{i}" for i in range(transformed.shape[1])]

    try:
        if model_id in TREE_MODEL_IDS:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
        elif model_id in LINEAR_MODEL_IDS:
            explainer = shap.LinearExplainer(model, transformed)
            shap_values = explainer.shap_values(transformed)
        elif model_id in KERNEL_MODEL_IDS:
            # KernelExplainer works on any callable; use k-means background for speed
            n_bg = min(10, transformed.shape[0])
            background = shap.kmeans(transformed, n_bg)
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            # Limit to small sample for KernelExplainer (it's O(n*background))
            kernel_sample = transformed[:min(50, transformed.shape[0])]
            shap_values = explainer.shap_values(kernel_sample, nsamples=100, silent=True)
            transformed = kernel_sample  # align local_examples to the subset used
        else:
            explainer = shap.Explainer(model, transformed)
            shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            # multi-class / KernelExplainer with predict_proba returns one array per class
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        global_importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(global_importance)[::-1]

        global_ranked = [
            {"feature": feature_names[i], "mean_abs_shap": float(global_importance[i])} for i in order[:25]
        ]

        n_local = min(5, transformed.shape[0])
        local_examples = []
        base_value = getattr(explainer, "expected_value", None)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).flatten()[-1])
        for row_idx in range(n_local):
            row_shap = shap_values[row_idx]
            row_order = np.argsort(np.abs(row_shap))[::-1][:10]
            local_examples.append(
                {
                    "row_index": int(row_idx),
                    "base_value": float(base_value) if base_value is not None else None,
                    "contributions": [
                        {"feature": feature_names[i], "shap_value": float(row_shap[i])} for i in row_order
                    ],
                }
            )

        return {
            "available": True,
            "global_importance": global_ranked,
            "local_examples": local_examples,
        }
    except Exception as exc:  # noqa: BLE001 - explainability must never crash the pipeline
        return {"available": False, "error": str(exc), "global_importance": [], "local_examples": []}
