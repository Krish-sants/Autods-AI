import logging

import joblib

from app.graph.state import PipelineState
from app.ml.features import save_feature_map
from app.ml.leaderboard import rank_models
from app.ml.metrics import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_regression_metrics,
)
from app.storage.run_store import artifact_path, save_json

logger = logging.getLogger(__name__)


async def evaluation_agent(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    problem_type = state["problem_type"]

    try:
        if problem_type == "clustering":
            result = state["clustering_result"]
            metrics = compute_clustering_metrics(
                result.transformed_X, result.labels, result.silhouette_scores.get(result.best_k, 0.0)
            )
            leaderboard = [
                {
                    "rank": 1,
                    "model_id": "kmeans",
                    "display_name": f"KMeans (k={result.best_k})",
                    "cv_score": result.silhouette_scores.get(result.best_k, 0.0),
                    "fit_time_s": None,
                    "best_params": {"n_clusters": result.best_k},
                }
            ]
            best_model_id = "kmeans"
            model_path = artifact_path(run_id, "model.pkl")
            joblib.dump(result.estimator, model_path)
            preprocessor_path = artifact_path(run_id, "pipeline.pkl")
            joblib.dump(result.fitted_preprocessor, preprocessor_path)

            state["best_pipeline"] = result.estimator
            metrics["cluster_profile"] = result.cluster_profile
        else:
            training_output = state["training_output"]
            leaderboard, best_model_id = rank_models(training_output.results)

            if best_model_id is None:
                raise RuntimeError("All models failed to train; cannot evaluate.")

            best_result = next(r for r in training_output.results if r.model_id == best_model_id)
            best_pipeline = best_result.estimator

            if problem_type == "classification":
                metrics = compute_classification_metrics(
                    best_pipeline, training_output.X_test, training_output.y_test
                )
            else:
                metrics = compute_regression_metrics(best_pipeline, training_output.X_test, training_output.y_test)

            model_path = artifact_path(run_id, "model.pkl")
            joblib.dump(best_pipeline, model_path)

            state["best_pipeline"] = best_pipeline
            state["X_test"] = training_output.X_test
            state["y_test"] = training_output.y_test

        feature_map = save_feature_map(state["preprocessor"], state["feature_engineering_report"])
        feature_map_path = save_json(run_id, "feature_map.json", feature_map)

        state["leaderboard"] = leaderboard
        state["best_model_id"] = best_model_id
        state["best_model_path"] = str(model_path)
        state["feature_map_path"] = str(feature_map_path)
        state["metrics"] = metrics

        state["current_step"] = "evaluation"
        state.setdefault("steps_completed", []).append("evaluation")
    except Exception as exc:
        logger.exception("evaluation_agent failed")
        state.setdefault("errors", []).append(f"evaluation_agent: {exc}")
        raise
    return state
