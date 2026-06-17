import logging

import joblib

from app.config import get_settings
from app.graph.state import PipelineState
from app.ml.forecasting import run_forecast, serialize_forecast_results
from app.storage.run_store import artifact_path

logger = logging.getLogger(__name__)


async def forecasting_agent(state: PipelineState) -> PipelineState:
    settings = get_settings()
    run_id = state["run_id"]
    target_column = state["target_column"]

    try:
        cleaned_df = state["cleaned_df"]
        output = run_forecast(cleaned_df, target_column, periods=settings.FORECAST_PERIODS)

        # Persist the fitted Prophet model
        model_path = artifact_path(run_id, "model.pkl")
        joblib.dump(output.model, model_path)

        state["forecast_model"] = output.model
        state["forecast_results"] = serialize_forecast_results(output)
        state["best_model_id"] = "prophet"
        state["best_model_path"] = str(model_path)
        state["leaderboard"] = [
            {
                "rank": 1,
                "model_id": "prophet",
                "display_name": "Prophet",
                "cv_score": round(1 - output.metrics["mape"] / 100, 4),
                "fit_time_s": None,
                "best_params": {"freq": output.freq, "periods": settings.FORECAST_PERIODS},
            }
        ]
        state["metrics"] = output.metrics
        state["all_model_metrics"] = {"prophet": output.metrics}

        state["current_step"] = "forecasting"
        state.setdefault("steps_completed", []).append("forecasting")
        logger.info("forecasting_agent: Prophet fit complete (MAE=%.3f)", output.metrics["mae"])
    except Exception as exc:
        logger.exception("forecasting_agent failed")
        state.setdefault("errors", []).append(f"forecasting_agent: {exc}")
        raise
    return state
