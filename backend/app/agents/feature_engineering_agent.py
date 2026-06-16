import logging

from app.graph.state import PipelineState
from app.ml.features import engineer_features, save_feature_map

logger = logging.getLogger(__name__)


async def feature_engineering_agent(state: PipelineState) -> PipelineState:
    try:
        df = state["cleaned_df"]
        target_column = state.get("target_column")
        datetime_columns = state["dataset_summary"].get("datetime_columns", [])

        X, y, preprocessor, report = engineer_features(df, target_column, datetime_columns)

        state["X"] = X
        state["y"] = y
        state["preprocessor"] = preprocessor
        state["feature_engineering_report"] = report
        state["current_step"] = "feature_engineering"
        state.setdefault("steps_completed", []).append("feature_engineering")
    except Exception as exc:
        logger.exception("feature_engineering_agent failed")
        state.setdefault("errors", []).append(f"feature_engineering_agent: {exc}")
        raise
    return state
