import logging

from app.agents.base import raise_if_cancelled
from app.config import get_settings
from app.graph.state import PipelineState
from app.ml.clustering import run_kmeans_scan
from app.ml.train import train_and_tune_all

logger = logging.getLogger(__name__)


async def training_agent(state: PipelineState) -> PipelineState:
    await raise_if_cancelled(state["run_id"])
    settings = get_settings()
    problem_type = state["problem_type"]

    try:
        if problem_type == "clustering":
            result = run_kmeans_scan(state["X"], state["preprocessor"], random_state=settings.RANDOM_STATE)
            state["clustering_result"] = result
        else:
            scoring = (
                settings.SCORING_METRIC_CLASSIFICATION
                if problem_type == "classification"
                else settings.SCORING_METRIC_REGRESSION
            )
            training_output = train_and_tune_all(
                state["X"],
                state["y"],
                problem_type,
                state["preprocessor"],
                cv_folds=settings.CV_FOLDS,
                n_trials=settings.OPTUNA_N_TRIALS,
                timeout_s=settings.OPTUNA_TIMEOUT_S,
                random_state=settings.RANDOM_STATE,
                scoring=scoring,
            )
            state["training_output"] = training_output

        state["current_step"] = "training"
        state.setdefault("steps_completed", []).append("training")
    except Exception as exc:
        logger.exception("training_agent failed")
        state.setdefault("errors", []).append(f"training_agent: {exc}")
        raise
    return state
