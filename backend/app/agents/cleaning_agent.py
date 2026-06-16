import logging

from app.graph.state import PipelineState
from app.ml.cleaning import clean_dataset

logger = logging.getLogger(__name__)


async def cleaning_agent(state: PipelineState) -> PipelineState:
    try:
        df = state["df"]
        summary = state["dataset_summary"]
        cleaned_df, report = clean_dataset(df, summary)

        state["cleaned_df"] = cleaned_df
        state["cleaning_report"] = report
        state["current_step"] = "cleaning"
        state.setdefault("steps_completed", []).append("cleaning")
    except Exception as exc:
        logger.exception("cleaning_agent failed")
        state.setdefault("errors", []).append(f"cleaning_agent: {exc}")
        # Degrade gracefully: continue with the uncleaned dataframe rather than aborting the run.
        state["cleaned_df"] = state["df"]
        state["cleaning_report"] = {
            "shape_before": list(state["df"].shape),
            "shape_after": list(state["df"].shape),
            "actions": [],
        }
    return state
