import logging

from app.graph.state import PipelineState
from app.ml.target_detection import guess_target_column

logger = logging.getLogger(__name__)


async def target_agent(state: PipelineState) -> PipelineState:
    try:
        df = state["cleaned_df"]
        column, confidence, reasoning = guess_target_column(df)

        state["candidate_target"] = column
        state["candidate_target_confidence"] = confidence
        state["candidate_target_reasoning"] = reasoning
        state["current_step"] = "target_detection"
        state.setdefault("steps_completed", []).append("target_detection")
    except Exception as exc:
        logger.exception("target_agent failed")
        state.setdefault("errors", []).append(f"target_agent: {exc}")
        state["candidate_target"] = None
        state["candidate_target_confidence"] = 0.0
        state["candidate_target_reasoning"] = "Target detection failed."
    return state
