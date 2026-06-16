import logging

from app.graph.state import PipelineState
from app.ml.eda import generate_eda

logger = logging.getLogger(__name__)


async def eda_agent(state: PipelineState) -> PipelineState:
    try:
        df = state["df"]
        summary = state["dataset_summary"]
        eda_results = generate_eda(df, summary)

        state["eda_results"] = eda_results
        state["current_step"] = "eda"
        state.setdefault("steps_completed", []).append("eda")
    except Exception as exc:
        logger.exception("eda_agent failed")
        state.setdefault("errors", []).append(f"eda_agent: {exc}")
        state["eda_results"] = {"figures": {}, "descriptive_stats": {}, "outlier_flags": {}, "notable_findings": []}
    return state
