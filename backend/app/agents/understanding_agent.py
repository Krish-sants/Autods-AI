import logging

from app.agents.base import safe_llm_call
from app.config import get_settings
from app.graph.state import PipelineState
from app.ml.loaders import load_dataset
from app.ml.profiling import profile_dataset

logger = logging.getLogger(__name__)

_FALLBACK_TEMPLATE = (
    "This dataset has {n_rows} rows and {n_cols} columns ({n_numeric} numeric, {n_categorical} categorical, "
    "{n_datetime} datetime). {missing_clause}Potential target columns include: {targets}."
)


async def understanding_agent(state: PipelineState) -> PipelineState:
    settings = get_settings()
    try:
        df = load_dataset(state["dataset_path"], state["dataset_format"], max_rows=settings.MAX_ROWS)
        summary = profile_dataset(df)

        missing_clause = "The dataset contains missing values. " if summary["has_missing"] else ""
        fallback = _FALLBACK_TEMPLATE.format(
            n_rows=summary["n_rows"],
            n_cols=summary["n_cols"],
            n_numeric=len(summary["numeric_columns"]),
            n_categorical=len(summary["categorical_columns"]),
            n_datetime=len(summary["datetime_columns"]),
            missing_clause=missing_clause,
            targets=", ".join(summary["potential_targets"]) or "none obviously detected",
        )

        prompt = (
            "You are a senior data scientist. In 2-3 plain-English sentences, describe this dataset to a "
            "non-technical business stakeholder.\n\n"
            f"Rows: {summary['n_rows']}, Columns: {summary['n_cols']}\n"
            f"Numeric columns: {summary['numeric_columns']}\n"
            f"Categorical columns: {summary['categorical_columns']}\n"
            f"Datetime columns: {summary['datetime_columns']}\n"
            f"Missing values: {summary['has_missing']}, Duplicates: {summary['duplicate_count']}\n"
            f"Potential target columns: {summary['potential_targets']}\n"
        )
        narrative, used_llm = await safe_llm_call(prompt, fallback)
        summary["narrative"] = narrative

        state["df"] = df
        state["dataset_summary"] = summary
        state["narrative_llm_used"] = state.get("narrative_llm_used", False) or used_llm
        state["current_step"] = "understanding"
        state.setdefault("steps_completed", []).append("understanding")
    except Exception as exc:
        logger.exception("understanding_agent failed")
        state.setdefault("errors", []).append(f"understanding_agent: {exc}")
        raise  # fatal: nothing downstream can run without a loaded dataframe
    return state
