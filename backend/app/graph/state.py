from typing import Any, TypedDict

import pandas as pd

JSON_STATE_FIELDS = [
    "run_id",
    "dataset_path",
    "dataset_format",
    "dataset_summary",
    "eda_results",
    "cleaning_report",
    "candidate_target",
    "candidate_target_confidence",
    "candidate_target_reasoning",
    "target_column",
    "problem_type",
    "feature_engineering_report",
    "feature_map_path",
    "leaderboard",
    "best_model_id",
    "best_model_path",
    "metrics",
    "all_model_metrics",
    "shap_results",
    "forecast_results",
    "report_paths",
    "narrative_llm_used",
    "executive_summary_source",
    "current_step",
    "steps_completed",
    "errors",
]


class PipelineState(TypedDict, total=False):
    run_id: str
    dataset_path: str
    dataset_format: str

    dataset_summary: dict[str, Any]
    eda_results: dict[str, Any]
    cleaning_report: dict[str, Any]

    candidate_target: str | None
    candidate_target_confidence: float
    candidate_target_reasoning: str

    target_column: str | None
    problem_type: str | None

    feature_engineering_report: dict[str, Any]
    feature_map_path: str | None

    leaderboard: list[dict[str, Any]]
    best_model_id: str | None
    best_model_path: str | None
    metrics: dict[str, Any]
    all_model_metrics: dict[str, dict[str, Any]]
    shap_results: dict[str, Any]
    forecast_results: dict[str, Any]
    report_paths: dict[str, str]

    narrative_llm_used: bool
    executive_summary_source: str
    current_step: str
    steps_completed: list[str]
    errors: list[str]

    # --- transient, in-process-only fields (never persisted to state.json) ---
    # NOTE: LangGraph only forwards keys declared in this TypedDict between node
    # hops — any field an agent sets that isn't listed here is silently dropped
    # before the next node runs. Every key any agent reads/writes must be here.
    # Fields in this block are intentionally excluded from JSON_STATE_FIELDS above
    # (they're not JSON-serializable) — don't add a field here AND expect it to
    # survive into state.json; if it needs to be read back via the API, it
    # belongs in JSON_STATE_FIELDS instead (see executive_summary_source above,
    # which was wrongly placed in this block until that bug was caught).
    df: pd.DataFrame
    cleaned_df: pd.DataFrame
    X: Any
    y: Any
    preprocessor: Any
    training_output: Any
    clustering_result: Any
    best_pipeline: Any
    X_test: Any
    y_test: Any
    forecast_model: Any


def to_json_safe(state: PipelineState) -> dict[str, Any]:
    """Project only JSON-serializable fields, dropping in-memory objects like DataFrames/estimators."""
    return {key: state.get(key) for key in JSON_STATE_FIELDS if key in state}


def new_state(run_id: str, dataset_path: str, dataset_format: str) -> PipelineState:
    return PipelineState(
        run_id=run_id,
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        current_step="queued",
        steps_completed=[],
        errors=[],
        narrative_llm_used=False,
    )
