import logging

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents.cleaning_agent import cleaning_agent
from app.agents.eda_agent import eda_agent
from app.agents.evaluation_agent import evaluation_agent
from app.agents.explainability_agent import explainability_agent
from app.agents.feature_engineering_agent import feature_engineering_agent
from app.agents.forecasting_agent import forecasting_agent
from app.agents.report_agent import report_agent
from app.agents.target_agent import target_agent
from app.agents.training_agent import training_agent
from app.agents.understanding_agent import understanding_agent
from app.graph.state import PipelineState, new_state, to_json_safe
from app.ml.problem_type import FORECASTING, infer_problem_type
from app.storage.run_store import artifact_path, load_json, save_json

logger = logging.getLogger(__name__)


def _route_post_target(state: PipelineState) -> str:
    """After entering post-target graph, route to forecasting or supervised pipeline."""
    if state.get("problem_type") == FORECASTING:
        return "forecasting"
    return "feature_engineering"


def _route_after_forecasting(state: PipelineState) -> str:
    """After forecasting, skip SHAP/FE (not applicable) and go straight to report."""
    return "report"


def build_pre_target_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("understanding", understanding_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("cleaning", cleaning_agent)
    graph.add_node("target_detection", target_agent)

    graph.set_entry_point("understanding")
    graph.add_edge("understanding", "eda")
    graph.add_edge("eda", "cleaning")
    graph.add_edge("cleaning", "target_detection")
    graph.add_edge("target_detection", END)
    return graph.compile()


def build_post_target_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("training", training_agent)
    graph.add_node("evaluation", evaluation_agent)
    graph.add_node("explainability", explainability_agent)
    graph.add_node("forecasting", forecasting_agent)
    graph.add_node("report", report_agent)

    # Conditional entry: forecasting datasets skip the supervised pipeline
    graph.add_conditional_edges(
        START,
        _route_post_target,
        {"forecasting": "forecasting", "feature_engineering": "feature_engineering"},
    )

    # Supervised path
    graph.add_edge("feature_engineering", "training")
    graph.add_edge("training", "evaluation")
    graph.add_edge("evaluation", "explainability")
    graph.add_edge("explainability", "report")

    # Forecasting path — skip FE/training/evaluation/SHAP
    graph.add_edge("forecasting", "report")

    graph.add_edge("report", END)
    return graph.compile()


_pre_target_graph = None
_post_target_graph = None


def get_pre_target_graph():
    global _pre_target_graph
    if _pre_target_graph is None:
        _pre_target_graph = build_pre_target_graph()
    return _pre_target_graph


def get_post_target_graph():
    global _post_target_graph
    if _post_target_graph is None:
        _post_target_graph = build_post_target_graph()
    return _post_target_graph


async def run_pre_target(run_id: str, dataset_path: str, dataset_format: str) -> PipelineState:
    """Graph A: understanding -> eda -> cleaning -> target_detection."""
    state = new_state(run_id, dataset_path, dataset_format)
    graph = get_pre_target_graph()
    final_state = await graph.ainvoke(state)

    cleaned_df = final_state.get("cleaned_df")
    if cleaned_df is not None:
        cleaned_df.to_parquet(artifact_path(run_id, "cleaned.parquet"))

    save_json(run_id, "state.json", to_json_safe(final_state))
    return final_state


async def run_post_target(run_id: str, target_column: str | None) -> PipelineState:
    """Graph B: routes to forecasting or supervised pipeline depending on problem_type."""
    persisted = load_json(run_id, "state.json") or {}
    cleaned_df = pd.read_parquet(artifact_path(run_id, "cleaned.parquet"))

    problem_type = infer_problem_type(cleaned_df, target_column)

    state: PipelineState = {
        **persisted,
        "run_id": run_id,
        "cleaned_df": cleaned_df,
        "target_column": target_column,
        "problem_type": problem_type,
    }

    graph = get_post_target_graph()
    final_state = await graph.ainvoke(state)

    save_json(run_id, "state.json", to_json_safe(final_state))
    return final_state
