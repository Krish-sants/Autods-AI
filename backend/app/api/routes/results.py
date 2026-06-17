import json
from typing import Any

import joblib
import numpy as np
import plotly.express as px
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session
from app.database.crud import get_run
from app.storage.run_store import artifact_exists, artifact_path, load_json

router = APIRouter(tags=["results"])


def _not_ready(status: str) -> JSONResponse:
    return JSONResponse(status_code=202, content={"detail": "Run not ready", "status": status})


async def _get_run_or_404(session: AsyncSession, run_id: str):
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/summary")
async def get_summary(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or "dataset_summary" not in state:
        return _not_ready(run.status)
    return state["dataset_summary"]


@router.get("/runs/{run_id}/eda")
async def get_eda(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or "eda_results" not in state:
        return _not_ready(run.status)
    return state["eda_results"]


@router.get("/runs/{run_id}/leaderboard")
async def get_leaderboard(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or not state.get("leaderboard"):
        return _not_ready(run.status)
    return {"leaderboard": state["leaderboard"], "best_model_id": state.get("best_model_id")}


@router.get("/runs/{run_id}/metrics/{model_id}")
async def get_metrics(run_id: str, model_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or not state.get("leaderboard"):
        return _not_ready(run.status)

    # Use all_model_metrics (Phase 2) if available
    all_metrics = state.get("all_model_metrics") or {}
    if all_metrics and model_id in all_metrics:
        return all_metrics[model_id]

    # Fallback: best-model metrics from Phase 1 runs
    if "metrics" in state and model_id == state.get("best_model_id"):
        return state["metrics"]

    raise HTTPException(status_code=404, detail=f"Metrics for model '{model_id}' not available")


@router.get("/runs/{run_id}/feature-importance")
async def get_feature_importance(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    if not artifact_exists(run_id, "model.pkl"):
        return _not_ready(run.status)

    state = load_json(run_id, "state.json") or {}
    pipeline = joblib.load(artifact_path(run_id, "model.pkl"))

    preprocess = pipeline.named_steps.get("preprocess") if hasattr(pipeline, "named_steps") else None
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline

    try:
        feature_names = [str(n) for n in preprocess.get_feature_names_out()] if preprocess else []
    except Exception:
        feature_names = []

    importances: list[dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        values = None

    if values is not None and feature_names and len(values) == len(feature_names):
        order = np.argsort(values)[::-1][:25]
        importances = [{"feature": feature_names[i], "importance": float(values[i])} for i in order]

    # Fallback: derive importances from SHAP global importance when no built-in attr
    if not importances and artifact_exists(run_id, "shap_results.json"):
        shap_data = load_json(run_id, "shap_results.json") or {}
        if shap_data.get("available") and shap_data.get("global_importance"):
            importances = [
                {"feature": entry["feature"], "importance": entry["mean_abs_shap"]}
                for entry in shap_data["global_importance"]
            ]

    figure = None
    if importances:
        fig = px.bar(
            x=[i["importance"] for i in importances],
            y=[i["feature"] for i in importances],
            orientation="h",
            title=f"Feature Importance — {state.get('best_model_id', '')}",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        figure = json.loads(pio.to_json(fig))

    return {"model_id": state.get("best_model_id"), "importances": importances, "plotly_figure": figure}


@router.get("/runs/{run_id}/shap")
async def get_shap(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or "shap_results" not in state:
        return _not_ready(run.status)
    return state["shap_results"]


@router.get("/runs/{run_id}/forecast")
async def get_forecast(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or "forecast_results" not in state:
        return _not_ready(run.status)
    return state["forecast_results"]


@router.get("/runs/{run_id}/report")
async def get_report(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await _get_run_or_404(session, run_id)
    state = load_json(run_id, "state.json")
    if state is None or "report_paths" not in state:
        return _not_ready(run.status)

    md_path = artifact_path(run_id, "report.md")
    html_path = artifact_path(run_id, "report.html")
    return {
        "markdown": md_path.read_text(encoding="utf-8") if md_path.exists() else "",
        "html": html_path.read_text(encoding="utf-8") if html_path.exists() else "",
        "executive_summary_source": state.get("executive_summary_source", "template"),
    }
