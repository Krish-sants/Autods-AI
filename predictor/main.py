"""AutoDS-AI Predict Microservice.

Loads a trained pipeline artifact (model.pkl) for a given run and serves
predictions via a simple REST API. Mount the same /app/data volume as the
backend (read-only) so it can access run artifacts.

Endpoints:
  GET  /health                        — liveness check
  GET  /runs/{run_id}/schema          — expected input columns
  POST /runs/{run_id}/predict         — returns predictions for a JSON payload
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
app = FastAPI(title="AutoDS-AI Predictor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    return DATA_DIR / "runs" / run_id


def _model_path(run_id: str) -> Path:
    return _run_dir(run_id) / "model.pkl"


def _state_path(run_id: str) -> Path:
    return _run_dir(run_id) / "state.json"


def _feature_map_path(run_id: str) -> Path:
    return _run_dir(run_id) / "feature_map.json"


@lru_cache(maxsize=32)
def _load_pipeline(run_id: str):
    """Cache loaded pipelines in-process — evicted on restart."""
    path = _model_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"model.pkl not found for run {run_id!r}")
    return joblib.load(path)


def _load_state(run_id: str) -> dict:
    path = _state_path(run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_map(run_id: str) -> list[str]:
    path = _feature_map_path(run_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    # feature_map.json stores input columns broken out by type
    numeric = data.get("numeric_features", [])
    low_card = data.get("low_cardinality_categorical", [])
    high_card = data.get("high_cardinality_categorical", [])
    combined = numeric + low_card + high_card
    return combined if combined else data.get("output_features", [])


def _serialize(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]
    return_probabilities: bool = False


class PredictResponse(BaseModel):
    run_id: str
    model_id: str
    problem_type: str
    predictions: list[Any]
    probabilities: list[list[float]] | None = None
    n_rows: int


class SchemaResponse(BaseModel):
    run_id: str
    model_id: str
    problem_type: str
    target_column: str | None
    input_features: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "autods-predictor"}


@app.get("/runs/{run_id}/schema", response_model=SchemaResponse)
async def get_schema(run_id: str) -> SchemaResponse:
    if not _run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if not _model_path(run_id).exists():
        raise HTTPException(status_code=404, detail=f"model.pkl not found for run '{run_id}'")

    state = _load_state(run_id)
    features = _load_feature_map(run_id)

    return SchemaResponse(
        run_id=run_id,
        model_id=state.get("best_model_id", "unknown"),
        problem_type=state.get("problem_type", "unknown"),
        target_column=state.get("target_column"),
        input_features=features,
    )


@app.post("/runs/{run_id}/predict", response_model=PredictResponse)
async def predict(run_id: str, body: PredictRequest) -> PredictResponse:
    if not _run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    try:
        pipeline = _load_pipeline(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"model.pkl not found for run '{run_id}'")

    state = _load_state(run_id)
    problem_type = state.get("problem_type", "unknown")

    if not body.rows:
        raise HTTPException(status_code=422, detail="'rows' must contain at least one record")

    try:
        df = pd.DataFrame(body.rows)
        raw_preds = pipeline.predict(df)
        predictions = [_serialize(p) for p in raw_preds]

        probabilities = None
        if body.return_probabilities and hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(df)
                probabilities = [[float(v) for v in row] for row in proba]
            except Exception:
                pass
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        run_id=run_id,
        model_id=state.get("best_model_id", "unknown"),
        problem_type=problem_type,
        predictions=predictions,
        probabilities=probabilities,
        n_rows=len(body.rows),
    )
