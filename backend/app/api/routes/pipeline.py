import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import async_session, get_session
from app.database.crud import get_run, update_run
from app.graph.workflow import run_post_target, run_pre_target
from app.ml.problem_type import infer_problem_type
from app.schemas.api_models import ConfirmTargetRequest, ConfirmTargetResponse, StartRunResponse, StatusResponse
from app.storage.run_store import artifact_path

import pandas as pd

router = APIRouter(tags=["pipeline"])
logger = logging.getLogger(__name__)


async def _run_graph_b(run_id: str, target_column: str | None) -> None:
    try:
        final_state = await run_post_target(run_id, target_column)
        async with async_session() as session:
            await update_run(
                session,
                run_id,
                status="complete",
                current_step="complete",
                best_model_id=final_state.get("best_model_id"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph B failed for run %s", run_id)
        async with async_session() as session:
            await update_run(session, run_id, status="failed", error_message=str(exc))


@router.post("/runs/{run_id}/start", response_model=StartRunResponse)
async def start_run(
    run_id: str, background_tasks: BackgroundTasks, session: AsyncSession = Depends(get_session)
) -> StartRunResponse:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in ("uploaded", "failed"):
        raise HTTPException(status_code=409, detail=f"Run is already in status '{run.status}'")

    await update_run(session, run_id, status="running", current_step="queued")

    async def _task() -> None:
        await _run_graph_a(run_id, run.raw_path, run.dataset_format)

    background_tasks.add_task(_task)
    return StartRunResponse(run_id=run_id, status="running", message="Pipeline started")


async def _run_graph_a(run_id: str, raw_path: str, dataset_format: str) -> None:
    try:
        final_state = await run_pre_target(run_id, raw_path, dataset_format)
        async with async_session() as session:
            await update_run(
                session,
                run_id,
                status="awaiting_target_confirmation",
                current_step=final_state.get("current_step", "target_detection"),
                candidate_target=final_state.get("candidate_target"),
                candidate_target_confidence=final_state.get("candidate_target_confidence"),
                candidate_target_reasoning=final_state.get("candidate_target_reasoning"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph A failed for run %s", run_id)
        async with async_session() as session:
            await update_run(session, run_id, status="failed", error_message=str(exc))


@router.post("/runs/{run_id}/confirm-target", response_model=ConfirmTargetResponse)
async def confirm_target(
    run_id: str,
    payload: ConfirmTargetRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> ConfirmTargetResponse:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "awaiting_target_confirmation":
        raise HTTPException(status_code=409, detail=f"Run is in status '{run.status}', not awaiting confirmation")

    cleaned_path = artifact_path(run_id, "cleaned.parquet")
    if not cleaned_path.exists():
        raise HTTPException(status_code=500, detail="Cleaned dataset artifact missing; cannot confirm target")

    target_column = payload.target_column or None
    if target_column:
        cleaned_df = pd.read_parquet(cleaned_path)
        if target_column not in cleaned_df.columns:
            raise HTTPException(status_code=422, detail=f"Column '{target_column}' not found in dataset")
        problem_type = infer_problem_type(cleaned_df, target_column)
    else:
        problem_type = "clustering"

    await update_run(
        session, run_id, status="running", current_step="feature_engineering",
        target_column=target_column, problem_type=problem_type,
    )

    background_tasks.add_task(_run_graph_b, run_id, target_column)

    return ConfirmTargetResponse(run_id=run_id, status="running", target_column=target_column or "", problem_type=problem_type)


@router.get("/runs/{run_id}/status", response_model=StatusResponse)
async def get_run_status(run_id: str, session: AsyncSession = Depends(get_session)) -> StatusResponse:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StatusResponse(
        run_id=run.id,
        status=run.status,
        current_step=run.current_step,
        candidate_target=run.candidate_target,
        candidate_target_confidence=run.candidate_target_confidence,
        candidate_target_reasoning=run.candidate_target_reasoning,
        target_column=run.target_column,
        problem_type=run.problem_type,
        best_model_id=run.best_model_id,
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    await update_run(session, run_id, cancel_requested=True, status="cancelled")
    return {"run_id": run_id, "status": "cancelled"}
