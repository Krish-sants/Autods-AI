from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session
from app.database.crud import get_run, list_runs
from app.schemas.api_models import RunDetail, RunListItem

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunListItem])
async def get_runs(
    limit: int = Query(50, le=200), offset: int = Query(0, ge=0), session: AsyncSession = Depends(get_session)
) -> list[RunListItem]:
    runs = await list_runs(session, limit=limit, offset=offset)
    return [
        RunListItem(run_id=r.id, filename=r.filename, status=r.status, current_step=r.current_step, created_at=r.created_at)
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run_detail(run_id: str, session: AsyncSession = Depends(get_session)) -> RunDetail:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetail(
        run_id=run.id,
        filename=run.filename,
        dataset_format=run.dataset_format,
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
