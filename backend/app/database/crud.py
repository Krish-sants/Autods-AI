from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Run


async def create_run(session: AsyncSession, *, filename: str, dataset_format: str, raw_path: str) -> Run:
    run = Run(filename=filename, dataset_format=dataset_format, raw_path=raw_path)
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


async def get_run(session: AsyncSession, run_id: str) -> Run | None:
    return await session.get(Run, run_id)


async def list_runs(session: AsyncSession, limit: int = 50, offset: int = 0) -> list[Run]:
    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_run(session: AsyncSession, run_id: str, **fields) -> Run | None:
    run = await session.get(Run, run_id)
    if run is None:
        return None
    for key, value in fields.items():
        setattr(run, key, value)
    await session.commit()
    await session.refresh(run)
    return run


async def update_run_step(session: AsyncSession, run_id: str, *, status: str, current_step: str) -> Run | None:
    return await update_run(session, run_id, status=status, current_step=current_step)


async def update_run_failed(session: AsyncSession, run_id: str, *, error_message: str) -> Run | None:
    return await update_run(session, run_id, status="failed", error_message=error_message)
