from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session
from app.database.crud import get_run
from app.storage.run_store import artifact_path

router = APIRouter(tags=["downloads"])

_ARTIFACT_FILENAMES = {
    "model": "model.pkl",
    "pipeline": "pipeline.pkl",
    "report-html": "report.html",
    "report-md": "report.md",
}


@router.get("/runs/{run_id}/download/{artifact}")
async def download_artifact(run_id: str, artifact: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    run = await get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if artifact == "raw-dataset":
        path = run.raw_path
        filename = run.filename
    else:
        name = _ARTIFACT_FILENAMES.get(artifact)
        if name is None:
            raise HTTPException(status_code=404, detail=f"Unknown artifact '{artifact}'")
        path = str(artifact_path(run_id, name))
        filename = name

    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact}' not yet available for this run")

    return FileResponse(path, filename=filename)
