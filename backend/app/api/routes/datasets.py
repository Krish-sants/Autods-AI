from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.connection import get_session
from app.database.crud import create_run
from app.ml.loaders import UnsupportedFormatError, detect_format, download_from_url
from app.schemas.api_models import FromUrlRequest, UploadResponse
from app.storage.run_store import run_dir

router = APIRouter(tags=["datasets"])
settings = get_settings()


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile, session: AsyncSession = Depends(get_session)) -> UploadResponse:
    try:
        fmt = detect_format(file.filename, content_type=file.content_type)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB limit")
        chunks.append(chunk)
    data = b"".join(chunks)

    run = await create_run(session, filename=file.filename, dataset_format=fmt, raw_path="")

    dest_dir = run_dir(run.id)
    raw_path = dest_dir / f"raw_{file.filename}"
    raw_path.write_bytes(data)

    run.raw_path = str(raw_path)
    session.add(run)
    await session.commit()

    return UploadResponse(run_id=run.id, filename=file.filename, format=fmt, size_bytes=total, status=run.status)


@router.post("/datasets/from-url", response_model=UploadResponse)
async def upload_from_url(payload: FromUrlRequest, session: AsyncSession = Depends(get_session)) -> UploadResponse:
    run = await create_run(session, filename="pending", dataset_format="unknown", raw_path="")
    dest_dir = run_dir(run.id)

    try:
        local_path, fmt = download_from_url(payload.url, dest_dir)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download dataset: {exc}") from exc

    size_bytes = local_path.stat().st_size
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB limit")

    run.filename = local_path.name
    run.dataset_format = fmt
    run.raw_path = str(local_path)
    session.add(run)
    await session.commit()

    return UploadResponse(run_id=run.id, filename=local_path.name, format=fmt, size_bytes=size_bytes, status=run.status)
