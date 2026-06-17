import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.config import get_settings
from app.database.connection import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def _cleanup_old_artifacts() -> None:
    """Delete run artifact directories older than ARTIFACT_RETENTION_DAYS."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.ARTIFACT_RETENTION_DAYS)
    runs_dir = settings.runs_dir
    removed = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            shutil.rmtree(run_dir, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Artifact cleanup: removed %d expired run director(ies).", removed)


async def _periodic_cleanup() -> None:
    """Run artifact cleanup once per day in the background."""
    while True:
        try:
            await _cleanup_old_artifacts()
        except Exception:
            logger.exception("Artifact cleanup task failed.")
        await asyncio.sleep(86_400)  # 24 hours


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info(
        "AutoDS-AI backend started (LLM provider: %s, key configured: %s)",
        settings.LLM_PROVIDER,
        bool(settings.GOOGLE_API_KEY),
    )
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.APP_NAME}


app.include_router(api_router)
