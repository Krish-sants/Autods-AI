from fastapi import APIRouter, Depends

from app.api.auth import require_app_password
from app.api.routes import datasets, downloads, pipeline, results, runs

api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_app_password)])
api_router.include_router(datasets.router)
api_router.include_router(pipeline.router)
api_router.include_router(results.router)
api_router.include_router(downloads.router)
api_router.include_router(runs.router)
