from datetime import datetime
from typing import Any

from pydantic import BaseModel


class UploadResponse(BaseModel):
    run_id: str
    filename: str
    format: str
    size_bytes: int
    status: str


class FromUrlRequest(BaseModel):
    url: str


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class ConfirmTargetRequest(BaseModel):
    target_column: str


class ConfirmTargetResponse(BaseModel):
    run_id: str
    status: str
    target_column: str
    problem_type: str


class StatusResponse(BaseModel):
    run_id: str
    status: str
    current_step: str
    candidate_target: str | None = None
    candidate_target_confidence: float | None = None
    candidate_target_reasoning: str | None = None
    target_column: str | None = None
    problem_type: str | None = None
    best_model_id: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class RunListItem(BaseModel):
    run_id: str
    filename: str
    status: str
    current_step: str
    created_at: datetime


class RunDetail(StatusResponse):
    filename: str
    dataset_format: str


class NotReadyResponse(BaseModel):
    detail: str
    status: str


class GenericResultResponse(BaseModel):
    run_id: str
    data: dict[str, Any]
