import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    dataset_format: Mapped[str] = mapped_column(String, nullable=False)
    raw_path: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[str] = mapped_column(String, default="uploaded")
    # uploaded -> running -> awaiting_target_confirmation -> running -> complete | failed | cancelled
    current_step: Mapped[str] = mapped_column(String, default="uploaded")

    candidate_target: Mapped[str | None] = mapped_column(String, nullable=True)
    candidate_target_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_target_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    target_column: Mapped[str | None] = mapped_column(String, nullable=True)
    problem_type: Mapped[str | None] = mapped_column(String, nullable=True)

    best_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(default=False)

    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)
