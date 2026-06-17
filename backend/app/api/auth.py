import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


async def require_app_password(x_app_password: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.APP_ACCESS_PASSWORD:
        return  # gate disabled (local dev default)
    if not x_app_password or not hmac.compare_digest(x_app_password, settings.APP_ACCESS_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid or missing access password")
