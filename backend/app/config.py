import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    APP_NAME: str = "AutoDS-AI"
    ENV: str = "development"

    # If set, all /api/v1/* routes require a matching X-App-Password header.
    # Left empty by default so local dev needs no setup; deployments should set it.
    APP_ACCESS_PASSWORD: str = ""

    # LLM
    LLM_PROVIDER: str = "gemini"
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Storage
    DATA_DIR: Path = BASE_DIR / "data"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'autods.db'}"

    # CORS
    CORS_ORIGINS: str = '["http://localhost:3000"]'

    # Upload guards
    MAX_UPLOAD_MB: int = 200
    MAX_ROWS: int = 500_000

    # ML
    CV_FOLDS: int = 5
    SCORING_METRIC_CLASSIFICATION: str = "f1_weighted"
    SCORING_METRIC_REGRESSION: str = "r2"
    RANDOM_STATE: int = 42
    TUNING_N_ITER: int = 8          # kept for compat; Optuna uses OPTUNA_N_TRIALS
    OPTUNA_N_TRIALS: int = 30
    OPTUNA_TIMEOUT_S: int = 180     # max seconds per model study
    SHAP_SAMPLE_SIZE: int = 200

    # Forecasting
    FORECAST_PERIODS: int = 30      # number of future periods Prophet predicts

    # Artifact retention
    ARTIFACT_RETENTION_DAYS: int = 30

    @property
    def cors_origins_list(self) -> list[str]:
        try:
            return json.loads(self.CORS_ORIGINS)
        except json.JSONDecodeError:
            return [self.CORS_ORIGINS]

    @property
    def runs_dir(self) -> Path:
        path = self.DATA_DIR / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return settings
