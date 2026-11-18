from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEROVISION_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/aerovision.db"   # PostgreSQL : postgresql+psycopg://...
    storage_dir: Path = Path("./data/storage")
    models_dir: Path = Path("./models")
    device: str | None = None
    max_upload_mb: int = 20
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    # Politique de décision (sur la confiance calibrée, 0-1).
    # Aucune décision n'est finale sans validation humaine : ces seuils servent
    # uniquement à PRIORISER la file de validation.
    review_low: float = 0.20     # en dessous : « conforme probable »
    review_high: float = 0.80    # au-dessus : « non conforme probable »
    yolo_min_conf: float = 0.25
    yolo_match_iou: float = 0.1


@lru_cache
def get_settings() -> Settings:
    return Settings()
