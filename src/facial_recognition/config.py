"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Facial Recognition System"
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/faces.db"

    # Recognition settings
    detection_model: Literal["hog", "cnn"] = Field(
        default="hog",
        description="Face detection model: 'hog' (faster, CPU) or 'cnn' (accurate, GPU)",
    )
    encoding_model: Literal["small", "large"] = Field(
        default="large",
        description="Encoding model: 'small' (5 landmarks) or 'large' (68 landmarks)",
    )
    tolerance: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Face matching tolerance (lower = stricter)",
    )
    min_face_size: int = Field(
        default=20,
        ge=1,
        description="Minimum face size in pixels to detect",
    )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Paths
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        path = Path("./data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        path = self.data_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
