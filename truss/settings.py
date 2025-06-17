"""Centralised application configuration powered by *pydantic-settings*.

The :class:`Settings` object provides a typed view over environment variables
prefixed with ``TRUSS_``.  A singleton instance can be obtained via
:func:`get_settings` which caches the loaded values for the process lifetime.

Example environment variables recognised::

    TRUSS_ENVIRONMENT=prod
    TRUSS_DATABASE_URL=postgresql+psycopg://user:pass@localhost/truss
    TRUSS_TEMPORAL_HOST=temporal.example.com
    TRUSS_TEMPORAL_PORT=7233
    TRUSS_REDIS_URL=redis://cache:6379/0

When running locally with no env vars set, sane defaults are used (SQLite DB,
Temporal and Redis on localhost).
"""

from __future__ import annotations

import functools
from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Environment",
    "Settings",
    "get_settings",
]


class Environment(str, Enum):
    """Deployment environment discriminator."""

    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    """Application configuration loaded from the OS environment or .env file."""

    environment: Environment = Environment.LOCAL
    database_url: str = "sqlite:///truss.db"
    temporal_host: str = "localhost"
    temporal_port: int = 7233
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_prefix="TRUSS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # allow unrelated TRUSS_* vars (e.g., API keys) without error
    )

    @property
    def temporal_url(self) -> str:
        """Return *host:port* string for connecting to the Temporal server."""
        return f"{self.temporal_host}:{self.temporal_port}"



@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance for the process."""
    return Settings()  # type: ignore[call-arg] 
