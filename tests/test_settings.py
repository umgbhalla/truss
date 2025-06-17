
from truss.settings import get_settings, Environment


def test_default_settings():
    settings = get_settings()
    assert settings.environment == Environment.LOCAL
    assert settings.database_url.startswith("sqlite")
    assert settings.temporal_url == "localhost:7233"
    assert settings.redis_url.endswith(":6379/0")


def test_env_override(monkeypatch):
    monkeypatch.setenv("TRUSS_ENVIRONMENT", "prod")
    monkeypatch.setenv("TRUSS_DATABASE_URL", "postgresql://u:p@db/truss")
    monkeypatch.setenv("TRUSS_TEMPORAL_HOST", "temporal.prod")
    monkeypatch.setenv("TRUSS_TEMPORAL_PORT", "7234")
    # Ensure cached instance is cleared

    get_settings.cache_clear()  # type: ignore[attr-defined]

    settings = get_settings()
    assert settings.environment == Environment.PROD
    assert settings.database_url == "postgresql://u:p@db/truss"
    assert settings.temporal_url == "temporal.prod:7234"

    # Clean
    get_settings.cache_clear()  # type: ignore[attr-defined] 
