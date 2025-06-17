"""Alembic environment configuration.

This script exposes the SQLAlchemy metadata for `alembic revision --autogenerate`
commands.  It is intentionally minimal because the project uses the `uv` tool
and prefers to keep runtime dependencies light.  The database URL should be
provided via the `DATABASE_URL` environment variable when running Alembic
commands.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
import sys
import pathlib

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure the project root is discoverable for module imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from truss.core.models.base import metadata  # noqa: E402 – imported after sys.path setup


config = context.config  # type: ignore[attr-defined]

# Interpret the config file for Python logging. This line sets up loggers
# basically according to the config file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata  # noqa  # required by Alembic

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///truss.db")

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = DATABASE_URL
    context.configure(  # type: ignore[attr-defined]
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    connectable = engine_from_config(  # type: ignore[arg-type]
        config.get_section(config.config_ini_section),  # type: ignore[arg-type]
        prefix="sqlalchemy.",
        url=DATABASE_URL,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(  # type: ignore[attr-defined]
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


def main() -> None:  # pragma: no cover – entrypoint
    if context.is_offline_mode():  # type: ignore[attr-defined]
        run_migrations_offline()
    else:
        run_migrations_online()


main() 
