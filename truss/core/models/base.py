"""Base declarative class and shared SQLAlchemy metadata.

This module defines a single :data:`Base` object that should be subclassed by all
ORM models in the codebase.  It centralises the naming convention for indexes
and constraints which helps Alembic autogenerate predictable migration scripts.

The design adheres to the guidance in SQLAlchemy docs:
https://docs.sqlalchemy.org/en/20/core/metadata.html#naming-conventions
"""

from __future__ import annotations

from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData

__all__ = [
    "Base",
    "metadata",
    "NAMING_CONVENTION",
]

# These ensure deterministic constraint/index names across all environments.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Metadata instance shared across all declarative models.
metadata = MetaData(naming_convention=NAMING_CONVENTION)

# Base class for declarative models.
Base = declarative_base(metadata=metadata) 
