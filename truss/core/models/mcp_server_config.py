"""SQLAlchemy ORM model for MCP server configuration entries."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Column, String, Boolean, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID

from .base import Base

try:
    _UUID_TYPE = UUID  # type: ignore
except ImportError:  # pragma: no cover – SQLite fallback
    from sqlalchemy import String as _Str

    _UUID_TYPE = _Str(36)


class MCPServerConfigORM(Base):  # noqa: D101 – naming constrained by spec
    """Persistent representation of an MCP server configuration."""

    __tablename__ = "mcp_server_configs"

    # Human-readable unique identifier (not a UUID) – e.g. "local-sqlite", "prod-delta".
    name = Column(String(length=128), primary_key=True, nullable=False)

    command = Column(String(length=255), nullable=False)
    args = Column(JSON, nullable=False, default=list)  # list[str]
    env = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MCPServerConfigORM name={self.name!r} enabled={self.enabled}>" 

    def _to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "description": self.description,
            "enabled": self.enabled,
        }   
