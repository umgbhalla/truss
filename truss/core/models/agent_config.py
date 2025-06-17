"""SQLAlchemy ORM model for the *agent_configs* table.

This mirrors :class:`truss.data_models.AgentConfig` but is designed for persistent
storage in Postgres.  Nested JSON structures such as *llm_config* and *tools*
are stored in JSON/JSONB columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy import ForeignKey
from typing import Any

from .base import Base


_JSON_TYPE = JSON  # Generic JSON works across dialects

try:
    _UUID_TYPE = UUID  # type: ignore
except ImportError:  # pragma: no cover
    _UUID_TYPE = String(36)  # Fallback for SQLite tests


class AgentConfigORM(Base):
    """Persistent representation of an agent configuration."""

    __tablename__ = "agent_configs"

    id = Column(_UUID_TYPE, primary_key=True, default=uuid.uuid4, nullable=False)
    name = Column(String(length=255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    llm_config = Column(_JSON_TYPE, nullable=False)
    mcp_servers = Column(_JSON_TYPE, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentConfigORM id={self.id} name={self.name!r}>" 
    
    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "system_prompt": self.system_prompt,
            "llm_config": self.llm_config,
            "mcp_servers": self.mcp_servers,
        }
