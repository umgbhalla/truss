from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID

from .base import Base

try:
    _UUID_TYPE = UUID  # type: ignore
except ImportError:  # pragma: no cover
    # When the Postgres dialect is unavailable (e.g. in-memory SQLite tests)
    _UUID_TYPE = String(36)


class RunSessionORM(Base):
    """Persistent representation of a session grouping multiple Runs.

    A *run session* represents a conversation or workflow instance executed by
    a particular :class:`truss.core.models.agent_config.AgentConfigORM` for a
    specific user.  It holds high-level metadata and links subsequent
    :class:`truss.core.models.run.RunORM` rows via a foreign-key relationship.
    """

    __tablename__ = "run_sessions"

    id = Column(_UUID_TYPE, primary_key=True, default=uuid.uuid4, nullable=False)

    # Foreign key to the agent configuration that produced this session
    agent_config_id = Column(
        _UUID_TYPE,
        ForeignKey("agent_configs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Identifier representing the end-user that initiated the session.  We keep
    # it as a simple string to remain backend-agnostic (could be a UUID, email,
    # auth provider subject, etc.).
    user_id = Column(String(length=64), nullable=False)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    
    def __repr__(self) -> str:  # pragma: no cover
        return f"<RunSessionORM id={self.id} user_id={self.user_id!r}>" 

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "agent_config_id": str(self.agent_config_id),
        }
