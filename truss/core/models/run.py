from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


try:
    _UUID_TYPE = UUID  
except ImportError:  
    _UUID_TYPE = String(36)


class RunStatus(str, enum.Enum):
    """Enumerated lifecycle states for a Run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    


class RunORM(Base):
    """Persistent representation of a single execution *run*.

    A *run* represents an invocation of an agent (referenced via
    :class:`truss.core.models.run_session.RunSessionORM`) and tracks its
    lifecycle status alongside any error message.
    """

    __tablename__ = "runs"

    id = Column(_UUID_TYPE, primary_key=True, default=uuid.uuid4, nullable=False)

    session_id = Column(
        _UUID_TYPE,
        ForeignKey("run_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status = Column(
        Enum(RunStatus, name="run_status_enum", native_enum=False),
        default=RunStatus.PENDING,
        nullable=False,
    )

    error = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    
    def __repr__(self) -> str:  
        return (
            f"<RunORM id={self.id} session_id={self.session_id} status={self.status}>"
        ) 

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "status": self.status,
            "error": self.error,
        }
