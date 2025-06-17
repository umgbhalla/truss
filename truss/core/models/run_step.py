from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

import enum

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON

from .base import Base

try:
    _UUID_TYPE = UUID  # type: ignore
except ImportError:  # pragma: no cover
    _UUID_TYPE = String(36)

_JSON_TYPE = JSON 


class MessageRole(str, enum.Enum):
    """Enumerated roles for messages stored as RunStep rows."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStepORM(Base):
    """Persistent representation of a single assistant/user/tool message.

    A *run step* corresponds to one atomic exchange inside a :class:`truss.core.models.run.RunORM`.
    It stores the message role, content and any tool invocation details so that the
    full conversation can be reconstructed deterministically.
    """

    __tablename__ = "run_steps"

    id: uuid.UUID | None = Column(
        _UUID_TYPE, primary_key=True, default=uuid.uuid4, nullable=False
    )

    # Foreign-key relationship back to the owning Run
    run_id: uuid.UUID | None = Column(
        _UUID_TYPE,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

   
    role: MessageRole | str = Column(String(length=32), nullable=False)
    content: Optional[str] = Column(Text, nullable=True)

    
    # We store the raw LiteLLM "tool_calls" array for full fidelity.
    tool_calls: Optional[List[dict[str, Any]]] = Column(_JSON_TYPE, nullable=True)

    # In follow-up messages referencing a particular tool call, we persist
    # the identifier so we can correlate tool execution results.
    tool_call_id: Optional[str] = Column(String(length=64), nullable=True)

    created_at: datetime = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RunStepORM id={self.id} run_id={self.run_id} role={self.role!r}>"
        ) 
        
    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "run_id": str(self.run_id),
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "created_at": self.created_at,
        }   
