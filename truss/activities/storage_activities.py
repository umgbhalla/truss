"""Temporal activity class providing durable database operations.

The *StorageActivities* class is meant to be registered with a Temporal worker.
It wraps blocking SQLAlchemy CRUD operations provided by
:class:`truss.core.storage.postgres_storage.PostgresStorage` with
``anyio.to_thread.run_sync`` so the worker event-loop remains responsive.
"""

from __future__ import annotations

import anyio
from uuid import UUID
from typing import List

from temporalio import activity

from truss.core.storage import PostgresStorage
from truss.data_models import Message, AgentMemory, AgentConfig


class StorageActivities:  # noqa  # name dictated by tech spec
    """Temporal activity collection encapsulating DB persistence logic."""

    def __init__(self, storage: PostgresStorage):
        self._storage = storage
        
    @activity.defn(name="CreateRun")  # explicit names keep API stable
    async def create_run(self, session_id: UUID) -> UUID:  # noqa: D401 – imperative
        """Insert a new *Run* row and return its primary-key UUID."""
        run = await anyio.to_thread.run_sync(self._storage.create_run, session_id)
        return run.id  # type: ignore[return-value]

    @activity.defn(name="CreateRunStep")
    async def create_run_step(self, run_id: UUID, message: Message) -> UUID:
        """Persist a :class:`Message` as a new *RunStep* row."""
        step = await anyio.to_thread.run_sync(
            self._storage.create_run_step_from_message, run_id, message
        )
        return step.id  # type: ignore[return-value]

    @activity.defn(name="GetRunMemory")
    async def get_run_memory(self, session_id: UUID) -> AgentMemory:
        """Fetch all messages for a session and convert to :class:`AgentMemory`."""
        steps = await anyio.to_thread.run_sync(
            self._storage.get_steps_for_session, session_id
        )
        messages: List[Message] = []
        for step in steps:
            role_val = step.role.value if hasattr(step.role, "value") else step.role
            messages.append(
                Message(
                    role=role_val,
                    content=step.content,
                    tool_calls=step.tool_calls,  # type: ignore[arg-type]
                    tool_call_id=step.tool_call_id,
                )
            )
        return AgentMemory(messages=messages)

    @activity.defn(name="LoadAgentConfig")
    async def load_agent_config(self, agent_id: UUID) -> AgentConfig:
        """Return the Pydantic :class:`AgentConfig` for the given identifier."""
        return await anyio.to_thread.run_sync(self._storage.load_agent_config, agent_id)

    @activity.defn(name="FinalizeRun")
    async def finalize_run(self, run_id: UUID, status: str, error: str | None = None) -> None:
        """Update final *Run* status and optional error message."""
        await anyio.to_thread.run_sync(
            self._storage.update_run_status, run_id, status, error
        ) 
