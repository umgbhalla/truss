"""Storage abstractions for persisting and retrieving Truss runtime data.

Currently provides a thin `PostgresStorage` class that wraps synchronous SQLAlchemy
operations.  All public methods are designed to be thread-safe so they can be
invoked from Temporal activities via `anyio.to_thread.run_sync`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Any
from uuid import UUID

from sqlalchemy import create_engine, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from truss.core.models.agent_config import AgentConfigORM
from truss.core.models.mcp_server_config import MCPServerConfigORM
from truss.core.models.run import RunORM, RunStatus
from truss.core.models.run_session import RunSessionORM
from truss.core.models.run_step import RunStepORM, MessageRole
from truss.data_models import Message, AgentConfig


class PostgresStorage:  
    """Concrete storage implementation backed by a Postgres database.

    Parameters
    engine
        SQLAlchemy *Engine* instance connected to the target Postgres (or
        compatible) database.
    """

    def __init__(self, engine: Engine) -> None:  
        self._engine: Engine = engine
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    @contextmanager
    def _session_scope(self) -> Iterable[Session]:  
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:  
            session.rollback()
            raise
        finally:
            session.close()


    def create_run(self, session_id: UUID) -> RunORM:
        """Insert a new *Run* row and return the ORM instance."""
        with self._session_scope() as session:
            run = RunORM(session_id=session_id, status=RunStatus.PENDING)
            session.add(run)
            session.flush()  # populate PK
            session.refresh(run)
            return run

    def create_run_step_from_message(self, run_id: UUID, message: Message) -> RunStepORM:
        """Persist a Pydantic *Message* as a *RunStep* row."""
        tool_calls_json = (
            [tool_call.model_dump() for tool_call in message.tool_calls]
            if message.tool_calls
            else None
        )
        with self._session_scope() as session:
            step = RunStepORM(
                run_id=run_id,
                role=MessageRole(message.role),
                content=message.content,
                tool_calls=tool_calls_json,
                tool_call_id=message.tool_call_id,
            )
            session.add(step)
            session.flush()
            session.refresh(step)
            return step

    def get_steps_for_session(self, session_id: UUID) -> List[RunStepORM]:
        """Return all *RunStep* rows for a given *RunSession*, ordered chronologically."""
        with self._session_scope() as session:
            stmt = (
                select(RunStepORM)
                .join(RunORM, RunStepORM.run_id == RunORM.id)
                .where(RunORM.session_id == session_id)
                .order_by(RunStepORM.created_at)
            )
            return list(session.execute(stmt).scalars())

    def update_run_status(self, run_id: UUID, status: str, error: Optional[str] = None) -> None:
        """Update *Run.status* (and optionally *error*) atomically."""
        with self._session_scope() as session:
            stmt = (
                update(RunORM)
                .where(RunORM.id == run_id)
                .values(status=status, error=error)
            )
            session.execute(stmt)

    def load_agent_config(self, agent_id: UUID) -> AgentConfig:
        """Fetch :class:`AgentConfig` Pydantic model for a given identifier."""
        with self._session_scope() as session:
            obj = session.get(AgentConfigORM, agent_id)
            if obj is None:  
                raise KeyError(f"AgentConfig {agent_id} not found")
            return AgentConfig(
                id=str(obj.id),
                name=obj.name,
                system_prompt=obj.system_prompt,
                llm_config=obj.llm_config,  
                tools=obj.tools,
            )

    def create_session(self, agent_config_id: UUID, user_id: str) -> RunSessionORM:
        """Insert a new :class:`RunSessionORM` row and return the instance."""

        # Validate agent exists – provides nicer error than FK violation.
        with self._session_scope() as session:
            if session.get(AgentConfigORM, agent_config_id) is None:
                raise KeyError(f"AgentConfig {agent_config_id} not found")

            session_obj = RunSessionORM(agent_config_id=agent_config_id, user_id=user_id)
            session.add(session_obj)
            session.flush()
            session.refresh(session_obj)
            return session_obj

    @classmethod
    def from_database_url(cls, url: str) -> "PostgresStorage":
        """Create a :class:`PostgresStorage` from database URL."""
        engine = create_engine(url, future=True)
        return cls(engine)


    def get_session(self, session_id: UUID) -> RunSessionORM:
        """Return :class:`RunSessionORM` or raise ``KeyError`` if missing."""
        with self._session_scope() as session:
            obj = session.get(RunSessionORM, session_id)
            if obj is None:
                raise KeyError(f"RunSession {session_id} not found")
            return obj


    def create_agent_config(
        self,
        *,
        name: str,
        system_prompt: str,
        llm_config: dict[str, Any],
        mcp_servers: list[str] = [],
    ) -> AgentConfigORM:
        """Insert a new :class:`AgentConfigORM` row and return the instance."""

        with self._session_scope() as session:
            # for mcp_server in mcp_servers: # mcp_server is the name column of MCPServerConfigORM
            #     mcp_obj = session.query(MCPServerConfigORM).filter(MCPServerConfigORM.name == mcp_server).first()
            #     if mcp_obj is None:
            #         raise KeyError(f"MCPServerConfig '{mcp_server}' not found")
            mcp_server_objs = []
            for mcp_server in mcp_servers:
                mcp_obj = session.query(MCPServerConfigORM).filter(MCPServerConfigORM.name == mcp_server).first()
                if mcp_obj is None:
                    raise KeyError(f"MCPServerConfig '{mcp_server}' not found")
                mcp_server_objs.append(mcp_obj._to_dict())

            obj = AgentConfigORM(
                name=name,
                system_prompt=system_prompt,
                llm_config=llm_config,
                mcp_servers=mcp_server_objs,
            )
            session.add(obj)
            session.flush()
            session.refresh(obj)
            return obj

    def get_agent_config(self, agent_id: UUID) -> AgentConfigORM:
        """Return :class:`AgentConfigORM` or raise ``KeyError`` if missing."""

        with self._session_scope() as session:
            obj = session.get(AgentConfigORM, agent_id)
            if obj is None:
                raise KeyError(f"AgentConfig {agent_id} not found")
            return obj

    def list_agent_configs(self) -> List[AgentConfigORM]:
        """Return list of all :class:`AgentConfigORM` rows."""

        with self._session_scope() as session:
            stmt = select(AgentConfigORM).order_by(AgentConfigORM.created_at)
            return list(session.execute(stmt).scalars())

    def delete_agent_config(self, agent_id: UUID) -> None:
        """Delete an AgentConfig by ID. Raises KeyError if not found."""

        with self._session_scope() as session:
            obj = session.get(AgentConfigORM, agent_id)
            if obj is None:
                raise KeyError(f"AgentConfig {agent_id} not found")
            session.delete(obj)

    def create_mcp_server_config(self, **kwargs: Any) -> MCPServerConfigORM:
        """Insert a new MCPServerConfig row.

        Accepts flexible keyword arguments so callers can specify only the
        fields relevant for the chosen transport (e.g. *url* for HTTP, or
        *command* / *args* for stdio). 
        """

        if "name" not in kwargs:
            raise TypeError("create_mcp_server_config() missing required argument: 'name'")


        with self._session_scope() as session:
            if session.get(MCPServerConfigORM, kwargs["name"]):
                raise ValueError(f"MCPServerConfig '{kwargs['name']}' already exists")

            obj = MCPServerConfigORM(**kwargs)  # type: ignore[arg-type]
            session.add(obj)
            session.flush()
            session.refresh(obj)
            return obj

    def get_mcp_server_config(self, name: str) -> MCPServerConfigORM:
        with self._session_scope() as session:
            obj = session.get(MCPServerConfigORM, name)
            if obj is None:
                raise KeyError(f"MCPServerConfig '{name}' not found")
            return obj

    def list_mcp_server_configs(self) -> List[MCPServerConfigORM]:
        with self._session_scope() as session:
            stmt = select(MCPServerConfigORM).order_by(MCPServerConfigORM.name)
            return list(session.execute(stmt).scalars())

    def update_mcp_server_config(self, name: str, **updates: Any) -> MCPServerConfigORM:
        with self._session_scope() as session:
            obj = session.get(MCPServerConfigORM, name)
            if obj is None:
                raise KeyError(f"MCPServerConfig '{name}' not found")
            for k, v in updates.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            session.add(obj)
            session.flush()
            session.refresh(obj)
            return obj

    def delete_mcp_server_config(self, name: str) -> None:
        with self._session_scope() as session:
            obj = session.get(MCPServerConfigORM, name)
            if obj is None:
                raise KeyError(f"MCPServerConfig '{name}' not found")
            session.delete(obj)
