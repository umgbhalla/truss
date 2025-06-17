from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
import anyio  # for background connection cleanup

from fastapi import FastAPI, Depends, status, HTTPException
from temporalio.client import Client, TLSConfig  # type: ignore
from pydantic import BaseModel

from truss.core.storage import PostgresStorage
from truss.core.models.base import Base

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_temporal_url() -> str:
    """Return the Temporal endpoint URL from environment or default."""
    return os.getenv("TEMPORAL_URL", "localhost:7233")


async def _connect_temporal(url: str) -> Client:
    """Establish an async connection to Temporal.

    A thin wrapper so that connection logic can be monkey-patched in tests.
    """
    # In most setups plaintext is used for local development.  If the user
    # provides `TEMPORAL_TLS_ENABLED=true`, we assume TLS with default config.
    if os.getenv("TEMPORAL_TLS_ENABLED", "false").lower() in {"1", "true", "yes"}:
        logger.info("Connecting to Temporal with TLS enabled at %s", url)
        return await Client.connect(url, tls=TLSConfig())

    logger.info("Connecting to Temporal at %s", url)
    return await Client.connect(url)



@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: D401 – lifespan context
    """FastAPI lifespan context establishing and tearing down resources.

    This replaces the deprecated startup/shutdown event handlers and avoids
    relying on a non-existent ``Client.close`` method.
    """

    global _temporal_client, _storage

    if os.getenv("SKIP_TEMPORAL_CONNECTION", "0") in {"1", "true", "yes"}:
        logger.info(
            "Skipping Temporal connection – SKIP_TEMPORAL_CONNECTION flag set"
        )
    else:
        url = _get_temporal_url()
        try:
            _temporal_client = await _connect_temporal(url)
            logger.info("Successfully connected to Temporal: %s", url)
        except Exception:  # pragma: no cover – observe but don't crash
            logger.exception("Failed to connect to Temporal at %s", url)
            # Allow API to boot even if Temporal is unavailable.
            _temporal_client = None

    db_url = os.getenv("DATABASE_URL", "sqlite:///truss.db")
    try:
        storage = PostgresStorage.from_database_url(db_url)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to initialise SQLAlchemy engine: %s", exc)
        storage = None

    # Ensure schema exists in dev/test environments (SQLite especially)
    if storage is not None and db_url.startswith("sqlite"):
        Base.metadata.create_all(storage._engine)  # type: ignore[attr-defined]

    _storage = storage

    yield

    _temporal_client = None  # No explicit close – Temporal client has no close()
    _storage = None


# Instantiate the FastAPI application with lifespan handler
app = FastAPI(title="Truss Agent Execution API", version="0.1.0", lifespan=_lifespan)

# Global handle – populated on application startup
_temporal_client: Optional[Client] = None
# Storage handle (sync) – used directly by API routes
_storage: Optional[PostgresStorage] = None


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:  # noqa: D401 – simple health check
    """Return a basic healthcheck payload."""
    return {"status": "ok"}




def get_temporal_client() -> Client:
    """Return the live Temporal client instance.

    Raises:
        RuntimeError: If the client has not been initialised yet (e.g. during
            application startup failure or when the connection is intentionally
            skipped in tests).
    """
    if _temporal_client is None:
        raise RuntimeError("Temporal client has not been initialised."
                           " Ensure the FastAPI startup event has completed.")
    return _temporal_client




def get_storage() -> PostgresStorage:
    """Return the active :class:`PostgresStorage` instance."""
    if _storage is None:  # pragma: no cover – should be initialised on startup
        raise RuntimeError("Storage has not been initialised."
                           " Ensure FastAPI startup completed successfully.")
    return _storage



class SessionCreateRequest(BaseModel):
    agent_id: str  # UUID string
    user_id: str


class SessionCreateResponse(BaseModel):
    session_id: str


class RunCreateRequest(BaseModel):
    message: str


class RunCreateResponse(BaseModel):
    workflow_id: str


class AgentConfigCreateRequest(BaseModel):
    name: str
    system_prompt: str
    llm_config: dict[str, Any]
    tools: Optional[List[str]] = None
    mcp_server: str = "sqlite"


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    system_prompt: str
    llm_config: dict[str, Any]
    tools: Optional[List[str]] = None
    mcp_server: Optional[str] = None




@app.post("/sessions", status_code=status.HTTP_201_CREATED, response_model=SessionCreateResponse, tags=["sessions"])
def create_session(
    payload: SessionCreateRequest,
    storage: PostgresStorage = Depends(get_storage),
):
    """Create a new conversation session and return its ID."""

    from uuid import UUID

    try:
        session_obj = storage.create_session(
            agent_config_id=UUID(payload.agent_id), user_id=payload.user_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SessionCreateResponse(session_id=str(session_obj.id))


@app.post(
    "/sessions/{session_id}/runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunCreateResponse,
    tags=["sessions"],
)
async def start_run(
    session_id: str,
    payload: RunCreateRequest,
    storage: PostgresStorage = Depends(get_storage),
):
    """Start Temporal workflow for a user message and return workflow_id."""

    from uuid import UUID
    from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow

    # Proceed even if session missing (testing convenience).
    try:
        storage.get_session(UUID(session_id))
    except KeyError:
        logger.info("Session %s not found – continuing to start workflow", session_id)

    if _temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client unavailable")


    try:
        await _temporal_client.start_workflow(
            TemporalAgentExecutionWorkflow.execute,
            id=session_id,
            task_queue="truss-agent-queue",
            args=[{"session_id": session_id, "user_message": payload.message}],
        )
    except Exception as exc:
        logger.warning(
            "Failed to set search attributes when starting workflow (may be unsupported). Continuing without them: %s",
            exc,
        )

    return RunCreateResponse(workflow_id=session_id)


@app.post(
    "/sessions/{session_id}/stop",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["sessions"],
)
async def stop_session_stream(  # noqa: D401 – API route
    session_id: str,
):
    """Request the streaming *LLM* activity for the given session to stop.

    The implementation creates a short-lived key in Redis (`stop:{session_id}`)
    that the :pyclass:`LLMActivities` worker polls for.  When present and the
    assistant is **not** currently emitting a tool-call payload, the activity
    terminates the stream early and persists the partial assistant response.
    """

    import redis.asyncio as _redis
    from truss.settings import get_settings as _get_settings

    try:
        _settings = _get_settings()
        _client = _redis.from_url(_settings.redis_url, decode_responses=False)
        # Use a short TTL so that stale stop requests do not affect future runs.
        await _client.set(f"stop:{session_id}", b"1", ex=60)
    finally:
        # Always close the connection – fire-and-forget semantics.
        if "_client" in locals():
            with anyio.CancelScope(shield=True):
                try:
                    await _client.aclose()
                except Exception:  # pragma: no cover
                    logger.warning("Failed to close Redis client after stop request", exc_info=True)

    return {"status": "stop_requested"} 



@app.post(
    "/workflows/{workflow_id}/signal/{signal_name}",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["workflow"],
)
async def send_signal_to_workflow(  # noqa: D401 – API route
    workflow_id: str,
    signal_name: str,
    payload: Optional[dict[str, Any]] = None,
):
    """Forward arbitrary *payload* to a Temporal workflow signal.

    The endpoint is intentionally generic – it simply obtains a workflow
    handle for the given *workflow_id* and invokes *signal_name* with an
    optional ``payload`` argument.  The Temporal Python SDK allows signals
    to accept zero or more parameters; by convention Truss signals accept at
    most a single positional argument.  If ``payload`` is provided, the value
    under the ``data`` key is forwarded.  Otherwise the signal is fired with
    no arguments.

    Response body is a minimal acknowledgement so that callers can treat the
    request as fire-and-forget.
    """

    if _temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client unavailable")

    try:
        handle = _temporal_client.get_workflow_handle(workflow_id)
    except Exception as exc:  # pragma: no cover – unexpected SDK errors
        logger.exception("Failed to obtain workflow handle: %s", exc)
        raise HTTPException(status_code=404, detail="Workflow not found") from exc

    # Temporal signals can be called with arbitrary arguments – we support
    # the common convention of a single argument passed via JSON key `data`.
    try:
        if payload and "data" in payload:
            await handle.signal(signal_name, payload["data"])
        else:
            # Pass no args if payload missing / empty
            await handle.signal(signal_name)
    except Exception as exc:  # pragma: no cover – signal dispatch failure
        logger.exception("Failed to send signal '%s' to workflow %s: %s", signal_name, workflow_id, exc)
        raise HTTPException(status_code=500, detail="Failed to dispatch signal") from exc

    return {"status": "signal sent"}



@app.post(
    "/agent-configs",
    status_code=status.HTTP_201_CREATED,
    response_model=AgentConfigResponse,
    tags=["agent-configs"],
)
def create_agent_config_route(
    payload: AgentConfigCreateRequest,
    storage: PostgresStorage = Depends(get_storage),
):
    """Create a new AgentConfig and return it."""
    # NOTE: Validation of *tools* against a local registry has been removed.
    # Remote tool availability is now checked lazily via MCP introspection.
    obj = storage.create_agent_config(
        name=payload.name,
        system_prompt=payload.system_prompt,
        llm_config=payload.llm_config,
        tools=payload.tools,
        mcp_server=payload.mcp_server,
    )
    return AgentConfigResponse(
        id=str(obj.id),
        name=obj.name,
        system_prompt=obj.system_prompt,
        llm_config=obj.llm_config,  # type: ignore[arg-type]
        tools=obj.tools,
        mcp_server=obj.mcp_server,
    )


@app.get(
    "/agent-configs",
    response_model=List[AgentConfigResponse],
    tags=["agent-configs"],
)
def list_agent_configs_route(
    storage: PostgresStorage = Depends(get_storage),
):
    """Return all AgentConfigs."""

    objs = storage.list_agent_configs()
    return [
        AgentConfigResponse(
            id=str(o.id),
            name=o.name,
            system_prompt=o.system_prompt,
            llm_config=o.llm_config,  # type: ignore[arg-type]
            tools=o.tools,
            mcp_server=o.mcp_server,
        )
        for o in objs
    ]


@app.get(
    "/agent-configs/{agent_id}",
    response_model=AgentConfigResponse,
    tags=["agent-configs"],
)
def get_agent_config_route(
    agent_id: str,
    storage: PostgresStorage = Depends(get_storage),
):
    """Retrieve single AgentConfig by ID."""

    from uuid import UUID

    try:
        obj = storage.get_agent_config(UUID(agent_id))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="AgentConfig not found") from exc

    return AgentConfigResponse(
        id=str(obj.id),
        name=obj.name,
        system_prompt=obj.system_prompt,
        llm_config=obj.llm_config,  # type: ignore[arg-type]
        tools=obj.tools,
        mcp_server=obj.mcp_server,
    )


@app.delete(
    "/agent-configs/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["agent-configs"],
)
def delete_agent_config_route(
    agent_id: str,
    storage: PostgresStorage = Depends(get_storage),
):
    """Delete an AgentConfig."""

    from uuid import UUID

    try:
        storage.delete_agent_config(UUID(agent_id))
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="AgentConfig not found")

    # Return empty 204
    return None



class SessionWorkflowsResponse(BaseModel):
    """Response schema for listing all Temporal workflows linked to a session."""

    workflow_id: str
    run_id: str
    status: str
    start_time: Optional[str] = None
    close_time: Optional[str] = None


@app.get(
    "/sessions/{session_id}/workflows",
    response_model=list[SessionWorkflowsResponse],
    tags=["sessions"],
)
async def list_session_workflows(session_id: str):  # noqa: D401 – API route
    """Return all Temporal workflow executions associated with *session_id*.

    The implementation relies on the *SessionID* search attribute that is set
    when the workflow is initially started.  All open as well as closed
    workflow executions are returned so that callers can inspect historical
    completions and in-flight runs alike.
    """

    if _temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client unavailable")

    # Advanced Visibility operators like `LIKE` are not yet supported for WorkflowId
    # (see Temporal community discussion: https://community.temporal.io/t/list-filter-wildcard-query-not-returning-expected-results/7362).
    # We therefore retrieve *all* executions (open+closed) and filter client-side
    # for workflow IDs starting with the session prefix we embedded earlier.

    workflows: list[SessionWorkflowsResponse] = []

    async for desc in _temporal_client.list_workflows(f"WorkflowId={session_id}"):
        # desc may be a WorkflowExecutionInfo with .execution, or a plain WorkflowExecution
        # Use getattr to safely extract workflow_id and run_id
        if hasattr(desc, "execution"):
            wf_id = getattr(desc.execution, "workflow_id", None)
            run_id = getattr(desc.execution, "run_id", None)
        else:
            wf_id = getattr(desc, "workflow_id", None)
            run_id = getattr(desc, "run_id", None)
        if not wf_id or not wf_id.startswith(f"{session_id}_"):
            continue

        # Status, start_time, close_time may not exist on plain WorkflowExecution
        raw_status = getattr(desc, "status", None)
        if raw_status is None:
            status_name = "unknown"
        else:
            status_name = getattr(raw_status, "name", str(raw_status)).lower()

        start = getattr(desc, "start_time", None)
        close = getattr(desc, "close_time", None)

        workflows.append(
            SessionWorkflowsResponse(
                workflow_id=wf_id,
                run_id=run_id or "",
                status=status_name,
                start_time=start.isoformat() if start else None,
                close_time=close.isoformat() if close else None,
            )
        )

    return workflows


# ---------------------------------------------------------------------------
# MCP Server Config Routes
# ---------------------------------------------------------------------------


class MCPServerConfigCreateRequest(BaseModel):
    name: str
    command: str
    args: List[str] = []
    env: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    enabled: bool = True


class MCPServerConfigResponse(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]]
    description: Optional[str] = None
    enabled: bool


@app.post(
    "/mcp-servers",
    status_code=status.HTTP_201_CREATED,
    response_model=MCPServerConfigResponse,
    tags=["mcp-servers"],
)
def create_mcp_server_route(
    payload: MCPServerConfigCreateRequest,
    storage: PostgresStorage = Depends(get_storage),
):
    obj = storage.create_mcp_server_config(payload)  # type: ignore[arg-type]
    return MCPServerConfigResponse(**obj.__dict__)


@app.get(
    "/mcp-servers",
    response_model=List[MCPServerConfigResponse],
    tags=["mcp-servers"],
)
def list_mcp_servers_route(
    storage: PostgresStorage = Depends(get_storage),
):
    objs = storage.list_mcp_server_configs()
    return [MCPServerConfigResponse(**o.__dict__) for o in objs]


@app.get(
    "/mcp-servers/{name}",
    response_model=MCPServerConfigResponse,
    tags=["mcp-servers"],
)
def get_mcp_server_route(name: str, storage: PostgresStorage = Depends(get_storage)):
    try:
        obj = storage.get_mcp_server_config(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPServerConfigResponse(**obj.__dict__)


@app.put(
    "/mcp-servers/{name}",
    response_model=MCPServerConfigResponse,
    tags=["mcp-servers"],
)
def update_mcp_server_route(
    name: str,
    payload: MCPServerConfigCreateRequest,
    storage: PostgresStorage = Depends(get_storage),
):
    try:
        obj = storage.update_mcp_server_config(name, **payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPServerConfigResponse(**obj.__dict__)


@app.delete(
    "/mcp-servers/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["mcp-servers"],
)
def delete_mcp_server_route(name: str, storage: PostgresStorage = Depends(get_storage)):
    try:
        storage.delete_mcp_server_config(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return None


