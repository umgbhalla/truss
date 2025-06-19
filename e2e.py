from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from temporalio.client import Client, TLSConfig  # type: ignore

from uuid import UUID
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow

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


async def main():
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

    # Global handle – populated on application startup
    _temporal_client: Optional[Client] = None
    # Storage handle (sync) – used directly by API routes
    _storage: Optional[PostgresStorage] = None
    url = _get_temporal_url()
    _temporal_client = await _connect_temporal(url)
    db_url = os.getenv("DATABASE_URL", "sqlite:///truss.db")
    storage = PostgresStorage.from_database_url(db_url)
    Base.metadata.create_all(storage._engine)  # type: ignore[attr-defined]
    _storage = storage

    # create agent
    mcp_list = get_storage().list_mcp_server_configs()
    if "sqlite" not in [mcp.name for mcp in mcp_list]:
        get_storage().create_mcp_server_config(**dict(
            name="sqlite",
            command="uvx",
            args= ["mcp-server-sqlite", "--db-path", "test.db"],
            ))
    if "mcp-cf" not in [mcp.name for mcp in mcp_list]:
        get_storage().create_mcp_server_config(
            **dict(
                name="mcp-cf",
                url="https://docs.mcp.cloudflare.com/sse",
            )
        )

    agent_list = get_storage().list_agent_configs()
    agent_list = [
        dict(
            id=str(o.id),
            name=o.name,
            system_prompt=o.system_prompt,
            llm_config=o.llm_config,
            mcp_servers=o.mcp_servers,
        )
        for o in agent_list
    ]
    print(agent_list)
    if "r247" not in [agent["name"] for agent in agent_list]:
        agent_config_orm = get_storage().create_agent_config(
      name="r247",
      system_prompt="You are a helpful assistant that can answer questions about the SQLite database.",
      llm_config={
        "model_name": "openrouter/openai/gpt-4.1-mini",
      },
      mcp_servers=["mcp-cf"],
    )
        agent_config = {
        "id": str(agent_config_orm.id),
        "name": agent_config_orm.name,
        "system_prompt": agent_config_orm.system_prompt,
        "llm_config": agent_config_orm.llm_config,
        "mcp_servers": agent_config_orm.mcp_servers
    }
        print(agent_config)
    else:
        agent_config = agent_list[0]
    session_id = str(get_storage().create_session(
            agent_config_id=UUID(agent_config["id"]), user_id="test"
        ).id)
    print(session_id)

    # Proceed even if session missing (testing convenience).
    try:
        get_storage().get_session(UUID(session_id))
    except KeyError:
        logger.info("Session %s not found – continuing to start workflow", session_id)

    try:
        await _temporal_client.start_workflow(
          TemporalAgentExecutionWorkflow.execute,
          id=session_id,  
          task_queue="truss-agent-queue",
          args=[{"session_id": session_id, "user_message": "can u show me what docs we can search?", "agent_config": agent_config}],
      )
    except Exception as exc:
        logger.warning(
          "Failed to start workflow: %s",
          exc,
      )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
