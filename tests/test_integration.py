import os
from uuid import UUID

import pytest
from httpx import AsyncClient
from httpx import ASGITransport

from temporalio import activity
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment
from temporalio.contrib.pydantic import pydantic_data_converter

from truss.api import app  # FastAPI instance
from truss.api import main as api_main
from truss.core.storage import PostgresStorage
from truss.core.models.agent_config import AgentConfigORM
from truss.activities.storage_activities import StorageActivities
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow
from truss.data_models import Message


@pytest.mark.asyncio
async def test_end_to_end_workflow_success(tmp_path):  # noqa: D401 – imperative test
    """Full integration: API → workflow → DB persistence."""

    db_path = tmp_path / "e2e.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"  # API will not auto-connect

    # Prepare shared storage instance used by StorageActivities and assertions
    storage = PostgresStorage.from_database_url(os.environ["DATABASE_URL"])
    storage_activities = StorageActivities(storage)

    # Ensure schema exists for fresh SQLite DB
    from truss.core.models.base import Base as _Base
    _Base.metadata.create_all(storage._engine)  # type: ignore[attr-defined]

    # Seed AgentConfig via direct ORM session
    with storage._session_scope() as session:  # type: ignore[attr-defined]
        ac = AgentConfigORM(
            name="IntegrationTestAgent",
            system_prompt="You are helpful",
            llm_config={"model_name": "gpt-4o"},
            tools=[],
        )
        session.add(ac)
        session.flush()
        agent_id = ac.id

    @activity.defn(name="LLMStreamPublish")
    async def fake_llm_stream_publish(agent_config, messages, session_id, run_id):  # noqa: D401
        # Mimic real activity by persisting assistant message
        assistant_msg = Message(role="assistant", content="done", tool_calls=None)
        storage.create_run_step_from_message(UUID(str(run_id)), assistant_msg)
        return assistant_msg

    @activity.defn(name="ExecuteTool")
    async def fake_execute_tool(tool_call):  # noqa: D401 – not used in this scenario
        return None

    env = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)

    # Inject Temporal client into API module so /runs endpoint works
    api_main._temporal_client = env.client  # type: ignore
    api_main._storage = storage  # type: ignore

    # Build activity list (bound storage funcs + fakes)
    activities = [
        *[
            storage_activities.create_run,
            storage_activities.create_run_step,
            storage_activities.get_run_memory,
            storage_activities.load_agent_config,
            storage_activities.finalize_run,
        ],
        fake_llm_stream_publish,
        fake_execute_tool,
    ]

    worker = Worker(
        env.client,
        task_queue="truss-agent-queue",
        workflows=[TemporalAgentExecutionWorkflow],
        activities=activities,
    )

    async with worker:
        # Use AsyncClient for FastAPI in async context
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create session via API
            resp = await client.post(
                "/sessions",
                json={"agent_id": str(agent_id), "user_id": "u-1"},
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            # Start run -> triggers workflow
            resp = await client.post(
                f"/sessions/{session_id}/runs",
                json={"message": "Hello Agent"},
            )
            assert resp.status_code == 202
            workflow_id = resp.json()["workflow_id"]

            # Wait for workflow completion
            handle = env.client.get_workflow_handle(workflow_id)
            result = await handle.result()
            assert result.status == "completed"

    steps = storage.get_steps_for_session(UUID(session_id))
    # At minimum expect user message + assistant reply
    assert len(steps) >= 2

    await env.shutdown() 
