import sys
import pathlib  # added
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # ensure project root on path

import pytest
import uuid

from temporalio.testing import WorkflowEnvironment
from temporalio import worker as _worker  # Avoid global dependency collisions
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter

from truss.data_models import AgentWorkflowInput, Message, AgentMemory
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow


@pytest.mark.asyncio
async def test_workflow_initialisation_completes_without_tools():  # noqa: D401 – imperative test name
    """Happy-path: workflow completes after a single assistant turn without tool calls."""

    session_uuid = uuid.uuid4()
    run_uuid = uuid.uuid4()

    recorder: dict[str, list] = {
        "create_run": [],
        "create_run_step": [],
        "finalize": [],
    }

    @activity.defn(name="CreateRun")
    async def fake_create_run(session_id):  # noqa: D401 – test stub
        # Temporal converter turns UUID -> str payload, so compare by value
        assert str(session_id) == str(session_uuid)
        recorder["create_run"].append(session_id)
        return run_uuid

    @activity.defn(name="CreateRunStep")
    async def fake_create_run_step(run_id, message):  # noqa: D401 – test stub
        # Persist *user* message only (assistant handled by LLM activity)
        assert str(run_id) == str(run_uuid)
        # Depending on the data converter we may get a dict or Message
        role_val = message.role if hasattr(message, "role") else message["role"]
        recorder["create_run_step"].append(role_val)
        return uuid.uuid4()

    @activity.defn(name="GetRunMemory")
    async def fake_get_run_memory(session_id):  # noqa: D401 – test stub
        # Temporal converter turns UUID -> str payload, so compare by value
        assert str(session_id) == str(session_uuid)
        # Provide minimal memory so workflow can reach reasoning loop
        return AgentMemory(messages=[Message(role="user", content="hi")])

    @activity.defn(name="LLMStreamPublish")
    async def fake_llm_stream_publish(agent_config, messages, session_id, run_id):  # noqa: D401
        # Return assistant reply *without* tool calls so the workflow finishes.
        assert str(session_id) == str(session_uuid)
        assert str(run_id) == str(run_uuid)
        return Message(role="assistant", content="hello", tool_calls=None)

    @activity.defn(name="FinalizeRun")
    async def fake_finalize_run(run_id, status, error_msg=None):  # noqa: D401
        recorder["finalize"].append((run_id, status, error_msg))
        assert str(run_id) == str(run_uuid)

    env = await WorkflowEnvironment.start_time_skipping(data_converter=pydantic_data_converter)

    worker = _worker.Worker(
        env.client,
        task_queue="test-agent-queue",
        workflows=[TemporalAgentExecutionWorkflow],
        activities=[
            fake_create_run,
            fake_create_run_step,
            fake_get_run_memory,
            fake_llm_stream_publish,
            fake_finalize_run,
        ],
    )

    async with worker:
        input_payload = AgentWorkflowInput(
            session_id=str(session_uuid),
            user_message="Hello",  # alias for user_message_text
        )

        handle = await env.client.start_workflow(
            TemporalAgentExecutionWorkflow.execute,
            input_payload,
            id=f"wf-{uuid.uuid4()}",
            task_queue="test-agent-queue",
        )

        result = await handle.result()

    assert result.status == "completed"
    assert result.run_id == str(run_uuid)
    assert result.final_message.content == "hello"

    # DB-persistence side-effects (via activities)
    assert recorder["create_run"] == [str(session_uuid)]
    # One run step persisted for the *user* message
    assert recorder["create_run_step"] == ["user"]
    # FinalizeRun invoked exactly once with success status
    assert recorder["finalize"] == [(str(run_uuid), "completed", None)]

    await env.shutdown()
