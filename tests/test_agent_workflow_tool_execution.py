import sys
import pathlib  # added
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # ensure root on path

import pytest
import uuid

from temporalio.testing import WorkflowEnvironment
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio import worker as _worker
from temporalio import activity

from truss.data_models import (
    AgentWorkflowInput,
    Message,
    AgentMemory,
    ToolCall,
    ToolCallResult,
)
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow


@pytest.mark.asyncio
async def test_workflow_executes_tool_and_persists_result():  # noqa: D401 – imperative test
    """Workflow should execute a requested tool then finish successfully."""

    session_uuid = uuid.uuid4()
    run_uuid = uuid.uuid4()

    # Shared recorders for validation after workflow completes
    recorder: dict[str, list] = {
        "create_run_step_roles": [],  # sequence of roles persisted
        "executed_tool_names": [],
        "finalize": [],
    }

    @activity.defn(name="CreateRun")
    async def fake_create_run(session_id):  # noqa: D401
        assert str(session_id) == str(session_uuid)
        return run_uuid

    @activity.defn(name="CreateRunStep")
    async def fake_create_run_step(run_id, message):  # noqa: D401
        assert str(run_id) == str(run_uuid)
        role_val = message.role if hasattr(message, "role") else message["role"]
        recorder["create_run_step_roles"].append(role_val)
        # Return synthetic PK
        return uuid.uuid4()

    @activity.defn(name="GetRunMemory")
    async def fake_get_run_memory(session_id):  # noqa: D401
        # Always return minimal memory – content is irrelevant for this test
        assert str(session_id) == str(session_uuid)
        return AgentMemory(messages=[Message(role="user", content="ping")])

    # The first assistant turn returns a tool call, second returns final answer.
    llm_call_count = {"n": 0}

    @activity.defn(name="LLMStreamPublish")
    async def fake_llm_stream_publish(agent_config, messages, session_id, run_id):  # noqa: D401
        assert str(session_id) == str(session_uuid)
        assert str(run_id) == str(run_uuid)
        llm_call_count["n"] += 1
        if llm_call_count["n"] == 1:
            # First invocation – request a tool call
            tool_call = ToolCall(name="web_search", arguments={"query": "cats"})
            return Message(role="assistant", content=None, tool_calls=[tool_call])
        else:
            # Second invocation – no tool calls -> workflow completes
            return Message(role="assistant", content="Here are the results", tool_calls=None)

    @activity.defn(name="ExecuteTool")
    async def fake_execute_tool(tool_call):  # noqa: D401
        # ToolCall may be dict (from JSON) or ToolCall instance according to converter
        name_val = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tc_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
        recorder["executed_tool_names"].append(name_val)
        return ToolCallResult(tool_call_id=tc_id, content="stub-result")

    @activity.defn(name="FinalizeRun")
    async def fake_finalize_run(run_id, status, error=None):  # noqa: D401
        recorder["finalize"].append((run_id, status, error))
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
            fake_execute_tool,
            fake_finalize_run,
        ],
    )

    async with worker:
        payload = AgentWorkflowInput(
            session_id=str(session_uuid),
            user_message="Need info",  # alias for user_message_text
        )

        handle = await env.client.start_workflow(
            TemporalAgentExecutionWorkflow.execute,
            payload,
            id=f"wf-{uuid.uuid4()}",
            task_queue="test-agent-queue",
        )

        result = await handle.result()

    # Workflow should succeed
    assert result.status == "completed"
    assert result.run_id == str(run_uuid)
    assert result.final_message.content == "Here are the results"

    # Tool execution path invoked exactly once
    assert recorder["executed_tool_names"] == ["web_search"]

    # RunStep rows: one for user + one (or more) for tool results. Ensure *tool* present.
    assert "user" in recorder["create_run_step_roles"]
    assert "tool" in recorder["create_run_step_roles"]

    # FinalizeRun called once with completed status
    assert recorder["finalize"] == [(str(run_uuid), "completed", None)]

    await env.shutdown()
