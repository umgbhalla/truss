import pytest
import asyncio

from uuid import uuid4

from temporalio import workflow as temporal_workflow

from truss.data_models import (
    AgentWorkflowInput,
    Message,
    AgentMemory,
    ToolCall,
    ToolCallResult,
)
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow


@pytest.mark.asyncio
async def test_workflow_success_no_tool_calls(monkeypatch):
    """Workflow should complete when LLM returns a final assistant message."""

    async def fake_execute_activity(name: str, *args, **kwargs):  # noqa: D401 – test stub
        if name == "CreateRun":
            return str(uuid4())
        if name == "CreateRunStep":
            return str(uuid4())
        if name == "GetRunMemory":
            return AgentMemory(messages=[Message(role="user", content="hi")])
        if name == "LLMStreamPublish":
            # Assistant returns no tool calls – workflow should finish.
            return Message(role="assistant", content="done", tool_calls=None)
        if name == "FinalizeRun":
            return None
        raise RuntimeError(f"Unexpected activity name {name}")

    # Patch Temporal execute_activity helper so that all calls within the workflow
    # are routed to our stub implementation.
    monkeypatch.setattr(temporal_workflow, "execute_activity", fake_execute_activity, raising=True)

    wf = TemporalAgentExecutionWorkflow()
    input_payload = AgentWorkflowInput(
        session_id=str(uuid4()),
        user_message="Hello",
        run_id=None,
    )

    result = await wf.execute(input_payload)

    assert result.status == "completed"


@pytest.mark.asyncio
async def test_workflow_tool_execution(monkeypatch):
    """Workflow should execute tools then complete on subsequent LLM call."""

    call_counters = {"llm": 0, "execute_tool": 0}

    async def fake_execute_activity(name: str, *args, **kwargs):
        if name == "CreateRun":
            return str(uuid4())
        if name == "CreateRunStep":
            return str(uuid4())
        if name == "GetRunMemory":
            return AgentMemory(messages=[Message(role="user", content="hey")])
        if name == "LLMStreamPublish":
            call_counters["llm"] += 1
            if call_counters["llm"] == 1:
                tool_call = ToolCall(name="web_search", arguments={"query": "hey"})
                return Message(role="assistant", content=None, tool_calls=[tool_call])
            return Message(role="assistant", content="done", tool_calls=None)
        if name == "ExecuteTool":
            call_counters["execute_tool"] += 1
            # Activity is invoked with kwargs containing 'args': [tool_call]
            tool_call = kwargs.get("args", [])[0]
            return ToolCallResult(tool_call_id=tool_call.id, content="result")
        if name == "FinalizeRun":
            return None
        raise RuntimeError(f"Unexpected activity {name}")

    monkeypatch.setattr(temporal_workflow, "execute_activity", fake_execute_activity, raising=True)

    wf = TemporalAgentExecutionWorkflow()
    input_payload = AgentWorkflowInput(
        session_id=str(uuid4()),
        user_message="Hello",
        run_id=None,
    )

    result = await wf.execute(input_payload)

    assert result.status == "completed"
    # Ensure tool executed exactly once and LLM called twice
    assert call_counters == {"llm": 2, "execute_tool": 1}


@pytest.mark.asyncio
async def test_workflow_cancellation(monkeypatch):
    """Workflow should propagate cancellation when request_cancellation signal received."""

    async def fake_execute_activity(name: str, *args, **kwargs):
        # We only need to stub activities used prior to cancellation check.
        if name in {"CreateRun", "CreateRunStep"}:
            return str(uuid4())
        if name == "GetRunMemory":
            return AgentMemory(messages=[Message(role="user", content="hi")])
        # We will never reach LLM or FinalizeRun due to cancellation.
        return None

    monkeypatch.setattr(temporal_workflow, "execute_activity", fake_execute_activity, raising=True)

    wf = TemporalAgentExecutionWorkflow()
    wf.request_cancellation()  # simulate external signal before execution begins

    input_payload = AgentWorkflowInput(
        session_id=str(uuid4()),
        user_message="Hello",
        run_id=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await wf.execute(input_payload)

    # Ensure flag is set
    assert wf.cancellation_requested is True 
 