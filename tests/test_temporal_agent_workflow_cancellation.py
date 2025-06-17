import pytest
from uuid import uuid4

from temporalio.testing import WorkflowEnvironment
from temporalio import worker as _worker  # local import to avoid global deps
from temporalio import activity
from temporalio.exceptions import CancelledError

from truss.data_models import AgentWorkflowInput, Message, AgentMemory
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow


@pytest.mark.asyncio
async def test_workflow_cancellation_signal():
    """Workflow should transition to *Cancelled* when request_cancellation signal is sent."""

    finalized_statuses: list[str] = []

    @activity.defn(name="CreateRun")
    async def fake_create_run(session_id):  # noqa: D401 – test stub
        return str(uuid4())

    @activity.defn(name="CreateRunStep")
    async def fake_create_run_step(run_id, message):  # noqa: D401 – test stub
        return str(uuid4())

    @activity.defn(name="GetRunMemory")
    async def fake_get_run_memory(session_id):  # noqa: D401 – test stub
        # Provide minimal memory so workflow can reach reasoning loop.
        return AgentMemory(messages=[Message(role="user", content="hi")])

    @activity.defn(name="LLMStreamPublish")
    async def fake_llm_stream_publish(agent_config, messages, session_id, run_id):  # noqa: D401
        # This activity should never be reached due to early cancellation, but return a safe default.
        return Message(role="assistant", content="ignored", tool_calls=None)

    @activity.defn(name="FinalizeRun")
    async def fake_finalize_run(run_id, status, error_msg):  # noqa: D401
        finalized_statuses.append(status)


    env = await WorkflowEnvironment.start_time_skipping()

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
            session_id=str(uuid4()),
            user_message="cancel me",
        )

        handle = await env.client.start_workflow(
            TemporalAgentExecutionWorkflow.execute,
            input_payload,
            id=f"wf-{uuid4()}",
            task_queue="test-agent-queue",
        )

        # Send cancellation signal *after* workflow has started.
        await handle.signal("request_cancellation")

        # Expect the workflow result to raise WorkflowFailureError with a
        # CancelledError cause since Temporal wraps cancellations in a
        # failure wrapper on the client side.
        from temporalio.client import WorkflowFailureError

        with pytest.raises(WorkflowFailureError) as exc_info:
            await handle.result()

        assert isinstance(exc_info.value.cause, CancelledError)

    # Ensure FinalizeRun was invoked exactly once with 'cancelled' status.
    assert finalized_statuses == ["cancelled"]

    await env.shutdown() 
