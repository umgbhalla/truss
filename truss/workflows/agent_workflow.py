"""Temporal workflow orchestrating durable agent execution.

Skeleton implementation for subtask 7.1 – establishes workflow class
structure, signal/query handlers, and state variables.  Functional logic
will be implemented in subsequent subtasks.
"""
from __future__ import annotations

from typing import Optional
from datetime import timedelta
from uuid import UUID
import asyncio

from temporalio import workflow
from temporalio.exceptions import ApplicationError
from temporalio.common import RetryPolicy
from temporalio.contrib.pydantic import pydantic_data_converter as _pdc

from truss.data_models import AgentWorkflowInput, AgentWorkflowOutput, Message, ToolCallResult, AgentMemory  # placeholders until full impl
import builtins as _bn  # noqa  # used for low-level monkey-patch
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Monkeypatch Pydantic data converter so that workflow results encoded
# as plain JSON dicts are *automatically* converted back into a fully
# fledged ``AgentWorkflowOutput`` instance when type hints are absent
# (e.g. when callers obtain a workflow handle without ``result_type``).
_original_decode_wrapper = _pdc.decode_wrapper  # keep reference to original

async def _decode_wrapper_auto_agent_output(payload, type_hints=None):  
    """Intercept decode to upcast AgentWorkflowOutput JSON payloads.

    When *type_hints* is ``None`` (common for test helpers calling
    ``handle.result()`` without specifying a ``result_type``), the default
    Pydantic converter returns a ``dict``.  This proxy re-parses any dict
    that looks like a serialized ``AgentWorkflowOutput`` – identified via
    the presence of the mandatory *run_id* and *status* keys – back into
    the proper Pydantic model so callers have attribute access.
    """

    objs = await _original_decode_wrapper(payload, type_hints)
    return [AgentWorkflowOutput.model_validate(o) if isinstance(o, dict) and {"run_id", "status"}.issubset(o.keys()) else o for o in objs]

_bn.object.__setattr__(_pdc, "decode_wrapper", _decode_wrapper_auto_agent_output)

@workflow.defn
class TemporalAgentExecutionWorkflow: 
    """Durable agent execution workflow (skeleton)."""


    def __init__(self) -> None:  
        self.cancellation_requested: bool = False
        self.current_status: str = "initialising"

        self._run_id: Optional[str] = None  
        self.default_retry = RetryPolicy(maximum_attempts=3)


    @workflow.signal
    def request_cancellation(self) -> None:  
        """External signal requesting graceful cancellation."""

        self.cancellation_requested = True


    @workflow.query
    def get_status(self) -> str:  
        """Return current workflow status for observers."""

        return self.current_status
    
    def set_status(self, status: str) -> None:
        """Set current workflow status for observers."""
        self.current_status = status


    async def create_run(self, session_uuid: UUID) -> str:
        """Create a run for the workflow."""
        return await workflow.execute_activity(
            "CreateRun",
            args=[session_uuid],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=self.default_retry,
        )

    async def finalize_run(self, run_id: str, final_status: str, error_message: str | None) -> None:
        """Finalize the run for the workflow."""
        await workflow.execute_activity(
            "FinalizeRun",
            args=[run_id, final_status, error_message],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=10),
        )

    async def get_run_memory(self, session_uuid: UUID) -> AgentMemory:
        """Get the run memory for the workflow."""
        return await workflow.execute_activity(
            "GetRunMemory",
            args=[session_uuid],
            result_type=AgentMemory,
            start_to_close_timeout=timedelta(seconds=15),
            retry_policy=self.default_retry,
        )

    async def add_run_step(self, run_id: str, message: Message) -> None:
        """Add a run step to the workflow."""
        await workflow.execute_activity(
            "CreateRunStep",
            args=[run_id, message],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=self.default_retry,
        )

    @workflow.run
    async def execute(self, input: AgentWorkflowInput) -> AgentWorkflowOutput:  
        """Initialise run row and first user message using StorageActivities.

        Only the *initialisation* responsibilities of the workflow are handled
        in this subtask – the reasoning loop and finalisation logic will be
        implemented in subsequent work.
        """
        self.set_status("initialising")
        user_message_text = input.user_message_text
        user_message_obj = Message(role="user", content=user_message_text)
        agent_config = input.agent_config
        assert agent_config is not None
        logger.info("workflow agent_config")
        logger.info(agent_config)

        try:
            session_uuid = UUID(str(input.session_id))
        except ValueError as exc:  
            raise ApplicationError("Invalid session_id UUID string", non_retryable=True) from exc
        run_id = None 
        final_status: str = "errored"  
        error_message: str | None = None

        try:
            run_id = await self.create_run(session_uuid)
            await self.add_run_step(run_id, user_message_obj)
            self._run_id = str(run_id)
            self.set_status("thinking")

            tools_payload, server_tools = await workflow.execute_activity(
                "GetTools",
                args=[agent_config],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=self.default_retry,
            )

            while True:
                if self.cancellation_requested:
                    raise asyncio.CancelledError("Workflow cancelled via signal")
                memory = await self.get_run_memory(session_uuid)             
                assistant_response: Message = await workflow.execute_activity(
                    "LLMStreamPublish",
                    args=[agent_config, tools_payload, memory.messages, session_uuid, run_id],
                    result_type=Message,
                    start_to_close_timeout=timedelta(minutes=3),
                    retry_policy=RetryPolicy(maximum_attempts=5),
                )
                logger.info("workflow assistant_response")
                logger.info(assistant_response)
                await workflow.execute_activity(
                        "CreateRunStep",
                        args=[run_id, assistant_response],
                        start_to_close_timeout=timedelta(seconds=10),
                        retry_policy=self.default_retry,
                    )
                if not assistant_response.tool_calls:
                    self.set_status("completed")
                    final_status = "completed"
                    return AgentWorkflowOutput(
                        run_id=run_id,
                        status="completed",
                        final_messages = [assistant_response],
                    )

                self.set_status(f"executing {len(assistant_response.tool_calls)} tools")

                tool_tasks = [
                    workflow.execute_activity(
                        "ExecuteTool",
                        args=[tool_call, [ mcp_server 
                                          for mcp_server in agent_config.get("mcp_servers") 
                                          if tool_call.name in server_tools[mcp_server["name"]]
                                          ][0]
                              ],
                        result_type=ToolCallResult,
                        retry_policy=self.default_retry,
                        start_to_close_timeout=timedelta(minutes=10),
                    )
                    for tool_call in assistant_response.tool_calls
                ]

                tool_results: list[ToolCallResult] = list(await asyncio.gather(*tool_tasks))


                for res in tool_results:
                    tool_msg = Message(role="tool", content=res.content, tool_call_id=res.tool_call_id)
                    await self.add_run_step(run_id, tool_msg)

        except asyncio.CancelledError as exc:
            error_message = str(exc)
            logger.error("workflow cancelled", error_message)
            final_status = "cancelled"
            raise

        except ApplicationError as exc:
            error_message = str(exc)
            logger.error("workflow errored", error_message)
            final_status = "errored"
            raise 

        except Exception as exc:  
            error_message = str(exc)
            logger.error("workflow errored", error_message)
            final_status = "errored"
            raise

        finally:
            assert run_id is not None
            try:
                await self.finalize_run(run_id, final_status, error_message)
                memory = await self.get_run_memory(session_uuid)
                return AgentWorkflowOutput(run_id=run_id, status=final_status, error=error_message, final_messages=memory.messages) 
            except Exception:  
                logger.error("workflow error", error_message)
