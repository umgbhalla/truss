"""Temporal activity handling LLM streaming with real-time UI updates.

This module exposes :func:`llm_activity` which streams assistant responses
from LiteLLM **and** publishes every raw chunk to Redis so the frontend can
render incremental updates.  Actual accumulation of chunks, persistence and
heartbeat logic are implemented in follow-up subtasks – this iteration
focuses solely on the *Redis publishing* requirement (Task 5.2).
"""

from __future__ import annotations

import json
from uuid import UUID
from typing import List, Dict, Any

import anyio
import redis.asyncio as redis
from temporalio import activity
from truss.core.llm_client import stream_completion
from truss.core.mcp_client import default_manager
from truss.core.storage import PostgresStorage
from truss.data_models import Message, ToolCall, MCPClientConfig, MCPServerConfig

from truss.settings import get_settings
from anyio import to_thread
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = [
    "LLMActivities",
]


async def _get_redis_client() -> "redis.Redis":
    """Return an *async* Redis client built from application settings."""

    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=False)


class LLMActivities: 
    """Collection of LLM-related Temporal activities.

    The class is *instantiated* with external dependencies (e.g. database
    storage) so that unit-tests can inject fakes/mocks and production workers
    can share long-lived resources efficiently.  Only the
    :pymeth:`llm_activity` method is exposed as a Temporal activity; additional
    helpers can be added in the future without changing worker registration.
    """

    def __init__(self, storage: PostgresStorage | None = None) -> None: 
        self._storage: PostgresStorage | None = storage

    @staticmethod
    def _prune_incomplete_tool_calls(original: List[Message]) -> List[Message]:
        """Return a copy of *original* with invalid tool-call sequences removed.

        OpenAI's function-calling guidelines require every assistant message
        that includes ``tool_calls`` to be *immediately* followed by **tool**
        messages for *each* ``tool_call_id``.  When this invariant is
        violated – for instance after an agent crash or manual DB edits – the
        provider returns a *400* error similar to::

            An assistant message with 'tool_calls' must be followed by tool
            messages responding to each 'tool_call_id'.

        To keep the agent workflow robust we detect such orphaned assistant
        messages **before** forwarding the conversation to the LLM and drop
        them (together with all following messages) so that we only send a
        valid history.  The pruned messages remain untouched in storage – we
        merely trim the *working copy* for the current completion request.
        """

        cleaned: List[Message] = []
        i = 0

        while i < len(original):
            msg = original[i]

            if not (msg.role == "assistant" and msg.tool_calls):
                cleaned.append(msg)
                i += 1
                continue

            expected_ids = {tc.id for tc in (msg.tool_calls or [])}

            j = i + 1
            while j < len(original):
                nxt = original[j]

                if nxt.role != "tool":
                    break 

                call_id = getattr(nxt, "tool_call_id", None)
                if call_id in expected_ids:
                    expected_ids.remove(call_id)
                    j += 1
                else:
                    break 

                if not expected_ids:
                    break 

            if expected_ids:
                break

            cleaned.append(msg)
            cleaned.extend(original[i + 1 : j])
            i = j

        return cleaned


    @activity.defn(name="LLMStreamPublish")
    async def llm_activity(  
        self,
        agent_config: dict[str, Any],
        tools_payload: List[dict[str, Any]],
        messages: List[Message],
        session_id: UUID,
        run_id: UUID,  
    ) -> Message:  
        """Stream assistant response and publish deltas to Redis.

        This implementation is functionally identical to the previous
        `llm_activity` function but operates as an *instance method* so it can
        leverage injected dependencies (e.g. *storage*).
        """

        assert agent_config is not None
        logger.info("llm_activity agent_config")
        logger.info(agent_config)
        redis_client = await _get_redis_client()

        if self._storage is None:
            self._storage = PostgresStorage.from_database_url(get_settings().database_url)

        try:

            if agent_config.get("system_prompt", None):
                if not messages or messages[0].role != "system":
                    messages.insert(0, Message(role="system", content=agent_config["system_prompt"]))

            safe_messages = self._prune_incomplete_tool_calls(messages)

            chunk_stream = await stream_completion(
                agent_config=agent_config,
                conversation=safe_messages,
                tools_payload=tools_payload,
            )

         
            channel = f"stream:{session_id}"
            full_content: List[str] = [] 

            stop_key = f"stop:{session_id}"
            stop_requested = False  

            _tool_buffers: Dict[str, Dict[str, Any]] = {}
            _tool_call_order: List[str] = [] 
            _index_to_id: Dict[int, str] = {}

           
            def _get(obj: Any, attr: str, default: Any = None): 
                if isinstance(obj, dict):
                    return obj.get(attr, default)
                return getattr(obj, attr, default)

            final_message: Message | None = None

            async for chunk in chunk_stream:
                try:
                    encoded = json.dumps(
                        chunk,
                        default=lambda o: getattr(o, "__dict__", str(o)),
                        ensure_ascii=False,
                    )
                except (TypeError, ValueError): 
                    encoded = json.dumps(str(chunk), ensure_ascii=False)

                with anyio.CancelScope(shield=True):
                    await redis_client.publish(channel, encoded)

                if chunk is None:
                    continue

                choices = _get(chunk, "choices")
                if not choices:
                    continue

                delta = _get(choices[0], "delta") 
                if not delta:
                    continue

                content_part = _get(delta, "content")
                if content_part:
                    full_content.append(str(content_part))

                tool_calls_delta = _get(delta, "tool_calls")
                if tool_calls_delta:
                    for tc_delta in tool_calls_delta:
                        idx = _get(tc_delta, "index", 0)
                        call_id = _get(tc_delta, "id")

                        if call_id is None:
                            call_id = _index_to_id.get(idx)
                        else:
                            _index_to_id[idx] = call_id

                        if call_id is None:
                            call_id = f"call_{idx}"

                        if call_id not in _tool_buffers:
                            _tool_buffers[call_id] = {"name": None, "arguments_parts": []}
                            _tool_call_order.append(call_id)

                        buf = _tool_buffers[call_id]

                        function_obj = _get(tc_delta, "function")
                        if function_obj:
                            name_val = _get(function_obj, "name")
                            if name_val:
                                buf["name"] = name_val

                            arg_part = _get(function_obj, "arguments")
                            if arg_part is not None:
                                buf["arguments_parts"].append(str(arg_part))

                if not stop_requested:
                    try:
                        if await redis_client.exists(stop_key):
                            if not tool_calls_delta:
                                stop_requested = True 
                                break 
                    except Exception:  
                        activity.logger.warning(
                            "Failed to check stop flag for %s", session_id, exc_info=True
                        )

            if stop_requested:
                with anyio.CancelScope(shield=True):
                    try:
                        await redis_client.delete(stop_key)
                    except Exception:  
                        activity.logger.warning(
                            "Failed to delete stop flag for %s", session_id, exc_info=True
                        )

            tool_calls_final: List[ToolCall] | None = None
            if _tool_call_order:
                tool_calls_final = []
                for call_id in _tool_call_order:
                    data = _tool_buffers[call_id]
                    name_val: str = data.get("name") or ""
                    args_str: str = "".join(data.get("arguments_parts", []))
                    try:
                        args_dict = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args_dict = {"_raw": args_str}

                    tool_calls_final.append(
                        ToolCall(id=call_id, name=name_val, arguments=args_dict)
                    )

            final_message = Message(
                role="assistant",
                content="".join(full_content) if full_content else None,
                tool_calls=tool_calls_final,
            )


            if stop_requested:
                from truss.core.models.run import RunStatus 

                await to_thread.run_sync(
                    self._storage.update_run_status,  
                    run_id,
                    RunStatus.CANCELLED,
                    None,
                    cancellable=True,
                )

        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await redis_client.aclose()
                except Exception:  
                    activity.logger.warning("Failed to close Redis client", exc_info=True)

        return final_message




