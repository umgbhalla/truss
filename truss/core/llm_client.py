"""Async wrapper around LiteLLM for streaming chat completions.

The helper exposed here converts our internal Pydantic data models to the
JSON shape expected by `litellm.acompletion` and forwards configurable
parameters extracted from :class:`truss.data_models.AgentConfig`.

The public coroutine :func:`stream_completion` deliberately returns the
_async generator_ produced by `litellm.acompletion` unchanged so callers
can iterate over the provider's tokens/chunks directly.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Dict, Any, List
import json
import litellm  # restored dependency

from truss.data_models import AgentConfig, Message

__all__ = [
    "stream_completion",
]
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def _build_messages_payload(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert internal :class:`Message` objects to LiteLLM JSON payload.

    LiteLLM expects a list of dictionaries with at minimum the keys
    ``role`` and ``content``.  We currently ignore tool call fields here –
    they will be handled at a higher level once we implement tool calling
    support in the streaming activity.
    """

    payload: List[Dict[str, Any]] = []

    for msg in messages:
        # Ensure ``content`` is always a *string* as required by OpenAI.
        content_val: Any = msg.content
        if isinstance(content_val, (dict, list)):
            # Serialise non-string payloads deterministically so downstream
            # tests can assert against a stable representation.
            content_val = json.dumps(content_val, separators=(",", ":"))
        elif content_val is None:
            content_val = ""

        item: Dict[str, Any] = {
            "role": msg.role,
            "content": content_val,
            # ``tool_calls`` and ``tool_call_id`` are included *only* when
            # present to remain compliant with the OpenAI schema. Some
            # providers reject explicit null values.
        }

        # Convert internal ToolCall objects to OpenAI-compatible shape when present.
        if msg.tool_calls:
            tc_payload: List[Dict[str, Any]] = []
            for tc in msg.tool_calls:
                tc_payload.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # OpenAI expects *stringified* JSON for arguments
                        "arguments": json.dumps(tc.arguments or {}),
                    },
                })
            item["tool_calls"] = tc_payload

        if msg.tool_call_id is not None:
            item["tool_call_id"] = msg.tool_call_id

        payload.append(item)

    return payload


async def stream_completion(
    *,
    agent_config: AgentConfig | Dict[str, Any],
    conversation: List[Message],
    tools_payload: List[Dict[str, Any]] | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Return an async iterator yielding streaming completion chunks.

    Parameters
    agent_config
        The agent configuration containing LLM parameters to forward.
    conversation
        Ordered list of user/assistant/system messages to include in the
        completion request.
    """
    # Accept both a fully parsed ``AgentConfig`` instance as well as the raw
    # ``dict`` representation that may be passed across process boundaries
    # (e.g. when coming from a JSON payload or the Temporal workflow input).
    # Converting eagerly here ensures we can rely on attribute access for the
    # remainder of the function and keeps the public interface unchanged for
    # internal callers that already pass in a proper model instance.

    if isinstance(agent_config, dict):
        agent_config = AgentConfig(**agent_config)

    # Pull out the nested LLMConfig for convenience
    llm_conf = agent_config.llm_config

    # Build parameter dict conditionally so we don't send ``None`` values
    # for optional parameters which some providers reject.
    params: Dict[str, Any] = {
        "model": llm_conf.model_name,
        "temperature": llm_conf.temperature,
        "top_p": llm_conf.top_p,
        "frequency_penalty": llm_conf.frequency_penalty,
        "presence_penalty": llm_conf.presence_penalty,
        "stream": True,
        # Messages converted to provider format
        "messages": _build_messages_payload(conversation),
    }
    if llm_conf.max_tokens is not None:
        params["max_tokens"] = llm_conf.max_tokens

    # Provider overrides – OpenRouter integration
    api_key_env = os.getenv("OPENROUTER_API_KEY")
    if api_key_env:
        # LiteLLM expects api_key/base_url params forwarded explicitly.
        params["api_key"] = api_key_env
        params.setdefault("base_url", "https://openrouter.ai/api/v1")

    # Config-level base_url takes precedence if provided
    if llm_conf.base_url is not None:
        params["base_url"] = llm_conf.base_url

    # Remote MCP tools can be injected by callers via *tools_payload*.
    if tools_payload:
        params["tools"] = tools_payload
        params["tool_choice"] = "auto"

    # Delegate the heavy lifting to LiteLLM – it returns an *async generator*
    # when ``stream=True`` so we simply forward that upstream. Any network
    # failures bubble up to the calling activity where Temporal retry
    # policies apply.
    # litellm.set_verbose = True
    logger.info("llm_client params")
    logger.info(params)
    return await litellm.acompletion(**params)
