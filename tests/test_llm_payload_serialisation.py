from typing import AsyncIterator, Dict, Any, List

import pytest
from unittest.mock import AsyncMock, patch

from truss.core.llm_client import stream_completion
from truss.data_models import AgentConfig, LLMConfig, Message


async def _noop_stream() -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover
    if False:
        yield None  # satisfying async iterator


@pytest.mark.asyncio
async def test_llm_client_serialises_dict_content() -> None:  # noqa: D401
    """Ensure message.content values that are *dict* are JSON-serialised."""

    agent_cfg = AgentConfig(
        name="Serialiser",
        system_prompt="system",
        llm_config=LLMConfig(model_name="openai/gpt-4.1-mini"),
    )

    conversation: List[Message] = [
        Message(role="tool", content={"foo": "bar"}, tool_call_id="call_123"),
    ]

    with patch("truss.core.llm_client.litellm.acompletion", new=AsyncMock()) as mocked:
        mocked.return_value = _noop_stream()
        _ = await stream_completion(agent_config=agent_cfg, conversation=conversation)

        mocked.assert_awaited_once()
        payload_msgs = mocked.call_args.kwargs["messages"]
        assert payload_msgs == [
            {
                "role": "tool",
                "content": "{\"foo\":\"bar\"}",
                "tool_call_id": "call_123",
            }
        ] 
