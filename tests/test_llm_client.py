from typing import List, Dict, Any, AsyncIterator

import pytest
from unittest.mock import AsyncMock, patch

from truss.core.llm_client import stream_completion
from truss.data_models import AgentConfig, LLMConfig, Message


async def _dummy_stream() -> AsyncIterator[Dict[str, Any]]:  # noqa: D401 – helper
    """Yield a fixed sequence of fake chunks mimicking provider stream."""
    yield {"choices": [{"delta": {"content": "Hello"}}]}
    yield {"choices": [{"delta": {"content": " world"}}]}


@pytest.mark.asyncio
async def test_stream_completion_wrapper_sends_correct_params() -> None:
    llm_cfg = LLMConfig(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=256,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.0,
    )
    agent_cfg = AgentConfig(
        name="TestAgent",
        system_prompt="You are a helpful assistant.",
        llm_config=llm_cfg,
    )
    conversation: List[Message] = [
        Message(role="user", content="Say hello"),
    ]

    # We'll monkeypatch litellm.acompletion so no external API call happens.
    with patch("truss.core.llm_client.litellm.acompletion", new=AsyncMock()) as mocked:
        mocked.return_value = _dummy_stream()


        stream = await stream_completion(agent_config=agent_cfg, conversation=conversation)

        mocked.assert_awaited_once()
        kwargs = mocked.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 256
        assert kwargs["top_p"] == 0.95
        assert kwargs["stream"] is True
        # Verify messages payload transformation – ``tool_calls`` and
        # ``tool_call_id`` keys should be *absent* when not applicable to
        # remain compliant with the OpenAI schema.
        assert kwargs["messages"] == [
            {
                "role": "user",
                "content": "Say hello",
            }
        ]

        # Collect streamed chunks to ensure generator works
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ] 
