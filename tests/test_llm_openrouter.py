from typing import Dict, Any, List

import pytest

from truss.data_models import LLMConfig, AgentConfig, Message
from truss.core import llm_client as llm_mod


@pytest.mark.asyncio
async def test_llm_client_forwards_openrouter_params(monkeypatch):
    """LLM client should forward api_key and base_url when OPENROUTER_API_KEY set."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "testkey")

    captured_params: Dict[str, Any] = {}

    async def _fake_acompletion(**kwargs):  # type: ignore[return-type]
        nonlocal captured_params
        captured_params = kwargs
        # Return async generator that yields nothing
        async def _gen():
            if False:
                yield None  # pragma: no cover
        return _gen()

    monkeypatch.setattr(llm_mod.litellm, "acompletion", _fake_acompletion)

    cfg = AgentConfig(
        name="TestAgent",
        system_prompt="You are helpful",
        llm_config=LLMConfig(
            model_name="openai/gpt-4.1-mini",
            base_url="https://openrouter.ai/api/v1",
        ),
        tools=[],
    )

    conversation: List[Message] = [Message(role="user", content="hi")]

    # Execute and exhaust generator
    gen = await llm_mod.stream_completion(agent_config=cfg, conversation=conversation)
    async for _ in gen:
        pass

    assert captured_params["api_key"] == "testkey"
    assert captured_params["base_url"] == "https://openrouter.ai/api/v1"
    assert captured_params["model"] == "openai/gpt-4.1-mini" 
