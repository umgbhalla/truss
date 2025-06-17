from typing import AsyncIterator, Dict, Any, List
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from truss.activities.llm_activities import LLMActivities
from truss.data_models import AgentConfig, LLMConfig, Message, ToolCall


class _FakeRedis:  # noqa: D101 – test helper
    def __init__(self) -> None:  # noqa: D401
        self.published: List[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:  # noqa: D401
        self.published.append((channel, message))

    async def aclose(self) -> None:  # noqa: D401
        pass


async def _dummy_stream() -> AsyncIterator[Dict[str, Any]]:  # noqa: D401 – helper
    # Two-part tool call arguments – OpenAI style
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"San"
                            },
                        }
                    ]
                }
            }
        ]
    }
    yield {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "arguments": " Francisco\"}"
                            },
                        }
                    ]
                }
            }
        ]
    }


class _DummyStorage:  # noqa: D101 – test helper
    def create_run_step_from_message(self, run_id, message):  # noqa: D401
        self.called_with = (run_id, message)
        return None


@pytest.mark.asyncio
async def test_llm_activity_accumulates_tool_calls() -> None:  # noqa: D401
    session_id = uuid4()
    run_id = uuid4()

    agent_cfg = AgentConfig(
        name="ToolTester",
        system_prompt="Hi",
        llm_config=LLMConfig(model_name="gpt-4o", temperature=0.0),
    )
    msgs = [Message(role="user", content="weather?")]

    fake_redis = _FakeRedis()
    dummy_storage = _DummyStorage()

    with (
        patch("truss.activities.llm_activities._get_redis_client", new=AsyncMock(return_value=fake_redis)),
        patch("truss.activities.llm_activities.stream_completion", new=AsyncMock(return_value=_dummy_stream())),
    ):
        act = LLMActivities(storage=dummy_storage)
        final_msg = await act.llm_activity(agent_cfg, msgs, session_id, run_id)  # type: ignore[arg-type]

    assert final_msg.tool_calls is not None
    assert len(final_msg.tool_calls) == 1
    tool_call: ToolCall = final_msg.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "San Francisco"} 
