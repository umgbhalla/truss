import json
from typing import AsyncIterator, Dict, Any, List
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from truss.activities.llm_activities import LLMActivities
from truss.data_models import AgentConfig, LLMConfig, Message

# Stub PostgresStorage for tests to avoid hitting a real database


class _DummyStorage:  # noqa: D101 – test helper
    def create_run_step_from_message(self, run_id, message):  # noqa: D401
        # Simply record that the method was called; could attach to self later if needed.
        self.called_with = (run_id, message)
        return None


class _FakeRedis:  # noqa: D101 – test helper
    def __init__(self) -> None:  # noqa: D401
        self.published: List[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:  # noqa: D401
        self.published.append((channel, message))

    async def aclose(self) -> None:  # noqa: D401
        # Nothing to clean
        pass


async def _dummy_stream() -> AsyncIterator[Dict[str, Any]]:  # noqa: D401 – helper
    yield {"choices": [{"delta": {"content": "Hi"}}]}
    yield {"choices": [{"delta": {"content": " there"}}]}


@pytest.mark.asyncio
async def test_llm_activity_publishes_chunks_to_redis() -> None:  # noqa: D401 – imperative
    session_id = uuid4()
    run_id = uuid4()

    llm_cfg = LLMConfig(model_name="gpt-4o", temperature=0.0)
    agent_cfg = AgentConfig(name="Test", system_prompt="Hi", llm_config=llm_cfg)
    msgs = [Message(role="user", content="Hello")]

    fake_redis = _FakeRedis()

    dummy_storage = _DummyStorage()

    with (
        patch("truss.activities.llm_activities._get_redis_client", new=AsyncMock(return_value=fake_redis)),
        patch("truss.activities.llm_activities.stream_completion", new=AsyncMock(return_value=_dummy_stream())),
    ):
        llm_act_instance = LLMActivities(storage=dummy_storage)
        await llm_act_instance.llm_activity(agent_cfg, msgs, session_id, run_id)  # type: ignore[arg-type]

    # Two chunks should have been published
    assert len(fake_redis.published) == 2
    expected_channel = f"stream:{session_id}"
    assert all(channel == expected_channel for channel, _ in fake_redis.published)

    # Verify payloads are JSON-encoded strings matching the original chunks
    decoded = [json.loads(payload) for _, payload in fake_redis.published]
    assert decoded == [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
    ] 
