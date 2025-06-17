import os
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

# Configure env
os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from truss.api import app  # noqa: E402
from truss.api import main as api_main  # noqa: E402
from truss.api.main import get_storage  # noqa: E402
from truss.core.models.agent_config import AgentConfigORM  # noqa: E402
from truss.api.main import create_session  # noqa: F401 imported for type hints


class _FakeTemporalClient:  # noqa: D401 – simple stub
    async def start_workflow(self, *_, **__):
        return None

    async def close(self):
        return None


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        # Re-inject fake client after startup overwrite
        api_main._temporal_client = _FakeTemporalClient()

        storage = get_storage()
        # Seed AgentConfig + Session via direct storage & endpoint
        with storage._session_scope() as session:  # type: ignore[attr-defined]
            ac = AgentConfigORM(
                name="Test Agent",
                system_prompt="You are helpful",
                llm_config={"model_name": "gpt-4o"},
                tools=[],
            )
            session.add(ac)
            session.flush()
            agent_id = ac.id

        # create session via API
        resp = c.post("/sessions", json={"agent_id": str(agent_id), "user_id": "user-1"})
        session_id = resp.json()["session_id"]
        yield c, session_id


@pytest.mark.asyncio
async def test_start_run_success(client):
    c, session_id = client
    resp = c.post(f"/sessions/{session_id}/runs", json={"message": "hello"})
    assert resp.status_code == 202
    data = resp.json()
    assert UUID(data["workflow_id"])  # ensure valid UUID string
