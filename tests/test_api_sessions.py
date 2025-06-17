import os

import pytest
from fastapi.testclient import TestClient

# Skip Temporal connection & use SQLite in-memory DB for storage
os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from truss.api import app  # noqa: E402
from truss.core.models.agent_config import AgentConfigORM  # noqa: E402
from truss.api.main import get_storage  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Provide TestClient with preloaded AgentConfig row."""

    with TestClient(app) as c:
        # Startup events have run; storage is initialised
        storage = get_storage()
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

        yield c, str(agent_id)


def test_create_session_success(client):
    client_obj, agent_id = client

    payload = {"agent_id": agent_id, "user_id": "user-xyz"}
    resp = client_obj.post("/sessions", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert "session_id" in body
    assert len(body["session_id"]) > 0 
