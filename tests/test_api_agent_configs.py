import os
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

# Setup environment to avoid Temporal connection and use SQLite
os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from truss.api import app  # noqa: E402
from truss.api import main as api_main  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient with clean database for each module run."""

    # Ensure storage initialised and in clean state
    with TestClient(app) as c:
        # Inject fake Temporal client so routes that might access it don't fail
        class _FakeTemporalClient:  # noqa: D401 – minimal stub
            async def close(self):
                return None

        api_main._temporal_client = _FakeTemporalClient()
        yield c


def _create_agent_config_payload():
    return {
        "name": "TestAgent",
        "system_prompt": "You are helpful",
        "llm_config": {"model_name": "gpt-4o"},
        "tools": ["web_search"],
    }


def test_create_agent_config(client):
    payload = _create_agent_config_payload()
    resp = client.post("/agent-configs", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data and UUID(data["id"])  # valid UUID
    for key in ("name", "system_prompt", "llm_config", "tools"):
        assert data[key] == payload[key]

    agent_id = data["id"]

    # List endpoint should include the newly created config
    resp = client.get("/agent-configs")
    assert resp.status_code == 200
    configs = resp.json()
    assert any(cfg["id"] == agent_id for cfg in configs)

    # Retrieve endpoint returns the exact config
    resp = client.get(f"/agent-configs/{agent_id}")
    assert resp.status_code == 200
    retrieved = resp.json()
    assert retrieved == data

    # Delete endpoint removes the config
    resp = client.delete(f"/agent-configs/{agent_id}")
    assert resp.status_code == 204

    # Subsequent GET should return 404
    resp = client.get(f"/agent-configs/{agent_id}")
    assert resp.status_code == 404 
