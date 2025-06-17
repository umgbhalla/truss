import os

import pytest
from fastapi.testclient import TestClient

# Set flag to skip Temporal connection during tests *before* importing app
os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"

from truss.api import app  # noqa: E402  must import after env var


@pytest.fixture(scope="module")
def client():
    """Return a configured TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """The /health endpoint should return 200 OK with expected payload."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"} 
