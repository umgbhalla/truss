import os
from fastapi.testclient import TestClient

# Configure test environment – skip Temporal connection and use SQLite
os.environ["SKIP_TEMPORAL_CONNECTION"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from truss.api import app  # noqa: E402
from truss.api import main as api_main  # noqa: E402


class _FakeWorkflowHandle:  # noqa: D401 – simple stub for Temporal handle
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.last_signal = None

    async def signal(self, name: str, *args):  # noqa: D401 – stub async method
        # Record last signal call for assertion if needed
        self.last_signal = (name, args)
        return None


class _FakeTemporalClient:  # noqa: D401 – stub Temporal client
    def __init__(self):
        self.handles: dict[str, _FakeWorkflowHandle] = {}

    def get_workflow_handle(self, workflow_id: str):  # noqa: D401 – sync in SDK
        handle = self.handles.get(workflow_id) or _FakeWorkflowHandle(workflow_id)
        self.handles[workflow_id] = handle
        return handle

    async def close(self):  # noqa: D401 – required by shutdown event
        return None


def _get_test_client():
    """Return TestClient for FastAPI app with fake Temporal client injected."""
    return TestClient(app)


def test_send_signal_success():
    client = _get_test_client()
    # Inject fake client *after* startup event to avoid being overwritten
    api_main._temporal_client = _FakeTemporalClient()

    workflow_id = "wf-123"
    resp = client.post(f"/workflows/{workflow_id}/signal/request_cancellation", json={"data": "now"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "signal sent"

    # Ensure the fake handle recorded the signal invocation
    handle = api_main._temporal_client.get_workflow_handle(workflow_id)
    assert handle.last_signal == ("request_cancellation", ("now",)) 
