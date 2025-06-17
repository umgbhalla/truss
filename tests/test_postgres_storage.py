import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


import sqlalchemy as sa
from sqlalchemy.orm import Session

from truss.core.models.base import Base
from truss.core.models.agent_config import AgentConfigORM
from truss.core.models.run_session import RunSessionORM
from truss.data_models import Message
from truss.core.storage import PostgresStorage
from truss.core.models.run import RunStatus


def _make_storage() -> PostgresStorage:
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PostgresStorage(engine)


def _seed_prereqs(storage: PostgresStorage):
    # Insert AgentConfig + RunSession so we can create runs
    with Session(storage._engine) as session:  # type: ignore[attr-defined]
        ac = AgentConfigORM(
            name="Seed Agent",
            system_prompt="You are helpful",
            llm_config={"model_name": "gpt-4o"},
            tools=[],
        )
        session.add(ac)
        session.commit()

        rs = RunSessionORM(agent_config_id=ac.id, user_id="user-abc")
        session.add(rs)
        session.commit()
        # Capture primary-key values before session closes
        ac_id = ac.id
        rs_id = rs.id
        return ac_id, rs_id


def test_storage_run_lifecycle():
    storage = _make_storage()
    ac, rs = _seed_prereqs(storage)

    # Create run
    run = storage.create_run(session_id=rs)
    assert run.session_id == rs
    assert run.status == RunStatus.PENDING

    # Persist message
    msg = Message(role="assistant", content="Hi")
    step = storage.create_run_step_from_message(run.id, msg)
    assert step.run_id == run.id
    assert step.content == "Hi"

    # Retrieve memory
    memory = storage.get_steps_for_session(rs)
    assert len(memory) == 1

    # Update status
    storage.update_run_status(run.id, RunStatus.SUCCEEDED, None)
    with Session(storage._engine) as session:  # type: ignore[attr-defined]
        refreshed = session.get(type(run), run.id)
        assert refreshed.status == RunStatus.SUCCEEDED

    # Load AgentConfig
    cfg = storage.load_agent_config(ac)
    assert cfg.name == "Seed Agent" 
