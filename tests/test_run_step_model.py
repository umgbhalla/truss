import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from sqlalchemy.orm import Session

from truss.core.models.base import Base
from truss.core.models.agent_config import AgentConfigORM
from truss.core.models.run_session import RunSessionORM
from truss.core.models.run import RunORM, RunStatus
from truss.core.models.run_step import RunStepORM, MessageRole


def _create_engine() -> sa.Engine:
    # SQLite is sufficient for unit tests; JSON will map to TEXT
    return sa.create_engine("sqlite:///:memory:")


def test_run_step_table_create_and_insert():
    engine = _create_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Insert prerequisite objects
        ac = AgentConfigORM(
            name="Agent Test",
            system_prompt="You are helpful",
            llm_config={"model": "gpt-4o"},
            tools=[{"name": "search"}],
        )
        session.add(ac)
        session.commit()

        rs = RunSessionORM(agent_config_id=ac.id, user_id="user-xyz")
        session.add(rs)
        session.commit()

        run = RunORM(session_id=rs.id, status=RunStatus.RUNNING)
        session.add(run)
        session.commit()

        step = RunStepORM(
            run_id=run.id,
            role=MessageRole.ASSISTANT,
            content="Hello, how can I help you?",
            tool_calls=[{"id": "tc1", "name": "search", "arguments": "..."}],
        )
        session.add(step)
        session.commit()

        fetched = session.get(RunStepORM, step.id)
        assert fetched is not None
        assert fetched.run_id == run.id
        assert fetched.role == MessageRole.ASSISTANT
        assert fetched.content.startswith("Hello")

    Base.metadata.drop_all(engine) 
