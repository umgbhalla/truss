import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from sqlalchemy.orm import Session

from truss.core.models.base import Base
from truss.core.models.agent_config import AgentConfigORM
from truss.core.models.run_session import RunSessionORM
from truss.core.models.run import RunORM, RunStatus


def _create_engine() -> sa.Engine:
    return sa.create_engine("sqlite:///:memory:")


def test_run_table_create_and_insert():
    engine = _create_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Insert prerequisite AgentConfig and RunSession
        ac = AgentConfigORM(
            name="Agent X",
            system_prompt="You are X",
            llm_config={"model": "gpt-4o"},
            tools=[],
        )
        session.add(ac)
        session.commit()

        rs = RunSessionORM(agent_config_id=ac.id, user_id="user-321")
        session.add(rs)
        session.commit()

        run = RunORM(session_id=rs.id, status=RunStatus.RUNNING)
        session.add(run)
        session.commit()

        fetched = session.get(RunORM, run.id)
        assert fetched is not None
        assert fetched.session_id == rs.id
        assert fetched.status == RunStatus.RUNNING

    Base.metadata.drop_all(engine) 
