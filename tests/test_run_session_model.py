import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from sqlalchemy.orm import Session

from truss.core.models.base import Base
from truss.core.models.agent_config import AgentConfigORM
from truss.core.models.run_session import RunSessionORM


def _create_engine() -> sa.Engine:
    # SQLite is sufficient for unit tests; we don't get FK constraint enforced
    return sa.create_engine("sqlite:///:memory:")


def test_run_session_table_create_and_insert():
    engine = _create_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # First insert an AgentConfig to satisfy FK relationship
        ac = AgentConfigORM(
            name="Test Agent",
            system_prompt="You are helpful",
            llm_config={"model_name": "gpt-4o", "temperature": 0.7},
            tools=["search"],
        )
        session.add(ac)
        session.commit()

        rs = RunSessionORM(agent_config_id=ac.id, user_id="user-123")
        session.add(rs)
        session.commit()

        fetched = session.get(RunSessionORM, rs.id)
        assert fetched is not None
        assert str(fetched.agent_config_id) == str(ac.id)
        assert fetched.user_id == "user-123"

    Base.metadata.drop_all(engine) 
