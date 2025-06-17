import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from sqlalchemy.orm import Session

from truss.core.models.base import Base
from truss.core.models.agent_config import AgentConfigORM


def _create_engine() -> sa.Engine:
    return sa.create_engine("sqlite:///:memory:")


def test_agent_config_table_create_and_insert():
    engine = _create_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        ac = AgentConfigORM(
            name="Test Agent",
            system_prompt="You are helpful",
            llm_config={"model_name": "gpt-4o", "temperature": 0.7},
            tools=["search"],
        )
        session.add(ac)
        session.commit()

        fetched = session.get(AgentConfigORM, ac.id)
        assert fetched is not None
        assert fetched.name == "Test Agent"
        assert fetched.llm_config["model_name"] == "gpt-4o"

    Base.metadata.drop_all(engine) 
