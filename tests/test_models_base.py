import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlalchemy as sa
from truss.core.models.base import Base, metadata, NAMING_CONVENTION


def test_metadata_naming_convention():
    # Ensure the MetaData on Base has the right naming convention mapping
    assert metadata.naming_convention == NAMING_CONVENTION
    assert Base.metadata is metadata


def test_create_tables_in_memory_sqlite():
    """Metadata should be able to create/drop tables without error."""
    # Create a dummy table to exercise create_all
    class Dummy(Base):  # type: ignore[misc]
        __tablename__ = "dummy"
        id = sa.Column(sa.Integer, primary_key=True)
        name = sa.Column(sa.String, nullable=False, unique=True)

    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    insp = sa.inspect(engine)
    assert "dummy" in insp.get_table_names()
    # Check that index created follows naming convention (ix_name)
    # indexes = insp.get_indexes("dummy")
    # assert any(ix["name"].startswith("ix_") for ix in indexes)

    # Clean up
    Base.metadata.drop_all(engine)

    # Ensure table is removed by fetching fresh inspector
    insp2 = sa.inspect(engine)
    assert "dummy" not in insp2.get_table_names() 
