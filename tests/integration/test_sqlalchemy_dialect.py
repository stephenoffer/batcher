"""The ``batcher://`` SQLAlchemy dialect, end to end on SQLAlchemy 2.0: Core, reflection, writes."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import batcher as bt
from batcher import dbapi

sa = pytest.importorskip("sqlalchemy")

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _registered() -> None:
    """Register the pyproject entry points by hand; the editable install predates them."""
    from sqlalchemy.dialects import registry

    for name in ("batcher", "batcher.dbapi"):
        registry.register(name, "batcher.integrations.sqlalchemy", "BatcherDialect")


@pytest.fixture
def session() -> bt.Session:
    s = bt.Session()
    s.register(
        "orders",
        bt.from_pydict({"id": [1, 2, 3], "amount": [5.0, 7.5, None], "name": ["a", "b", "c"]}),
    )
    s.sql("CREATE VIEW big AS SELECT id FROM orders WHERE amount > 6")
    return s


@pytest.fixture
def engine(session):
    return sa.create_engine("batcher+dbapi://", connect_args={"session": session})


def test_pyproject_registers_both_url_spellings() -> None:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    points = data["project"]["entry-points"]["sqlalchemy.dialects"]
    target = "batcher.integrations.sqlalchemy:BatcherDialect"
    assert points == {"batcher": target, "batcher.dbapi": target}


def test_bound_core_query(engine) -> None:
    orders = sa.Table("orders", sa.MetaData(), autoload_with=engine)
    query = (
        sa.select(orders.c.id, orders.c.amount)
        .where(orders.c.amount > sa.bindparam("lo"))
        .order_by(orders.c.id)
        .limit(10)
    )
    with engine.connect() as conn:
        assert conn.execute(query, {"lo": 6.0}).all() == [(2, 7.5)]
        assert conn.execute(sa.select(sa.func.count()).select_from(orders)).scalar() == 3


def test_reflection_reads_information_schema(engine) -> None:
    inspector = sa.inspect(engine)
    assert inspector.get_table_names() == ["orders"]
    assert inspector.get_view_names() == ["big"]
    assert inspector.has_table("orders") and inspector.has_table("big")
    assert not inspector.has_table("nope")
    assert "main" in inspector.get_schema_names()
    columns = {c["name"]: c for c in inspector.get_columns("orders")}
    assert isinstance(columns["id"]["type"], sa.BigInteger)
    assert isinstance(columns["amount"]["type"], sa.Double)
    assert isinstance(columns["name"]["type"], sa.String)
    assert inspector.get_pk_constraint("orders")["constrained_columns"] == []
    assert inspector.get_foreign_keys("orders") == []
    assert inspector.get_indexes("orders") == []
    with pytest.raises(sa.exc.NoSuchTableError):
        inspector.get_columns("nope")


@pytest.mark.parametrize(
    ("arrow", "expected"),
    [
        ("int64", "BigInteger"),
        ("bool", "Boolean"),
        ("date32[day]", "Date"),
        ("timestamp[us]", "DateTime"),
        ("time64[us]", "Time"),
        ("duration[s]", "Interval"),
        ("binary", "LargeBinary"),
        ("list<item: int64>", "NullType"),
    ],
)
def test_type_mapping(arrow, expected) -> None:
    from batcher.integrations.sqlalchemy.dialect import sqlalchemy_type

    assert type(sqlalchemy_type(arrow)).__name__ == expected


def test_type_mapping_keeps_precision_and_time_zone() -> None:
    from batcher.integrations.sqlalchemy.dialect import sqlalchemy_type

    decimal = sqlalchemy_type("decimal128(10, 2)")
    assert (decimal.precision, decimal.scale) == (10, 2)
    assert sqlalchemy_type("timestamp[us, tz=UTC]").timezone is True


def test_writes_in_begin_block_take_effect(engine, session) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE extra AS SELECT 1 AS k"))
        conn.execute(sa.insert(sa.table("extra", sa.column("k"))), [{"k": 2}, {"k": 3}])
    assert session.sql("SELECT SUM(k) AS s FROM extra").to_pydict() == {"s": [6]}


def test_a_rollback_after_a_write_is_refused_not_faked(engine, session) -> None:
    with pytest.raises(sa.exc.NotSupportedError, match="cannot undo"), engine.connect() as conn:
        conn.execute(sa.text("CREATE TABLE kept AS SELECT 1 AS k"))
        conn.rollback()
    assert "kept" in session


def test_a_read_only_block_closes_cleanly(engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT 1")).scalar() == 1
        conn.rollback()


def test_only_autocommit_isolation(engine) -> None:
    with engine.connect() as conn:
        assert conn.get_isolation_level() == "AUTOCOMMIT"
    with pytest.raises(sa.exc.ArgumentError, match="AUTOCOMMIT"):
        engine.execution_options(isolation_level="SERIALIZABLE").connect()


def test_a_url_naming_a_database_is_refused() -> None:
    with pytest.raises(sa.exc.ArgumentError, match="names no"):
        sa.create_engine("batcher://somehost/db").connect()


def test_the_default_url_uses_the_current_session() -> None:
    engine = sa.create_engine("batcher://")
    with engine.connect() as conn:
        raw = conn.connection.dbapi_connection
        assert isinstance(raw, dbapi.Connection)
        assert raw.session is bt.current_session()
