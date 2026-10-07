"""The PEP 249 adapter: globals, lifecycle, binding, fetching, errors and no transactions."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import dbapi

pytestmark = pytest.mark.unit


@pytest.fixture
def session() -> bt.Session:
    s = bt.Session()
    s.register("t", bt.from_pydict({"id": [1, 2, 3], "name": ["a", None, "c"]}))
    return s


@pytest.fixture
def cur(session):
    return dbapi.connect(session).cursor()


def test_module_globals_follow_pep_249() -> None:
    assert dbapi.apilevel == "2.0"
    assert dbapi.threadsafety == 1
    assert dbapi.paramstyle == "qmark"


def test_the_exception_tree_is_pep_249s() -> None:
    assert issubclass(dbapi.Error, Exception)
    assert issubclass(dbapi.Warning, Warning)
    assert issubclass(dbapi.InterfaceError, dbapi.Error)
    assert issubclass(dbapi.DatabaseError, dbapi.Error)
    for cls in (
        dbapi.DataError,
        dbapi.OperationalError,
        dbapi.IntegrityError,
        dbapi.InternalError,
        dbapi.ProgrammingError,
        dbapi.NotSupportedError,
    ):
        assert issubclass(cls, dbapi.DatabaseError)


def test_connect_defaults_to_the_current_session() -> None:
    assert dbapi.connect().session is bt.current_session()
    with pytest.raises(dbapi.InterfaceError, match=r"bt\.Session"):
        dbapi.connect("not a session")


def test_qmark_and_named_parameters_bind_as_values(cur) -> None:
    assert cur.execute("SELECT id FROM t WHERE id > ? ORDER BY id", [1]).fetchall() == [(2,), (3,)]
    assert cur.execute("SELECT id FROM t WHERE id = $k", {"k": 3}).fetchall() == [(3,)]
    assert cur.execute("SELECT id FROM t WHERE name = ?", ["a' OR '1'='1"]).fetchall() == []


def test_bad_parameters_are_programming_errors(cur) -> None:
    with pytest.raises(dbapi.ProgrammingError, match="sequence or a mapping"):
        cur.execute("SELECT ?", "1")
    with pytest.raises(dbapi.ProgrammingError):
        cur.execute("SELECT ? AS a", [])


def test_description_types_compare_to_type_objects(cur) -> None:
    cur.execute(
        "SELECT 1 AS i, 1.5 AS f, 'x' AS s, DATE '2024-01-01' AS d, 'ab'::BLOB AS b, true AS t"
    )
    codes = {d[0]: d[1] for d in cur.description}
    assert codes["i"] == dbapi.NUMBER and codes["f"] == dbapi.NUMBER
    assert codes["s"] == dbapi.STRING and codes["s"] != dbapi.NUMBER
    assert codes["d"] == dbapi.DATETIME
    assert codes["b"] == dbapi.BINARY
    assert codes["t"] != dbapi.ROWID
    assert all(len(d) == 7 for d in cur.description)
    assert isinstance(codes["i"], pa.DataType)


def test_fetch_paths_agree(cur) -> None:
    query = "SELECT id, name FROM t ORDER BY id"
    assert cur.execute(query).fetchone() == (1, "a")
    assert cur.fetchmany(5) == [(2, None), (3, "c")]
    assert cur.fetchall() == []
    cur.arraysize = 2
    assert cur.execute(query).fetchmany() == [(1, "a"), (2, None)]
    assert list(cur.execute(query)) == [(1, "a"), (2, None), (3, "c")]


def test_fetch_arrow_table_after_a_partial_fetch(cur) -> None:
    cur.execute("SELECT id FROM t ORDER BY id")
    assert cur.fetchone() == (1,)
    assert cur.fetch_arrow_table().to_pydict() == {"id": [2, 3]}
    assert cur.fetch_arrow_table().num_rows == 0
    assert cur.execute("SELECT id FROM t ORDER BY id").fetch_arrow_table().num_rows == 3


def test_many_batches_stream_in_order(cur) -> None:
    rows = cur.execute("SELECT i FROM range(100000) t(i) ORDER BY i").fetchmany(70000)
    assert rows[0] == (0,) and rows[-1] == (69999,) and len(rows) == 70000
    assert len(cur.fetchall()) == 30000


def test_zero_column_and_empty_results(cur) -> None:
    assert cur.execute("SELECT id FROM t WHERE id > 99").fetchall() == []
    assert cur.description[0][0] == "id"


def test_statements_without_a_result_set(session, cur) -> None:
    for statement in (
        "CREATE TABLE u AS SELECT 1 AS x",
        "INSERT INTO u VALUES (2)",
        "UPDATE u SET x = 3 WHERE x = 2",
        "DELETE FROM u WHERE x = 1",
    ):
        cur.execute(statement)
        assert cur.description is None
        assert cur.rowcount == -1
        with pytest.raises(dbapi.ProgrammingError, match="no result set"):
            cur.fetchall()
    assert session.sql("SELECT x FROM u").to_pydict() == {"x": [3]}
    cur.execute("DROP TABLE u")
    assert "u" not in session


def test_executemany_runs_each_and_refuses_queries(session, cur) -> None:
    cur.execute("CREATE TABLE u AS SELECT 0 AS x")
    cur.executemany("INSERT INTO u VALUES (?)", [[1], [2], [3]])
    assert session.sql("SELECT SUM(x) AS s FROM u").to_pydict() == {"s": [6]}
    with pytest.raises(dbapi.ProgrammingError, match="execute"):
        cur.executemany("SELECT ?", [[1]])


def test_error_mapping(cur) -> None:
    with pytest.raises(dbapi.ProgrammingError) as syntax:
        cur.execute("SELEC 1")
    assert isinstance(syntax.value.__cause__, bt.SQLSyntaxError)
    with pytest.raises(dbapi.ProgrammingError, match="no_such"):
        cur.execute("SELECT * FROM no_such")
    with pytest.raises(dbapi.NotSupportedError):
        cur.execute("SELECT no_such_fn(id) FROM t")


def test_translate_covers_the_hierarchy() -> None:
    from batcher._internal.errors import DataQualityError, ExecutionError, QueryCancelledError
    from batcher.dbapi.errors import translate

    assert type(translate(DataQualityError("bad"))) is dbapi.DataError
    assert type(translate(ExecutionError("boom"))) is dbapi.OperationalError
    assert type(translate(QueryCancelledError("stop"))) is dbapi.OperationalError
    assert type(translate(RuntimeError("x"))) is dbapi.DatabaseError


def test_a_read_only_session_refuses_writes(session) -> None:
    cur = dbapi.connect(bt.Session(read_only=True)).cursor()
    with pytest.raises(dbapi.ProgrammingError, match="read-only"):
        cur.execute("CREATE TABLE x AS SELECT 1 AS a")


def test_commit_is_a_no_op_and_rollback_refuses_lost_writes(session) -> None:
    conn = dbapi.connect(session)
    conn.rollback()  # nothing written yet: nothing to undo
    conn.cursor().execute("SELECT 1")
    conn.rollback()  # a read is not a write
    conn.cursor().execute("CREATE TABLE w AS SELECT 1 AS x")
    with pytest.raises(dbapi.NotSupportedError, match="1 statement"):
        conn.rollback()
    assert "w" in session  # the write stayed applied
    conn.rollback()  # acknowledged by the raise
    conn.cursor().execute("DROP TABLE w")
    conn.commit()
    conn.rollback()


def test_close_closes_cursors_and_leaves_the_session(session) -> None:
    conn = dbapi.connect(session)
    cur = conn.cursor()
    cur.execute("SELECT id FROM t")
    conn.close()
    assert cur.closed and conn.closed
    with pytest.raises(dbapi.InterfaceError):
        cur.fetchall()
    with pytest.raises(dbapi.InterfaceError):
        conn.cursor()
    assert "t" in session
    with dbapi.connect(session) as conn2, conn2.cursor() as c2:
        assert c2.execute("SELECT COUNT(*) FROM t").fetchone() == (3,)
    assert conn2.closed and c2.closed


def test_type_constructors() -> None:
    assert dbapi.Date(2024, 2, 1) == dt.date(2024, 2, 1)
    assert dbapi.Time(1, 2, 3) == dt.time(1, 2, 3)
    assert dbapi.Timestamp(2024, 2, 1, 1, 2, 3) == dt.datetime(2024, 2, 1, 1, 2, 3)
    assert dbapi.Binary(memoryview(b"z")) == b"z"
    assert isinstance(dbapi.TimestampFromTicks(0), dt.datetime)
