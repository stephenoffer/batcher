"""The SQL session surface: typed errors, dialect checks, read-only, scripts, and scoping.

Each test pins one contract of `bt.Session` and the three SQL entry points (`bt.sql`,
`Session.sql`, `Dataset.sql`), most of them a behaviour that used to fail in a way no test
could see: a raw sqlglot or ``NotImplementedError`` escaping, an unknown dialect accepted
until the first query, or the process-global default session leaking between tasks.
"""

from __future__ import annotations

import asyncio

import pytest

import batcher as bt


@pytest.fixture
def t():
    return bt.from_pydict({"a": [1, 2, 3], "s": ["x", "y", "z"]})


# --- errors (AP-306 / AP-345) ------------------------------------------------------------


def test_a_syntax_error_is_typed_and_located():
    with pytest.raises(bt.SQLSyntaxError) as raised:
        bt.sql("SELECT a\nFROM t WHERE (")
    err = raised.value
    assert isinstance(err, bt.PlanError)
    assert not isinstance(err, NotImplementedError)
    assert (err.line, err.column, err.start, err.end) == (2, 14, 22, 22)


def test_an_unknown_function_is_unsupported_and_located(t):
    with pytest.raises(bt.SQLUnsupportedError) as raised:
        bt.sql("SELECT a,\n  no_such_fn(a) FROM t", t=t)
    err = raised.value
    assert isinstance(err, bt.PlanError) and isinstance(err, NotImplementedError)
    assert (err.line, err.column, err.start, err.end) == (2, 3, 12, 21)
    assert "no_such_fn" in err.message


@pytest.mark.parametrize(
    "query",
    [
        # The two refusals that stood here, an aggregate inside a QUALIFY window and two
        # DISTINCT aggregates in one SELECT, are now supported and answer as DuckDB does,
        # so they no longer exercise a refusal. These two still do.
        "SELECT a, lag(a IGNORE NULLS) OVER () FROM t",
        "SELECT count(DISTINCT a, s) FROM t",
        "WITH RECURSIVE c(n) AS (SELECT 1 INTERSECT SELECT n FROM c) SELECT n FROM c",
    ],
)
def test_every_refusal_is_both_plan_error_and_not_implemented(t, query):
    """A refusal raised before the typed class existed keeps NotImplementedError."""
    with pytest.raises(bt.SQLUnsupportedError) as raised:
        bt.sql(query, t=t)
    assert isinstance(raised.value, NotImplementedError)
    assert isinstance(raised.value, bt.PlanError)


def test_schema_validates_without_running(t):
    """``.schema`` answers from the plan, so a query is checked before any work happens."""
    ds = bt.sql("SELECT a, upper(s) AS u FROM t WHERE a > 1", t=t)
    assert ds.schema.names == ["a", "u"]
    with pytest.raises(bt.PlanError):
        bt.sql("SELECT nope FROM t", t=t).schema  # noqa: B018 - the property raises


# --- dialects (AP-304) -------------------------------------------------------------------


def test_an_unknown_dialect_fails_at_construction():
    with pytest.raises(bt.PlanError, match="unknown SQL dialect 'nosuch'") as raised:
        bt.Session(dialect="nosuch")
    assert "duckdb" in raised.value.available


@pytest.mark.parametrize("call", ["bt", "session", "dataset"])
def test_an_unknown_per_call_dialect_is_a_plan_error(t, call):
    with pytest.raises(bt.PlanError, match="unknown SQL dialect"):
        if call == "bt":
            bt.sql("SELECT 1", dialect="nosuch")
        elif call == "session":
            bt.Session().sql("SELECT 1", dialect="nosuch")
        else:
            t.sql("SELECT a FROM self", dialect="nosuch")


def test_a_dialect_changes_grammar_not_semantics(t):
    """Postgres would answer 3 for 7 / 2; the dialect only picks the parser."""
    got = bt.sql("SELECT 7 / 2 AS q", dialect="postgres").to_pydict()
    assert got == {"q": [3.5]}


# --- one signature across the entry points (AP-302 / AP-309) -----------------------------


def test_the_three_entry_points_take_the_same_bindings(t):
    other = bt.from_pydict({"a": [2, 3], "w": [20, 30]})
    query = "SELECT t.a, w FROM t JOIN other USING (a) WHERE t.a > ? ORDER BY t.a"
    expected = {"a": [3], "w": [30]}
    session = bt.Session()
    assert bt.sql(query, {"t": t}, other=other, params=[2]).to_pydict() == expected
    assert session.sql(query, {"t": t}, other=other, params=[2]).to_pydict() == expected
    assert t.sql(query, table_name="t", other=other, params=[2]).to_pydict() == expected
    assert t.sql(query, {"other": other}, table_name="t", params=[2]).to_pydict() == expected


def test_session_sql_converts_a_bound_dict(t):
    """`Session.sql` coerces a binding the way `bt.sql` always has."""
    got = bt.Session().sql("SELECT sum(x) AS s FROM d", d={"x": [1, 2]}).to_pydict()
    assert got == {"s": [3]}


def test_dataset_sql_refuses_a_second_binding_of_its_own_name(t):
    with pytest.raises(bt.PlanError, match="already this dataset's name"):
        t.sql("SELECT * FROM x", table_name="x", x=t)


# --- read-only sessions (AP-310) ---------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE u AS SELECT 1 AS x",
        "CREATE VIEW v AS SELECT a FROM t",
        "CREATE SCHEMA scratch",
        "DROP TABLE t",
        "INSERT INTO t VALUES (4, 'w')",
        "DELETE FROM t WHERE a = 1",
        "UPDATE t SET a = 0",
        "MERGE INTO t USING t AS s ON t.a = s.a WHEN MATCHED THEN DELETE",
        "TRUNCATE TABLE t",
        "ALTER TABLE t RENAME TO t2",
    ],
)
def test_a_read_only_session_refuses_writes(t, statement):
    session = bt.Session(read_only=True)
    session.register("t", t)
    with pytest.raises(bt.PlanError, match="read-only"):
        session.sql(statement)
    assert session.list() == ["t"]
    assert session.table("t").count() == 3


@pytest.mark.parametrize(
    "statement",
    ["SELECT count(*) FROM t", "SHOW TABLES", "DESCRIBE t", "EXPLAIN SELECT a FROM t"],
)
def test_a_read_only_session_runs_reads(t, statement):
    session = bt.Session(read_only=True)
    session.register("t", t)
    assert session.sql(statement).collect().num_rows >= 1


def test_a_read_only_session_refuses_explaining_a_write(t):
    session = bt.Session(read_only=True)
    session.register("t", t)
    with pytest.raises(bt.PlanError, match="read-only"):
        session.sql("EXPLAIN INSERT INTO t VALUES (4, 'w')")


def test_read_only_carries_over_a_per_call_dialect(t):
    session = bt.Session(read_only=True)
    with pytest.raises(bt.PlanError, match="read-only"):
        session.sql("CREATE TABLE u AS SELECT 1 AS x", dialect="spark")


# --- scripts (AP-308) --------------------------------------------------------------------


def test_execute_script_runs_each_statement_in_order():
    session = bt.Session()
    results = session.execute_script(
        "CREATE TABLE x AS SELECT 1 AS a;\nINSERT INTO x VALUES (2);\nSELECT sum(a) AS s FROM x"
    )
    assert len(results) == 3
    assert results[-1].to_pydict() == {"s": [3]}


def test_execute_script_is_not_atomic_and_says_how_far_it_got():
    session = bt.Session()
    script = "CREATE TABLE y AS SELECT 1 AS a; SELECT no_fn(a) FROM y; CREATE TABLE z AS SELECT 2"
    with pytest.raises(bt.SQLUnsupportedError) as raised:
        session.execute_script(script)
    assert session.list() == ["y"]  # the first statement stays applied, the third never ran
    assert any("Statement 2 of 3" in note for note in raised.value.__notes__)


def test_execute_script_reports_a_syntax_error_before_running_anything():
    session = bt.Session()
    with pytest.raises(bt.SQLSyntaxError):
        session.execute_script("CREATE TABLE y AS SELECT 1 AS a; SELECT FROM WHERE")
    assert session.list() == []


# --- scoped default session (AP-021) -----------------------------------------------------


def test_activate_scopes_bt_sql_and_restores_on_exit():
    outer, inner = bt.Session(), bt.Session()
    before = bt.current_session()
    with outer.activate():
        bt.sql("CREATE TABLE o AS SELECT 1 AS x")
        with inner.activate():
            assert bt.current_session() is inner
            bt.sql("CREATE TABLE i AS SELECT 2 AS x")
        assert bt.current_session() is outer
        assert bt.sql("SELECT x FROM o").to_pydict() == {"x": [1]}
    assert bt.current_session() is before
    assert (outer.list(), inner.list()) == (["o"], ["i"])
    assert "o" not in before and "i" not in before


def test_activate_restores_after_an_error():
    before = bt.current_session()
    with pytest.raises(RuntimeError), bt.Session().activate():
        raise RuntimeError("boom")
    assert bt.current_session() is before


def test_concurrent_tasks_each_see_their_own_session():
    async def work(name: str) -> list[str]:
        session = bt.Session()
        with session.activate():
            bt.sql(f"CREATE TABLE {name} AS SELECT 1 AS x")
            await asyncio.sleep(0.01)  # let the other task run inside its own scope
            bt.register_function(f"f_{name}", lambda a: a, result_type="int64")
            return [*bt.current_session().list(), *bt.current_session().list_functions()]

    async def main() -> list[list[str]]:
        return await asyncio.gather(work("p"), work("q"))

    assert asyncio.run(main()) == [["p", "f_p"], ["q", "f_q"]]


def test_write_table_resolves_against_the_active_session():
    session = bt.Session()
    with session.activate():
        bt.from_pydict({"x": [1, 2]}).write.table("scoped_write")
    assert session.table("scoped_write").to_pydict() == {"x": [1, 2]}
    assert not bt.current_session().catalog.has_table("scoped_write")


def test_set_session_still_sets_the_process_default_inside_a_scope():
    previous = bt.current_session()
    scoped, default = bt.Session(), bt.Session()
    try:
        with scoped.activate():
            bt.set_session(default)
            assert bt.current_session() is scoped
        assert bt.current_session() is default
    finally:
        bt.set_session(previous)
