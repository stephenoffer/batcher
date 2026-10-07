"""DML clauses beyond the core trio, against DuckDB running the same statements.

`RETURNING`, `INSERT ... ON CONFLICT` and `DELETE ... USING` each lower onto machinery the
engine already has -- a projection over the touched rows, `compose_merge`, and the correlated
`EXISTS` decorrelation -- so what needs proving is that the lowering means what the SQL
means. Every case runs the identical statement text through both engines and compares the
statement's result and the table state after it.

Two deliberate divergences are pinned rather than hidden, in `TestUpsertRefusals`:

- DuckDB infers the conflict target of a bare `ON CONFLICT DO NOTHING` from the table's
  primary key. Batcher tables declare no key, so it refuses the statement instead.
- DuckDB accepts two inserted rows with the same key and keeps one of them. Which one is a
  question of row order, which a lazy relation does not fix, so Batcher refuses the
  statement before changing anything. Postgres refuses the `DO UPDATE` form too.
"""

from __future__ import annotations

import duckdb
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

T_ROWS = "(1, 10), (2, 20), (3, NULL)"
S_ROWS = "(1, TRUE), (1, TRUE), (NULL, TRUE), (3, FALSE)"


def _key(row: tuple) -> tuple:
    return tuple((v is None, v) for v in row)


def _sorted(rows) -> list[tuple]:
    return sorted((tuple(r) for r in rows), key=_key)


def _batcher_rows(ds: bt.Dataset) -> list[tuple]:
    table = ds.to_arrow()
    return _sorted(zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True))


def _duck(t_ddl: str = "CREATE TABLE t(id BIGINT, v BIGINT)") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute(t_ddl)
    conn.execute(f"INSERT INTO t VALUES {T_ROWS}")
    conn.execute("CREATE TABLE s(id BIGINT, f BOOLEAN)")
    conn.execute(f"INSERT INTO s VALUES {S_ROWS}")
    conn.execute("CREATE TABLE u(k BIGINT)")
    conn.execute("INSERT INTO u VALUES (1), (2)")
    return conn


def _session() -> bt.Session:
    """The same three tables, built by the same SQL text."""
    session = bt.Session()
    session.sql("CREATE TABLE t AS SELECT * FROM (VALUES " + T_ROWS + ") AS x(id, v)")
    session.sql("CREATE TABLE s AS SELECT * FROM (VALUES " + S_ROWS + ") AS x(id, f)")
    session.sql("CREATE TABLE u AS SELECT * FROM (VALUES (1), (2)) AS x(k)")
    return session


def _both(statement: str, t_ddl: str = "CREATE TABLE t(id BIGINT, v BIGINT)"):
    """Run `statement` on both engines: (batcher result, duck result, batcher t, duck t)."""
    conn = _duck(t_ddl)
    duck_result = _sorted(conn.execute(statement).fetchall())
    duck_t = _sorted(conn.execute("SELECT id, v FROM t").fetchall())

    session = _session()
    result = _batcher_rows(session.sql(statement))
    after = _batcher_rows(session.sql("SELECT id, v FROM t"))
    return result, duck_result, after, duck_t


class TestReturning:
    """`RETURNING` projects the rows a statement touched, and the statement still applies."""

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO t VALUES (4, 40), (5, NULL) RETURNING *",
            "INSERT INTO t VALUES (4, 40) RETURNING v + 1 AS w, id",
            "INSERT INTO t (id) VALUES (6) RETURNING t.id, t.v",
            "DELETE FROM t WHERE v >= 20 RETURNING *",
            "DELETE FROM t WHERE v > 100 RETURNING id",  # touches nothing
            "DELETE FROM t RETURNING id",  # touches everything
            "DELETE FROM t WHERE v IS NULL RETURNING id, v",  # a NULL row
            "UPDATE t SET v = v * 2 WHERE id <= 2 RETURNING *",
            "UPDATE t SET v = 0 WHERE id = 99 RETURNING *",  # touches nothing
            "UPDATE t SET v = 7 RETURNING id, v - 1 AS before",  # no WHERE: every row
            "UPDATE t SET v = id WHERE v IS NULL RETURNING id, v",
        ],
    )
    def test_matches_duckdb(self, statement):
        result, duck_result, after, duck_t = _both(statement)
        assert result == duck_result
        assert after == duck_t

    def test_without_returning_the_statement_returns_the_new_state(self):
        """The clause is what changes the result; the long-standing default is untouched."""
        session = _session()
        state = _batcher_rows(session.sql("DELETE FROM t WHERE id = 1"))
        assert state == [(2, 20), (3, None)]

    def test_returning_on_a_catalog_table_is_refused_before_writing(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA s1")
        session.sql("CREATE TABLE s1.t AS SELECT 1 AS id")
        with pytest.raises(PlanError, match="RETURNING on the catalog table"):
            session.sql("DELETE FROM s1.t WHERE id = 1 RETURNING *")
        assert session.sql("SELECT * FROM s1.t").to_pydict() == {"id": [1]}


class TestDeleteUsing:
    """`DELETE ... USING` deletes each matched target row once, however many rows match."""

    @pytest.mark.parametrize(
        "statement",
        [
            # id 1 matches two rows of s: deleted once, not twice, and nothing else goes.
            "DELETE FROM t USING s WHERE t.id = s.id",
            "DELETE FROM t USING s WHERE t.id = s.id AND s.f",
            "DELETE FROM t USING s WHERE t.id = s.id AND s.id > 100",  # no match
            # A target-only conjunct, including one that is NULL for a row.
            "DELETE FROM t USING s WHERE t.id = s.id AND t.v > 5",
            "DELETE FROM t USING s WHERE t.id = s.id AND v IS NULL",
            # Two USING relations.
            "DELETE FROM t USING s, u WHERE t.id = s.id AND s.id = u.k",
            "DELETE FROM t AS x USING s WHERE x.id = s.id RETURNING x.id, x.v",
            "DELETE FROM t USING s WHERE t.id = s.id RETURNING *",
        ],
    )
    def test_matches_duckdb(self, statement):
        result, duck_result, after, duck_t = _both(statement)
        assert after == duck_t
        if "RETURNING" in statement:
            assert result == duck_result

    def test_a_null_key_never_matches(self):
        """`t` has no NULL id here, `s` has one: the NULL joins nothing and deletes nothing."""
        _, _, after, duck_t = _both("DELETE FROM t USING s WHERE t.id = s.id AND s.id IS NULL")
        assert after == duck_t == [(1, 10), (2, 20), (3, None)]


_PK = "CREATE TABLE t(id BIGINT PRIMARY KEY, v BIGINT)"


class TestUpsert:
    """`INSERT ... ON CONFLICT (k)` against DuckDB with `k` declared as the key there."""

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO t VALUES (2, 99), (4, 40) ON CONFLICT (id) DO NOTHING",
            "INSERT INTO t VALUES (5, 50) ON CONFLICT (id) DO NOTHING",  # no conflict
            "INSERT INTO t VALUES (1, 1), (2, 2) ON CONFLICT (id) DO NOTHING",  # all conflict
            "INSERT INTO t VALUES (2, 99), (4, 40) ON CONFLICT (id) DO UPDATE SET v = excluded.v",
            # A bare column names the existing row.
            "INSERT INTO t VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET v = v + 100",
            "INSERT INTO t VALUES (1, 5), (2, 50) ON CONFLICT (id) "
            "DO UPDATE SET v = excluded.v + t.v WHERE excluded.v > t.v",
            # The existing value is NULL: NULL + 1 is NULL, and the WHERE on it is not true.
            "INSERT INTO t VALUES (3, 1) ON CONFLICT (id) DO UPDATE SET v = t.v + excluded.v",
            "INSERT INTO t VALUES (3, 1) ON CONFLICT (id) DO UPDATE SET v = 9 WHERE t.v > 0",
            "INSERT INTO t (id) VALUES (2) ON CONFLICT (id) DO UPDATE SET v = excluded.v",
            "INSERT INTO t SELECT id + 1, v FROM t WHERE v IS NOT NULL "
            "ON CONFLICT (id) DO UPDATE SET v = excluded.v",
        ],
    )
    def test_matches_duckdb(self, statement):
        _, _, after, duck_t = _both(statement, t_ddl=_PK)
        assert after == duck_t

    def test_a_null_key_never_conflicts(self):
        """DuckDB needs a UNIQUE (not a PRIMARY KEY) column to hold a NULL key."""
        statement = "INSERT INTO t VALUES (NULL, 7), (1, 8) ON CONFLICT (id) DO UPDATE SET v = 0"
        _, _, after, duck_t = _both(statement, t_ddl="CREATE TABLE t(id BIGINT UNIQUE, v BIGINT)")
        assert after == duck_t == [(1, 0), (2, 20), (3, None), (None, 7)]

    def test_two_null_keys_are_not_duplicates(self):
        """A pinned divergence. SQL says a NULL never equals anything, so two NULL keys never
        conflict and both rows are inserted, which is what Postgres does and what DuckDB
        does for the same rows *without* ON CONFLICT. DuckDB 1.x keeps only the first of
        them under ``ON CONFLICT DO NOTHING``; Batcher follows the rule, not the quirk."""
        statement = "INSERT INTO t VALUES (NULL, 7), (NULL, 8) ON CONFLICT (id) DO NOTHING"
        _, _, after, duck_t = _both(statement, t_ddl="CREATE TABLE t(id BIGINT UNIQUE, v BIGINT)")
        assert after == [(1, 10), (2, 20), (3, None), (None, 7), (None, 8)]
        assert duck_t == [(1, 10), (2, 20), (3, None), (None, 7)]


class TestUpsertRefusals:
    """Shapes refused before the target changes; two are deliberate DuckDB divergences."""

    def test_a_missing_conflict_target_is_refused_where_duckdb_infers_it(self):
        conn = _duck(_PK)
        conn.execute("INSERT INTO t VALUES (1, 5) ON CONFLICT DO NOTHING")  # DuckDB: fine
        session = _session()
        with pytest.raises(PlanError, match="needs a conflict target"):
            session.sql("INSERT INTO t VALUES (1, 5) ON CONFLICT DO NOTHING")

    @pytest.mark.parametrize("action", ["DO NOTHING", "DO UPDATE SET v = excluded.v"])
    def test_duplicate_inserted_keys_are_refused_where_duckdb_picks_one(self, action):
        statement = f"INSERT INTO t VALUES (7, 1), (7, 2) ON CONFLICT (id) {action}"
        conn = _duck(_PK)
        conn.execute(statement)  # DuckDB accepts it and keeps one of the two rows
        session = _session()
        with pytest.raises(PlanError, match="share the key"):
            session.sql(statement)
        assert _batcher_rows(session.sql("SELECT id, v FROM t")) == [(1, 10), (2, 20), (3, None)]

    @pytest.mark.parametrize(
        "statement",
        [
            "INSERT INTO t VALUES (1, 1) ON CONFLICT (nope) DO NOTHING",
            "INSERT INTO t VALUES (1, 1) ON CONFLICT (id) DO UPDATE SET nope = 1",
            "INSERT INTO t VALUES (1, 1) ON CONFLICT (id) DO NOTHING RETURNING *",
        ],
    )
    def test_unsupported_shapes_raise_plan_error(self, statement):
        with pytest.raises(PlanError):
            _session().sql(statement)

    def test_a_catalog_target_is_refused_before_writing(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA s1")
        session.sql("CREATE TABLE s1.t AS SELECT 1 AS id, 1 AS v")
        with pytest.raises(PlanError, match="CONFLICT"):
            session.sql("INSERT INTO s1.t VALUES (1, 2) ON CONFLICT (id) DO NOTHING")
        assert session.sql("SELECT * FROM s1.t").to_pydict() == {"id": [1], "v": [1]}
