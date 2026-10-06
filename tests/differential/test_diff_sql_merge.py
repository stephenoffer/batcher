"""`MERGE INTO` in SQL, against DuckDB running the same statement.

`MERGE` is the lakehouse DML statement — the one Delta, Snowflake and Databricks are all
written against, and the one a ported job is most likely to contain. Batcher had the
semantics already: `write.merge_into(...)` builds the same `WHEN` clauses, and
`api.merge.compose_merge` composes a target's post-merge state as one lazy relation. Only
the SQL spelling was missing, so this is wiring rather than a second implementation, and the
two surfaces cannot disagree because they end in the same function.

Every case here runs the **same statement text** through DuckDB and through Batcher and
compares the resulting table state. That is the strongest form available: not "does it look
right" but "does the engine that defines the semantics agree".

The `ON` condition is restricted to column equalities between the two sides (``t.id =
s.id`` or ``t.id = s.customer_id``), because the engine matches a source row to a target
row by key columns. Anything else is refused rather than approximated — quietly picking a
key would merge on a condition the user did not write, which is the failure mode a
differential test cannot see because both sides would be asked a different question.
"""

from __future__ import annotations

import duckdb
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

TARGET = {"k": [1, 2, 3], "val": [10, 20, 30]}
SOURCE = {"k": [2, 3, 4], "val": [99, 98, 40]}


def _duck():
    conn = duckdb.connect()
    conn.sql("CREATE TABLE t AS SELECT * FROM (VALUES (1,10),(2,20),(3,30)) AS v(k,val)")
    conn.sql("CREATE TABLE s AS SELECT * FROM (VALUES (2,99),(3,98),(4,40)) AS v(k,val)")
    return conn


def _session():
    session = bt.Session()
    session.register("t", bt.from_pydict(dict(TARGET)))
    session.register("s", bt.from_pydict(dict(SOURCE)))
    return session


def _both(statement: str):
    """Run `statement` on both engines and return their resulting table states, sorted."""
    conn = _duck()
    conn.sql(statement)
    expected = sorted(conn.sql("SELECT k, val FROM t").fetchall())

    session = _session()
    session.sql(statement)
    rows = session.sql("SELECT k, val FROM t").to_pydict()
    got = sorted(zip(rows["k"], rows["val"], strict=True))
    return got, expected


@pytest.mark.parametrize(
    ("label", "statement"),
    [
        (
            "upsert",
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED THEN UPDATE SET val = s.val "
            "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (s.k, s.val)",
        ),
        (
            "matched update only",
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET val = s.val",
        ),
        (
            "not matched insert only",
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (s.k, s.val)",
        ),
        (
            "matched delete",
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE",
        ),
        (
            "conditional matched",
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED AND s.val > 98 THEN UPDATE SET val = s.val",
        ),
        (
            "not matched by source delete",
            "MERGE INTO t USING s ON t.k = s.k WHEN NOT MATCHED BY SOURCE THEN DELETE",
        ),
        (
            "aliased source",
            "MERGE INTO t USING s AS src ON t.k = src.k WHEN MATCHED THEN UPDATE SET val = src.val",
        ),
        (
            "expression in the assignment",
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET val = t.val + s.val",
        ),
        (
            "insert star",
            "MERGE INTO t USING s ON t.k = s.k WHEN NOT MATCHED THEN INSERT *",
        ),
        (
            "update star",
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET *",
        ),
        (
            "key equality written source-first",
            "MERGE INTO t USING s ON s.k = t.k WHEN MATCHED THEN UPDATE SET val = s.val",
        ),
        (
            "delete and insert together",
            "MERGE INTO t USING s ON t.k = s.k "
            "WHEN MATCHED THEN DELETE "
            "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (s.k, s.val)",
        ),
    ],
)
def test_the_table_state_matches_duckdb(label, statement):
    got, expected = _both(statement)
    assert got == expected, label


def test_a_subquery_source_works():
    """`USING (SELECT ...)` is the form a real merge takes: the change set is computed."""
    statement = (
        "MERGE INTO t USING (SELECT k, val FROM s WHERE val > 50) AS src ON t.k = src.k "
        "WHEN MATCHED THEN UPDATE SET val = src.val"
    )
    got, expected = _both(statement)
    assert got == expected


def test_the_sql_spelling_agrees_with_the_python_builder(tmp_path):
    """The two surfaces end in the same `compose_merge`, so they must agree. Asserting it
    is what keeps the SQL path from drifting into a second implementation."""
    session = _session()
    session.sql(
        "MERGE INTO t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET val = s.val "
        "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (s.k, s.val)"
    )
    via_sql = session.sql("SELECT k, val FROM t ORDER BY k").to_pydict()

    path = str(tmp_path / "t")
    bt.from_pydict(dict(TARGET)).repartition(num_files=1).write(path, format="parquet")
    bt.from_pydict(dict(SOURCE)).write.merge_into(
        path, on="k"
    ).when_matched().update_all().when_not_matched().insert_all().execute()
    via_builder = bt.read.parquet(path).sort("k").to_pydict()

    assert via_sql["k"] == via_builder["k"]
    assert via_sql["val"] == via_builder["val"]


class TestRefusals:
    """Shapes the engine has no key for are refused, not approximated."""

    def test_a_non_equality_condition_is_refused(self):
        with pytest.raises(Exception, match="equalities of the form"):
            _session().sql("MERGE INTO t USING s ON t.k > s.k WHEN MATCHED THEN DELETE")

    def test_insert_star_with_differently_named_keys_is_refused(self):
        """`INSERT *` takes each column by name, so with ``t.k = o.j`` the key would come
        from a source column the ON never named. Refused rather than guessed."""
        session = _session()
        session.register("o", bt.from_pydict({"j": [9], "k": [1], "val": [1]}))
        with pytest.raises(Exception, match="writes every column by name"):
            session.sql("MERGE INTO t USING o ON t.k = o.j WHEN NOT MATCHED THEN INSERT *")

    def test_returning_is_refused(self):
        with pytest.raises(Exception, match="RETURNING is not supported"):
            _session().sql("MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE RETURNING *")

    def test_a_merge_with_no_when_clause_is_refused(self):
        """Refused by the *parser*, before the translation runs, which is why the message
        is sqlglot's rather than the engine's. Asserted anyway: the property that matters is
        that a MERGE doing nothing cannot be issued, not which layer stops it. The
        translator keeps its own guard for the same case reached with a hand-built AST,
        where a silent no-op would otherwise be the outcome."""
        with pytest.raises(Exception, match=r"could not parse SQL|at least one WHEN"):
            _session().sql("MERGE INTO t USING s ON t.k = s.k")

    def test_an_unknown_target_names_the_registered_tables(self):
        with pytest.raises(Exception, match="no table"):
            _session().sql("MERGE INTO nope USING s ON nope.k = s.k WHEN MATCHED THEN DELETE")


class TestDifferentKeyNames:
    """``ON t.k = c.customer_k``: the key pairs columns of different names (AP-329)."""

    @staticmethod
    def _both(statement: str):
        conn = _duck()
        conn.sql(
            "CREATE TABLE c AS SELECT * FROM (VALUES (2,99),(4,40),(NULL,7)) AS v(customer_k,val)"
        )
        conn.sql(statement)
        expected = sorted(conn.sql("SELECT k, val FROM t").fetchall(), key=str)

        session = _session()
        session.sql(
            "CREATE TABLE c AS SELECT * FROM (VALUES (2,99),(4,40),(NULL,7)) AS v(customer_k,val)"
        )
        session.sql(statement)
        rows = session.sql("SELECT k, val FROM t").to_pydict()
        return sorted(zip(rows["k"], rows["val"], strict=True), key=str), expected

    @pytest.mark.parametrize(
        "statement",
        [
            "MERGE INTO t USING c ON t.k = c.customer_k "
            "WHEN MATCHED THEN UPDATE SET val = c.val "
            "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (c.customer_k, c.val)",
            "MERGE INTO t USING c ON c.customer_k = t.k WHEN MATCHED THEN DELETE",
            "MERGE INTO t USING c AS src ON t.k = src.customer_k "
            "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET val = 0",
        ],
    )
    def test_matches_duckdb(self, statement):
        got, expected = self._both(statement)
        assert got == expected


class TestCatalogTarget:
    """MERGE into a *catalog* table: the same rewrite, collected and written back (AP-325)."""

    STATEMENT = (
        "MERGE INTO wh.t AS t USING s ON t.k = s.k "
        "WHEN MATCHED THEN UPDATE SET val = s.val "
        "WHEN NOT MATCHED THEN INSERT (k, val) VALUES (s.k, s.val)"
    )

    def _expected(self):
        conn = _duck()
        conn.sql(self.STATEMENT.replace("wh.t AS t", "t"))
        return sorted(conn.sql("SELECT k, val FROM t").fetchall())

    def _check(self, session):
        session.sql("CREATE SCHEMA wh")
        session.sql("CREATE TABLE wh.t AS SELECT * FROM (VALUES (1,10),(2,20),(3,30)) AS v(k,val)")
        session.register("s", bt.from_pydict(dict(SOURCE)))
        session.sql(self.STATEMENT)
        rows = session.sql("SELECT k, val FROM wh.t").to_pydict()
        assert sorted(zip(rows["k"], rows["val"], strict=True)) == self._expected()

    def test_memory_catalog(self):
        self._check(bt.Session())

    def test_directory_catalog(self, tmp_path):
        session = bt.Session()
        session.catalog.attach(bt.Catalog.from_directory(str(tmp_path), name="dir"))
        session.sql("USE dir")
        self._check(session)
        # The rows were written to storage: a fresh read of the table sees them.
        assert sorted(session.catalog.get_table("dir.wh.t").read().to_pydict()["k"]) == [
            1,
            2,
            3,
            4,
        ]
