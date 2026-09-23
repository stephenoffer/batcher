"""Session views, name resolution and plan-time laziness in the SQL front-end.

Each section pins a behaviour that was wrong, with no error, before it was fixed:

- A ``CREATE VIEW`` stored the plan its body produced at creation, so an ``INSERT INTO``
  or re-``register`` of a base table never reached the view, and ``DROP TABLE`` of a base
  table left the view answering. Views now store their query text and translate per query.
- ``CREATE VIEW vc(a, b) AS …`` registered a view named ``''`` with columns ``1`` and ``2``.
- After ``USE wh.raw``, ``CREATE TABLE x AS …`` made a session-only table rather than
  ``wh.raw.x``.
- Session names were case-sensitive outside SQL (``Session.table``, ``drop``,
  ``register``), so ``MYTAB`` beside ``MyTab`` created a duplicate that made every later
  query ambiguous; catalog names were case-sensitive in SQL; ``SELECT id FROM a JOIN b``
  silently picked a side; ``DROP VIEW`` dropped a table.
- ``bt.sql`` ran work while translating a multiply-referenced CTE, which it materialized.
  The shapes that still run at translation are pinned too, so the docs listing them stay
  true.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ConfigError, PlanError

pytestmark = pytest.mark.unit


def _session(**tables) -> bt.Session:
    s = bt.Session()
    for name, data in tables.items():
        s.register(name, bt.from_pydict(data))
    return s


# --- views are bound when queried ------------------------------------------------------------


def test_view_sees_insert_into_its_base_table():
    s = _session(t={"x": [1, 2, 3]})
    s.sql("CREATE VIEW v AS SELECT sum(x) AS sx FROM t WHERE x > 1")
    assert s.sql("SELECT * FROM v").to_pydict() == {"sx": [5]}
    s.sql("INSERT INTO t VALUES (10)")
    assert s.sql("SELECT * FROM v").to_pydict() == {"sx": [15]}


def test_view_sees_a_reregistered_base_table_and_a_view_over_it():
    s = _session(t={"x": [1, 2, 3]})
    s.sql("CREATE VIEW v1 AS SELECT x FROM t WHERE x > 1")
    s.sql("CREATE VIEW v2 AS SELECT sum(x) AS sx FROM v1")
    s.register("t", bt.from_pydict({"x": [100]}))
    assert s.sql("SELECT * FROM v2").to_pydict() == {"sx": [100]}
    assert s.table("v2").to_pydict() == {"sx": [100]}


def test_dropping_a_base_table_breaks_the_view_on_use():
    """DuckDB allows the DROP and fails the next query of the view; so does Batcher."""
    s = _session(t={"x": [1]})
    s.sql("CREATE VIEW v AS SELECT x FROM t")
    assert s.sql("DROP TABLE t").to_pydict() == {"dropped": ["t"]}
    with pytest.raises(PlanError, match="unknown table 't'"):
        s.sql("SELECT * FROM v")


def test_view_keeps_the_per_call_tables_it_was_defined_over():
    s = bt.Session()
    s.sql("CREATE VIEW v AS SELECT x * 2 AS y FROM src", src=bt.from_pydict({"x": [1, 2]}))
    assert s.sql("SELECT * FROM v ORDER BY y").to_pydict() == {"y": [2, 4]}


def test_view_over_a_catalog_table_sees_writes_to_it():
    s = bt.Session()
    s.sql("CREATE SCHEMA sales")
    s.catalog.create_table("sales.orders", bt.from_pydict({"id": [1, 2]}))
    s.sql("CREATE VIEW n AS SELECT count(*) AS c FROM sales.orders")
    s.sql("INSERT INTO sales.orders VALUES (3)")
    assert s.sql("SELECT c FROM n").to_pydict() == {"c": [3]}
    bt.from_pydict({"id": [4, 5]}).write.table("sales.orders", mode="append", session=s)
    assert s.sql("SELECT c FROM n").to_pydict() == {"c": [5]}


def test_view_cannot_reference_itself():
    s = _session(w={"a": [1]})
    with pytest.raises(PlanError, match="cannot reference itself"):
        s.sql("CREATE OR REPLACE VIEW w2 AS SELECT * FROM W2")


def test_view_is_not_a_dml_target():
    s = _session(t={"x": [1]})
    s.sql("CREATE VIEW v AS SELECT x FROM t")
    with pytest.raises(PlanError, match="is a view"):
        s.sql("INSERT INTO v VALUES (2)")


# --- CREATE VIEW / TABLE column lists ------------------------------------------------------


def test_view_column_list_names_the_view_and_its_columns():
    s = bt.Session()
    s.sql("CREATE VIEW vc(a, b) AS SELECT 1, 2")
    assert s.list() == ["vc"]
    assert s.sql("SELECT a, b FROM vc").to_pydict() == {"a": [1], "b": [2]}


def test_view_column_list_may_be_shorter_but_not_longer():
    s = bt.Session()
    s.sql("CREATE VIEW vc(a) AS SELECT 1 AS x, 2 AS y")
    assert s.table("vc").columns == ["a", "y"]
    with pytest.raises(PlanError, match="names 3 columns but its query returns 2"):
        s.sql("CREATE VIEW vd(a, b, c) AS SELECT 1, 2")


# --- USE decides where an unqualified CREATE TABLE lands --------------------------------------


def test_create_table_after_use_lands_in_the_catalog(tmp_path):
    s = bt.Session()
    wh = bt.Catalog.from_directory(str(tmp_path / "wh"), name="wh")
    s.catalog.attach(wh)
    wh.create_namespace("raw")
    s.sql("USE wh.raw")
    s.sql("CREATE TABLE ev AS SELECT 1 AS k")
    s.sql("INSERT INTO ev VALUES (2)")
    assert "ev" not in s  # not a session name
    assert wh.has_table("raw.ev")
    assert sorted(s.sql("SELECT k FROM wh.raw.ev").to_pydict()["k"]) == [1, 2]


def test_create_table_without_use_stays_a_session_table():
    s = bt.Session()
    s.sql("CREATE TABLE plain AS SELECT 1 AS z")
    assert s.list() == ["plain"]
    assert not s.catalog.has_table("plain")


# --- one case-insensitive namespace -----------------------------------------------------------


def test_session_names_are_case_insensitive_everywhere():
    s = _session(MyTab={"Col": [1]})
    assert "mytab" in s
    assert s.table("MYTAB").columns == ["Col"]
    assert s.sql("SELECT col FROM mytab").to_pydict() == {"Col": [1]}
    s.register("MYTAB", bt.from_pydict({"Col": [2]}))
    assert s.list() == ["MYTAB"]
    s.drop("mytab")
    assert s.list() == []


def test_create_table_refuses_a_name_differing_only_in_case():
    s = _session(MyTab={"a": [1]})
    with pytest.raises(PlanError, match="already exists"):
        s.sql("CREATE TABLE MYTAB AS SELECT 2 AS a")
    s.sql("CREATE OR REPLACE TABLE MYTAB AS SELECT 2 AS a")
    assert s.list() == ["MyTab"]
    assert s.sql("SELECT a FROM mytab").to_pydict() == {"a": [2]}


def test_sql_dml_and_drop_resolve_names_case_insensitively():
    s = _session(MyTab={"a": [1, 2]})
    s.sql("DELETE FROM MYTAB WHERE a = 1")
    assert s.table("MyTab").to_pydict() == {"a": [2]}
    s.sql("DROP TABLE mytab")
    assert s.list() == []


def test_catalog_names_are_case_insensitive_in_sql():
    s = bt.Session()
    s.sql("CREATE SCHEMA sales")
    s.catalog.create_table("sales.orders", bt.from_pydict({"id": [1, 2]}))
    assert s.sql("SELECT count(*) AS c FROM SALES.ORDERS").to_pydict() == {"c": [2]}
    assert s.sql("SELECT count(*) AS c FROM Memory.Sales.Orders").to_pydict() == {"c": [2]}
    s.sql("USE SALES")
    assert s.catalog.current_namespace() == "sales"
    assert s.sql("SELECT count(*) AS c FROM ORDERS").to_pydict() == {"c": [2]}


def test_unqualified_column_two_sources_share_is_ambiguous():
    s = _session(a={"id": [1], "x": [1]}, b={"id": [1], "y": [2]})
    with pytest.raises(PlanError, match='ambiguous reference to column name "id"'):
        s.sql("SELECT id FROM a JOIN b ON a.id = b.id")
    # The spellings that are not ambiguous keep working.
    assert s.sql("SELECT a.id, x, y FROM a JOIN b ON a.id = b.id").to_pydict()["id"] == [1]
    assert s.sql("SELECT id FROM a JOIN b USING (id)").to_pydict() == {"id": [1]}
    assert s.sql("SELECT a.id AS id FROM a JOIN b ON a.id = b.id ORDER BY id").count() == 1


def test_drop_is_kind_checked():
    s = _session(tt={"a": [1]})
    s.sql("CREATE VIEW vv AS SELECT a FROM tt")
    with pytest.raises(PlanError, match="'tt' is a table, not a view"):
        s.sql("DROP VIEW tt")
    with pytest.raises(PlanError, match="'vv' is a view, not a table"):
        s.sql("DROP TABLE vv")
    s.sql("DROP VIEW vv")
    s.sql("DROP TABLE tt")
    assert s.list() == []
    s.sql("DROP VIEW IF EXISTS vv")


# --- DML on catalog tables --------------------------------------------------------------------


def test_delete_and_update_rewrite_a_catalog_table():
    s = bt.Session()
    s.sql("CREATE SCHEMA sales")
    s.catalog.create_table("sales.orders", bt.from_pydict({"id": [1, 2, 3]}))
    s.sql("DELETE FROM sales.orders WHERE id = 1")
    s.sql("UPDATE sales.orders SET id = id * 10 WHERE id = 2")
    assert sorted(s.table("sales.orders").to_pydict()["id"]) == [3, 20]


def test_merge_into_a_catalog_table_names_the_alternative():
    s = bt.Session()
    s.catalog.create_table("orders", bt.from_pydict({"id": [1]}))
    s.register("src", bt.from_pydict({"id": [2]}))
    with pytest.raises(PlanError, match="MERGE into the catalog table"):
        s.sql(
            "MERGE INTO orders USING src ON orders.id = src.id "
            "WHEN NOT MATCHED THEN INSERT VALUES (src.id)"
        )


# --- nothing runs while the SQL is translated ---------------------------------------------------


class _Counter:
    """A `map_batches` callback counting how many batches it has seen."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return batch


_LAZY = [
    "WITH a AS (SELECT g, sum(v) AS s FROM t GROUP BY g) SELECT x.g FROM a x JOIN a y ON x.s = y.s",
    "WITH a AS (SELECT g, max(v) AS m FROM t GROUP BY g) "
    "SELECT g FROM a WHERE m IN (SELECT m FROM a)",
    "SELECT v, EXISTS (SELECT 1 FROM t t2 WHERE t2.g = t.g AND t2.v > 2) AS e FROM t",
    "SELECT v FROM t t1 WHERE v = (SELECT max(v) FROM t t2 WHERE t2.g = t1.g)",
    "SELECT v FROM t t1 WHERE v NOT IN (SELECT v FROM t t2 WHERE t2.g = t1.g AND t2.v > 1)",
    "SELECT v, (SELECT count(*) + 1 FROM t t2 WHERE t2.g = t1.g AND t2.v > 1) AS n FROM t t1",
    "SELECT v FROM t WHERE v IN (SELECT v FROM t WHERE v > 1)",
]


@pytest.mark.parametrize("query", _LAZY)
def test_translation_runs_no_work(query):
    """`Session.sql` builds a plan; the source runs only at the terminal op.

    The positive control is the second half: the same counter does move once the result is
    collected, so a zero before it is a measurement rather than a callback never wired in.
    """
    counter = _Counter()
    s = bt.Session()
    source = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]}).map_batches(counter)
    s.register("t", source)
    ds = s.sql(query)
    assert counter.calls == 0, f"translating ran the source {counter.calls} time(s)"
    ds.collect()
    assert counter.calls > 0


def test_a_view_does_not_freeze_a_value_computed_at_translation():
    """The uncorrelated scalar is evaluated per query, so the view follows its base table."""
    s = bt.Session()
    s.register("t", bt.from_pydict({"v": [1, 2, 3]}))
    s.sql("CREATE VIEW above AS SELECT v FROM t WHERE v > (SELECT avg(v) FROM t)")
    assert s.sql("SELECT v FROM above").to_pydict() == {"v": [3]}
    s.register("t", bt.from_pydict({"v": [10, 20, 30, 40]}))
    assert sorted(s.sql("SELECT v FROM above").to_pydict()["v"]) == [30, 40]


_STAYS_EAGER = [
    # Documented in the Session.sql docstring and docs/user-guide/analyze/sql.md.
    "WITH RECURSIVE r(n) AS (SELECT min(v) FROM t UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
    "SELECT n FROM r",
    "SELECT g, sum(v) FROM t GROUP BY g HAVING sum(v) > (SELECT avg(v) FROM t)",
    "SELECT v FROM t WHERE v > (SELECT avg(v) FROM t)",
    "SELECT v FROM t WHERE EXISTS (SELECT 1 FROM t WHERE v > 1)",
    "SELECT v FROM t WHERE v NOT IN (SELECT v FROM t WHERE v > 1)",
    "SELECT v FROM t WHERE v > 5 OR v IN (SELECT v FROM t WHERE v > 1)",
]


@pytest.mark.parametrize("query", _STAYS_EAGER)
def test_the_documented_eager_shapes_still_run_at_translation(query):
    """The exceptions the docs list are real, so the docs' claim is checked, not assumed."""
    counter = _Counter()
    s = bt.Session()
    s.register("t", bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}).map_batches(counter))
    s.sql(query)
    assert counter.calls > 0


# --- RocksDB lookups open read-only -------------------------------------------------------------


class _FakeRocksdict:
    """Stands in for `rocksdict`, which is not installed here, recording how it was opened."""

    class AccessType:
        @staticmethod
        def read_only():
            return "read-only"

    def __init__(self) -> None:
        self.opened: list[tuple[str, object]] = []
        self.Rdict = self._open  # rocksdict's entry point is a class named `Rdict`

    def _open(self, path, access_type=None):
        self.opened.append((path, access_type))
        return object()


def test_rocksdb_lookup_refuses_a_missing_path(monkeypatch, tmp_path):
    from batcher.io.lookup import backends

    fake = _FakeRocksdict()
    monkeypatch.setattr(backends, "require", lambda *a, **k: fake)
    with pytest.raises(ConfigError, match="does not exist"):
        backends.RocksDBLookup(str(tmp_path / "typo"), pa.schema([("v", pa.int64())]))
    assert fake.opened == []  # never reached the driver, which would have created it


def test_rocksdb_lookup_opens_an_existing_database_read_only(monkeypatch, tmp_path):
    from batcher.io.lookup import backends

    fake = _FakeRocksdict()
    monkeypatch.setattr(backends, "require", lambda *a, **k: fake)
    backends.RocksDBLookup(str(tmp_path), pa.schema([("v", pa.int64())]))
    assert fake.opened == [(str(tmp_path), "read-only")]
