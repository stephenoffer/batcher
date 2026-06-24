"""Unit tests for the SQL `Session` catalog/registry and UDF scoping (no execution)."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.mark.unit
def test_register_and_list_tables():
    s = bt.Session()
    s.register("nums", bt.from_pydict({"v": [1, 2, 3]}))
    s.register("more", pa.table({"w": [4]}))
    assert s.list() == ["more", "nums"]
    assert s.table("nums").columns == ["v"]


@pytest.mark.unit
def test_table_missing_raises():
    s = bt.Session()
    with pytest.raises(PlanError, match="no table"):
        s.table("absent")


@pytest.mark.unit
def test_drop_and_clear():
    s = bt.Session()
    s.register("t", bt.from_pydict({"v": [1]}))
    s.drop("t")
    assert s.list() == []
    s.register("t", bt.from_pydict({"v": [1]}))
    s.clear()
    assert s.list() == []


@pytest.mark.unit
def test_register_functions_listed():
    s = bt.Session()
    s.register_function("f", lambda a: a)
    s.register_function("g", lambda a: a, table=True, output_columns=["a"])
    assert s.list_functions() == ["f", "g"]


@pytest.mark.unit
def test_dataset_sql_binds_self():
    ds = bt.from_pydict({"a": [1, 2, 3]})
    out = ds.sql("SELECT a, a * 2 AS d FROM self")
    assert out.columns == ["a", "d"]


@pytest.mark.unit
def test_create_duplicate_table_raises():
    s = bt.Session()
    s.register("src", bt.from_pydict({"v": [1, 2]}))
    s.sql("CREATE TABLE t AS SELECT v FROM src")
    with pytest.raises(PlanError, match="already exists"):
        s.sql("CREATE TABLE t AS SELECT v FROM src")


@pytest.mark.unit
def test_create_or_replace_overwrites():
    s = bt.Session()
    s.register("src", bt.from_pydict({"v": [1, 2]}))
    s.sql("CREATE VIEW t AS SELECT v FROM src")
    s.sql("CREATE OR REPLACE VIEW t AS SELECT v AS w FROM src")
    assert s.table("t").columns == ["w"]


@pytest.mark.unit
def test_drop_missing_raises():
    s = bt.Session()
    with pytest.raises(PlanError, match="no table"):
        s.sql("DROP TABLE absent")
    # IF EXISTS is a no-op, not an error.
    s.sql("DROP TABLE IF EXISTS absent")


@pytest.mark.unit
def test_star_does_not_leak_internal_udf_columns():
    s = bt.Session()
    s.register("t", bt.from_pydict({"id": [1, 2, 3], "x": [10, 20, 30]}))
    s.register_function("dbl", lambda a: a)
    # The synthetic hoist column must not appear in `SELECT *`.
    assert s.sql("SELECT * FROM t WHERE dbl(x) > 0").columns == ["id", "x"]
    assert s.sql("SELECT *, dbl(x) AS y FROM t").columns == ["id", "x", "y"]


@pytest.mark.unit
def test_unknown_function_error_names_it():
    s = bt.Session()
    s.register("t", bt.from_pydict({"x": [1]}))
    with pytest.raises(NotImplementedError, match="unknown function 'nope'"):
        s.sql("SELECT nope(x) FROM t")


@pytest.mark.unit
def test_value_window_explicit_frame_rejected():
    s = bt.Session()
    s.register("t", bt.from_pydict({"t": [1, 2, 3], "x": [5, 6, 7]}))
    with pytest.raises(NotImplementedError, match="FIRST_VALUE / LAST_VALUE"):
        s.sql(
            "SELECT LAST_VALUE(x) OVER "
            "(ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS v FROM t"
        )


@pytest.mark.unit
def test_residual_join_ambiguous_column_rejected():
    s = bt.Session()
    s.register("a", bt.from_pydict({"k": [1, 2], "v": [10, 20]}))
    s.register("b", bt.from_pydict({"k": [1, 2], "v": [15, 25]}))
    # `a.v < b.v` would lose its qualifiers post-join (both are `v`) — reject it.
    with pytest.raises(NotImplementedError, match="present on both sides"):
        s.sql("SELECT a.k AS k FROM a JOIN b ON a.k = b.k AND a.v < b.v")


@pytest.mark.unit
def test_unknown_table_function_raises():
    s = bt.Session()
    s.register("t", bt.from_pydict({"v": [1]}))
    with pytest.raises(PlanError, match="unknown table function"):
        s.sql("SELECT * FROM missing_fn(t)")


@pytest.mark.unit
def test_scalar_udf_in_from_raises():
    s = bt.Session()
    s.register("t", bt.from_pydict({"v": [1]}))
    s.register_function("scal", lambda a: a)
    with pytest.raises(PlanError, match="scalar function"):
        s.sql("SELECT * FROM scal(t)")


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "SELECT g, SUM(udf(v)) AS s FROM t GROUP BY g",
        "SELECT g, SUM(v) AS s FROM t GROUP BY g HAVING SUM(udf(v)) > 0",
        "SELECT g, COUNT(*) AS n FROM t GROUP BY g ORDER BY udf(g)",
    ],
)
def test_scalar_udf_rejected_in_agg_positions(query):
    s = bt.Session()
    s.register("t", bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}))
    s.register_function("udf", lambda a: a)
    with pytest.raises(PlanError, match="registered scalar function"):
        s.sql(query)
