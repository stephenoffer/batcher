"""Differential tests for advanced grouping: ROLLUP/CUBE/GROUPING SETS, the
GROUPING() function, aggregate FILTER combined with DISTINCT, and expression /
multi-construct grouping keys — each pinned against DuckDB.

These pin bugs where the SQL grouping path crashed or diverged from DuckDB:

* ``agg(DISTINCT x) FILTER (WHERE c)`` crashed (the guard was wrapped *around* the
  ``DISTINCT`` node, producing invalid ``count(CASE WHEN c THEN DISTINCT x END)``);
  the guard must move *inside* the distinct.
* an expression grouping key inside ``ROLLUP``/``CUBE``/``GROUPING SETS``
  (``ROLLUP(a, b * 10)``) crashed with an empty column name.
* multiple grouping constructs (``GROUP BY ROLLUP(a), ROLLUP(b)``) ignored all but
  the first; DuckDB takes their Cartesian product.
* ``GROUPING(...)`` was entirely unsupported.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": pa.array(["x", "x", "y", "y", "z", None], pa.string()),
            "b": pa.array([1, 1, 2, 2, 1, 1], pa.int64()),
            "c": pa.array(["p", "q", "p", "q", "p", "q"], pa.string()),
            "v": pa.array([10, 20, 30, 40, 50, 60], pa.int64()),
            "w": pa.array([1.0, -0.0, 0.0, float("nan"), 2.0, None], pa.float64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def _run(t, duck, sql):
    from conftest import assert_same

    assert_same(bt.sql(sql, t=bt.from_arrow(t)).collect(), duck.sql(sql))


# --- ROLLUP / CUBE / GROUPING SETS super-aggregate rows ----------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a, b, sum(v) s FROM t GROUP BY ROLLUP(a, b)",
        "SELECT a, b, sum(v) s FROM t GROUP BY CUBE(a, b)",
        "SELECT a, b, sum(v) s FROM t GROUP BY GROUPING SETS ((a,b),(a),())",
        "SELECT sum(v) s FROM t GROUP BY GROUPING SETS (())",
        "SELECT a, sum(v) s FROM t GROUP BY GROUPING SETS ((a),(a))",
        "SELECT a, b, count(*) c, min(v) mn, max(v) mx, avg(v) av FROM t GROUP BY ROLLUP(a, b)",
        "SELECT a, count(DISTINCT b) c FROM t GROUP BY ROLLUP(a)",
        "SELECT b, sum(v) s FROM t GROUP BY ROLLUP(a, b)",
    ],
)
def test_rollup_cube_grouping_sets(t, duck, sql):
    _run(t, duck, sql)


# --- expression and multi-construct grouping keys ----------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # a plain base column alongside a construct
        "SELECT c, a, sum(v) s FROM t GROUP BY c, ROLLUP(a)",
        "SELECT c, a, b, sum(v) s FROM t GROUP BY c, CUBE(a, b)",
        # an expression key inside the construct
        "SELECT a, b * 10 k, sum(v) s FROM t GROUP BY ROLLUP(a, b * 10)",
        "SELECT b + 1 k, sum(v) s FROM t GROUP BY GROUPING SETS ((b + 1), ())",
        # two constructs → Cartesian product of levels
        "SELECT a, b, sum(v) s FROM t GROUP BY ROLLUP(a), ROLLUP(b)",
        "SELECT a, b, c, sum(v) s FROM t GROUP BY CUBE(a, b), c",
    ],
)
def test_expression_and_multi_construct_keys(t, duck, sql):
    _run(t, duck, sql)


# --- GROUPING() function -----------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a, GROUPING(a) g, sum(v) s FROM t GROUP BY ROLLUP(a)",
        "SELECT a, b, GROUPING(a, b) g FROM t GROUP BY CUBE(a, b)",
        "SELECT a, b, GROUPING(b, a) g FROM t GROUP BY CUBE(a, b)",
        "SELECT a, b, GROUPING(a) ga, GROUPING(b) gb FROM t GROUP BY CUBE(a, b)",
        "SELECT a, b, GROUPING(a, b) g, sum(v) s FROM t GROUP BY GROUPING SETS ((a,b),(a),())",
        "SELECT a, GROUPING(a) g, sum(v) s FROM t GROUP BY a",  # plain GROUP BY → 0
        "SELECT a, sum(v) s FROM t GROUP BY ROLLUP(a) HAVING GROUPING(a) = 0",
    ],
)
def test_grouping_function(t, duck, sql):
    _run(t, duck, sql)


# --- aggregate FILTER, incl. combined with DISTINCT --------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a, sum(v) FILTER (WHERE b = 1) s FROM t GROUP BY a",
        "SELECT a, sum(v) FILTER (WHERE v > 1000) s FROM t GROUP BY a",  # matches nothing
        "SELECT a, count(*) FILTER (WHERE v > 25) c FROM t GROUP BY a",
        # DISTINCT + FILTER: the guard must push inside the DISTINCT set.
        "SELECT a, count(DISTINCT b) FILTER (WHERE v > 15) c FROM t GROUP BY a",
        "SELECT count(DISTINCT b) FILTER (WHERE v > 15) c FROM t",
        "SELECT a, count(DISTINCT w) FILTER (WHERE v > 15) c FROM t GROUP BY a",
    ],
)
def test_aggregate_filter(t, duck, sql):
    _run(t, duck, sql)
