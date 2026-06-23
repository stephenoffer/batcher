"""Differential tests for empty (zero-row) input against DuckDB.

Empty input is a first-class contract edge (see .claude/rules/testing.md): every
operator must handle a relation with no rows the way DuckDB does. Constructing an
empty in-memory Dataset is itself exercised here — ``from_arrow`` preserves a
zero-row table's schema via an empty morsel, so these pipelines type-check and run.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count

_SCHEMA = pa.schema([("dept", pa.string()), ("salary", pa.int64()), ("bonus", pa.float64())])


def _empty() -> pa.Table:
    return pa.table(
        {
            "dept": pa.array([], pa.string()),
            "salary": pa.array([], pa.int64()),
            "bonus": pa.array([], pa.float64()),
        },
        schema=_SCHEMA,
    )


def test_empty_dataset_constructs_and_collects():
    """An empty table round-trips through the engine, keeping its columns and types."""
    out = bt.from_arrow(_empty()).collect()
    assert out.num_rows == 0
    # Column order is not part of the collect() contract; compare the schema as a set.
    assert dict(zip(out.schema.names, out.schema.types, strict=True)) == {
        "dept": pa.string(),
        "salary": pa.int64(),
        "bonus": pa.float64(),
    }


def test_empty_filter_project_vs_duckdb(duck):
    from conftest import assert_same

    t = _empty()
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("salary") > 100).select("dept", net=col("salary")).collect()
    expected = duck.sql("SELECT dept, salary AS net FROM t WHERE salary > 100")
    assert_same(out, expected)


def test_empty_group_by_vs_duckdb(duck):
    from conftest import assert_same

    t = _empty()
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("dept").agg(total=col("salary").sum(), n=count()).collect()
    expected = duck.sql("SELECT dept, SUM(salary) AS total, COUNT(*) AS n FROM t GROUP BY dept")
    assert_same(out, expected)


def test_empty_global_aggregate_vs_duckdb(duck):
    from conftest import assert_same

    t = _empty()
    duck.register("t", t)
    # A global aggregate over no rows yields one row (NULL sum, 0 count) in both engines.
    out = bt.from_arrow(t).group_by().agg(total=col("salary").sum(), n=count()).collect()
    expected = duck.sql("SELECT SUM(salary) AS total, COUNT(*) AS n FROM t")
    assert_same(out, expected)


def test_empty_distinct_sort_limit_vs_duckdb(duck):
    from conftest import assert_same

    t = _empty()
    duck.register("t", t)
    out = bt.from_arrow(t).select("dept", "salary").distinct().sort("salary").limit(5).collect()
    expected = duck.sql("SELECT DISTINCT dept, salary FROM t ORDER BY salary LIMIT 5")
    assert_same(out, expected)
