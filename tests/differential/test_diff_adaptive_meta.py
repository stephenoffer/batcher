"""Differential tests vs DuckDB for the `adaptive_meta` rules.

Each rule rewrites the plan on a *provably-EXACT* cardinality; the result must stay
byte-identical to DuckDB. Covered: an inert `LIMIT` dropped over a fully-known source,
an `OFFSET` past the end folded to empty, and a schema-preserving chain over an empty
source folded away — with nulls and empty inputs where relevant.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.adaptive_meta  # registers the rules into DEFAULT_REGISTRY
from batcher import col


def _reg(duck, name, table):
    duck.register(name, table)
    return bt.from_arrow(table)


# --- drop_inert_limit ----------------------------------------------------------


def test_inert_limit_returns_all_rows(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3, None, 5], "y": [10, 20, 30, 40, 50]})
    ds = _reg(duck, "t", t)
    # LIMIT 100 over 5 exact rows → the cap is inert; every row (incl. the null) stays.
    out = ds.limit(100).collect()
    assert_same(out, duck.sql("SELECT * FROM t LIMIT 100"))


def test_inert_limit_at_exact_row_count(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    ds = _reg(duck, "t2", t)
    out = ds.limit(3).collect()  # cap == row count → still all rows
    assert_same(out, duck.sql("SELECT * FROM t2 LIMIT 3"))


def test_inert_limit_over_global_aggregate(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3, 4]})
    ds = _reg(duck, "t3", t)
    # A global aggregate is EXACTLY one row, so LIMIT 5 over it is inert.
    out = ds.agg(n=col("x").count(), s=col("x").sum()).limit(5).collect()
    assert_same(out, duck.sql("SELECT count(x) AS n, sum(x) AS s FROM t3 LIMIT 5"))


# --- empty_limit_past_offset ---------------------------------------------------


def test_offset_past_end_is_empty(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3], "y": [10, 20, 30]})
    ds = _reg(duck, "t4", t)
    # OFFSET 3 over 3 rows → window opens past the last row → empty.
    out = ds.limit(10, offset=3).collect()
    assert_same(out, duck.sql("SELECT * FROM t4 LIMIT 10 OFFSET 3"))


def test_offset_well_past_end_is_empty(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2], "y": ["a", "b"]})
    ds = _reg(duck, "t5", t)
    out = ds.limit(5, offset=100).collect()
    assert_same(out, duck.sql("SELECT * FROM t5 LIMIT 5 OFFSET 100"))


# --- fold_exact_empty_input ----------------------------------------------------


def test_filter_sort_over_empty_source(duck):
    from conftest import assert_same

    empty = pa.table({"x": pa.array([], pa.int64()), "y": pa.array([], pa.int64())})
    ds = _reg(duck, "e1", empty)
    out = ds.filter(col("x") > 1).sort("x").collect()
    assert_same(out, duck.sql("SELECT * FROM e1 WHERE x > 1 ORDER BY x"))


def test_distinct_over_empty_source(duck):
    from conftest import assert_same

    empty = pa.table({"x": pa.array([], pa.int64()), "y": pa.array([], pa.string())})
    ds = _reg(duck, "e2", empty)
    out = ds.filter(col("x") > 1).distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT * FROM e2 WHERE x > 1"))
