"""Predicate-inference rules preserve results vs DuckDB.

Every rewrite in `kyber.rules.predicate_infer` (bound tightening, contradiction →
empty, IN-list refinement, redundant-conjunct removal, transitive inference) must leave
the query result byte-for-byte identical. These cases run the query through the full
optimizer (the rules register on import) and compare against DuckDB, covering NULL rows
and empty inputs where they matter under three-valued logic.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt

# Importing the module registers its @rule decorators into the default registry, so the
# full Optimizer that `.collect()` runs will apply them.
import batcher.kyber.rules.extra.predicate_infer
from _harness import assert_same
from batcher import col
from batcher.plan.expr_ir import InList


def _t(duck, rows=None):
    a = rows if rows is not None else [1, 5, 5, 8, 3, None]
    b = list(range(len(a)))
    t = pa.table({"a": pa.array(a, type=pa.int64()), "b": pa.array(b, type=pa.int64())})
    duck.register("t", t)
    return bt.from_arrow(t)


def _multi(duck):
    t = pa.table(
        {
            "a": [1, 2, 3, 4, None],
            "b": [2, 3, 4, 5, 5],
            "c": [3, 4, 5, 6, 6],
        }
    )
    duck.register("m", t)
    return bt.from_arrow(t)


def test_tighten_bounds(duck):
    ds = _t(duck).filter((col("a") > 3) & (col("a") > 1))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a > 3 AND a > 1"))


def test_range_contradiction_empty(duck):
    ds = _t(duck).filter((col("a") > 5) & (col("a") < 3))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a > 5 AND a < 3"))


def test_equality_contradiction_empty(duck):
    ds = _t(duck).filter((col("a") == 1) & (col("a") == 2))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a = 1 AND a = 2"))


def test_eq_vs_range_contradiction(duck):
    ds = _t(duck).filter((col("a") == 1) & (col("a") > 5))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a = 1 AND a > 5"))


def test_eq_neq_contradiction(duck):
    ds = _t(duck).filter((col("a") == 5) & (col("a") != 5))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a = 5 AND a <> 5"))


def test_bound_dominated_neq(duck):
    ds = _t(duck).filter((col("a") > 5) & (col("a") != 3))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a > 5 AND a <> 3"))


def test_redundant_is_not_null_with_nulls(duck):
    ds = _t(duck).filter((col("a") > 3) & col("a").is_not_null())
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a > 3 AND a IS NOT NULL"))


def test_in_list_refined_by_comparison(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3, 5, 8)) & (col("a") > 3))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3, 5, 8) AND a > 3"))


def test_in_list_refined_by_equality(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3, 5)) & (col("a") == 5))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3, 5) AND a = 5"))


def test_in_list_equality_absent_is_empty(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3, 5)) & (col("a") == 7))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3, 5) AND a = 7"))


def test_in_list_refined_by_neq(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3, 5)) & (col("a") != 3))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3, 5) AND a <> 3"))


def test_intersect_in_lists(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3, 5)) & InList(col("a"), (3, 5, 8)))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3, 5) AND a IN (3, 5, 8)"))


def test_intersect_in_lists_disjoint_is_empty(duck):
    ds = _t(duck).filter(InList(col("a"), (1, 3)) & InList(col("a"), (5, 8)))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (1, 3) AND a IN (5, 8)"))


def test_singleton_in_list(duck):
    ds = _t(duck).filter(InList(col("a"), (5,)))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a IN (5)"))


def test_transitive_comparison_chain(duck):
    ds = _multi(duck).filter((col("a") < col("b")) & (col("b") < col("c")))
    assert_same(ds.collect(), duck.sql("SELECT * FROM m WHERE a < b AND b < c"))


def test_empty_input_stays_empty(duck):
    ds = _t(duck, rows=[]).filter((col("a") > 3) & (col("a") > 1))
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE a > 3 AND a > 1"))
