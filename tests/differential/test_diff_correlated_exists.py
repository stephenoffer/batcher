"""Correlated EXISTS / NOT EXISTS with a `<>` residual decorrelates correctly vs DuckDB.

`EXISTS (SELECT * FROM inner WHERE inner.k = outer.k AND inner.c <> outer.c [AND local])`
cannot become a plain equi-semijoin (the `<>` correlates on a value, not a key). It
decorrelates to `group_min/max(c) per k` joined back and bound-tested — the shape of TPC-H
q21's two `EXISTS`/`NOT EXISTS` clauses, and distributed-safe (group-by + join, no row id).
The edges: the empty correlated group (NOT EXISTS must be true), a `local` filter on the
inner (q21's late-receipt condition), and a random multiset checked against DuckDB.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import batcher as bt
from _harness import assert_same


def _tbl(duck):
    tbl = pa.table(
        {
            "ok": pa.array([1, 1, 2, 2, 3, 3, 4, 4], type=pa.int64()),
            "sk": pa.array([10, 20, 21, 22, 30, 31, 40, 40], type=pa.int64()),
            "late": pa.array([True, False, True, True, True, True, True, False]),
        }
    )
    duck.register("l", tbl)
    return bt.from_arrow(tbl)


def test_q21_shaped_exists_and_not_exists(duck):
    ds = _tbl(duck)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM l l1 WHERE l1.late "
        "AND EXISTS (SELECT * FROM l l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk) "
        "AND NOT EXISTS (SELECT * FROM l l3 WHERE l3.ok=l1.ok AND l3.sk<>l1.sk AND l3.late)"
    )
    assert_same(bt.sql(sql, l=ds).collect(), duck.sql(sql))


def test_correlated_neq_exists_only(duck):
    ds = _tbl(duck)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM l l1 "
        "WHERE EXISTS (SELECT * FROM l l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk)"
    )
    assert_same(bt.sql(sql, l=ds).collect(), duck.sql(sql))


def test_correlated_neq_not_exists_only(duck):
    # NOT EXISTS is true for an order with a single distinct supplier (empty `<>` group).
    ds = _tbl(duck)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM l l1 "
        "WHERE NOT EXISTS (SELECT * FROM l l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk)"
    )
    assert_same(bt.sql(sql, l=ds).collect(), duck.sql(sql))


def test_fusion_both_subqueries_filtered(duck):
    # Two correlated-`<>` subqueries over the SAME base table, each with its OWN local
    # filter (`late` vs `sk > 20`), fuse into one group-by with two conditional min/max
    # pairs (two flag columns). Must still match DuckDB.
    ds = _tbl(duck)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM l l1 "
        "WHERE EXISTS (SELECT * FROM l l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk AND l2.late) "
        "AND NOT EXISTS (SELECT * FROM l l3 WHERE l3.ok=l1.ok AND l3.sk<>l1.sk AND l3.sk>20)"
    )
    assert_same(bt.sql(sql, l=ds).collect(), duck.sql(sql))


def test_fusion_falls_back_on_distinct_tables(duck):
    # Two correlated-`<>` subqueries over DIFFERENT base tables cannot share a scan, so
    # fusion must decline and each decorrelates independently — still correct vs DuckDB.
    a = pa.table({"ok": [1, 1, 2, 3], "sk": [10, 20, 30, 40], "late": [True, True, True, False]})
    b = pa.table({"ok": [1, 2, 2, 3], "sk": [11, 30, 31, 40], "late": [True, True, False, True]})
    duck.register("a", a)
    duck.register("b", b)
    da, db = bt.from_arrow(a), bt.from_arrow(b)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM a l1 WHERE l1.late "
        "AND EXISTS (SELECT * FROM a l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk) "
        "AND NOT EXISTS (SELECT * FROM b l3 WHERE l3.ok=l1.ok AND l3.sk<>l1.sk AND l3.late)"
    )
    assert_same(bt.sql(sql, a=da, b=db).collect(), duck.sql(sql))


def test_correlated_neq_random_multiset(duck):
    rng = np.random.default_rng(11)
    n = 4000
    tbl = pa.table(
        {
            "ok": pa.array(rng.integers(0, 500, n), type=pa.int64()),
            "sk": pa.array(rng.integers(0, 9, n), type=pa.int64()),
            "late": pa.array(rng.integers(0, 2, n).astype(bool)),
        }
    )
    duck.register("r", tbl)
    ds = bt.from_arrow(tbl)
    sql = (
        "SELECT l1.ok AS ok, l1.sk AS sk FROM r l1 WHERE l1.late "
        "AND EXISTS (SELECT * FROM r l2 WHERE l2.ok=l1.ok AND l2.sk<>l1.sk) "
        "AND NOT EXISTS (SELECT * FROM r l3 WHERE l3.ok=l1.ok AND l3.sk<>l1.sk AND l3.late)"
    )
    assert_same(bt.sql(sql, r=ds).collect(), duck.sql(sql))
