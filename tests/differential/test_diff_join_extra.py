"""Differential tests vs DuckDB for the `join_extra` structural join rewrites.

Each query runs through the full optimizer (so the rewrite fires) and its result
must match DuckDB computing the same logical join. Tables carry duplicate keys and
NULL keys so the rewrites are checked to preserve row multiplicity and NULL-key
join semantics, not just the happy path.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.join_extra
from _harness import assert_same
from batcher.api.dataset.frame import Dataset


def _a() -> pa.Table:
    # Duplicate key (2 appears twice) and a NULL key row.
    return pa.table({"k": [1, 2, 2, None, 3], "v": [10, 20, 21, 99, 30]})


def _b() -> pa.Table:
    return pa.table({"k": [1, 2], "w": [1, 2]})


def _register(duck, a: pa.Table, b: pa.Table) -> None:
    duck.register("a", a)
    duck.register("b", b)


# --- semi/anti empty left ----------------------------------------------------


def test_semi_empty_left(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = bt.from_arrow(a).limit(0).join(bt.from_arrow(b), on="k", how="semi").collect()
    expected = duck.sql(
        "SELECT t.k, t.v FROM (SELECT * FROM a LIMIT 0) t "
        "WHERE EXISTS (SELECT 1 FROM b WHERE b.k = t.k)"
    )
    assert_same(out, expected)


def test_anti_empty_left(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = bt.from_arrow(a).limit(0).join(bt.from_arrow(b), on="k", how="anti").collect()
    expected = duck.sql(
        "SELECT t.k, t.v FROM (SELECT * FROM a LIMIT 0) t "
        "WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.k = t.k)"
    )
    assert_same(out, expected)


# --- semi empty right (→ empty) ----------------------------------------------


def test_semi_empty_right(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = bt.from_arrow(a).join(bt.from_arrow(b).limit(0), on="k", how="semi").collect()
    expected = duck.sql(
        "SELECT a.k, a.v FROM a "
        "WHERE EXISTS (SELECT 1 FROM (SELECT * FROM b LIMIT 0) bb WHERE bb.k = a.k)"
    )
    assert_same(out, expected)


# --- anti empty right (→ all left rows, dups + null preserved) ---------------


def test_anti_empty_right(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = bt.from_arrow(a).join(bt.from_arrow(b).limit(0), on="k", how="anti").collect()
    expected = duck.sql(
        "SELECT a.k, a.v FROM a "
        "WHERE NOT EXISTS (SELECT 1 FROM (SELECT * FROM b LIMIT 0) bb WHERE bb.k = a.k)"
    )
    assert_same(out, expected)


# --- inner join empty side (single-sided output → empty) ---------------------


def test_inner_empty_right(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = (
        bt.from_arrow(a)
        .join(bt.from_arrow(b).limit(0), on="k", how="inner")
        .select("k", "v")
        .collect()
    )
    expected = duck.sql("SELECT a.k, a.v FROM a JOIN (SELECT * FROM b LIMIT 0) bb ON a.k = bb.k")
    assert_same(out, expected)


# --- outer joins with an empty null-supplying side ---------------------------


def test_left_join_empty_right(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = (
        bt.from_arrow(a)
        .join(bt.from_arrow(b).limit(0), on="k", how="left")
        .select("k", "v")
        .collect()
    )
    expected = duck.sql(
        "SELECT a.k, a.v FROM a LEFT JOIN (SELECT * FROM b LIMIT 0) bb ON a.k = bb.k"
    )
    assert_same(out, expected)


def test_right_join_empty_left(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = (
        bt.from_arrow(a)
        .limit(0)
        .join(bt.from_arrow(b), on="k", how="right")
        .select("k", "w")
        .collect()
    )
    expected = duck.sql(
        "SELECT b.k, b.w FROM (SELECT * FROM a LIMIT 0) aa RIGHT JOIN b ON aa.k = b.k"
    )
    assert_same(out, expected)


def test_full_join_empty_right(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    out = bt.from_arrow(a).join(bt.from_arrow(b).limit(0), on="k", how="full").select("v").collect()
    expected = duck.sql("SELECT a.v FROM a FULL JOIN (SELECT * FROM b LIMIT 0) bb ON a.k = bb.k")
    assert_same(out, expected)


# --- dedup_join_keys (hand-built duplicate key pair) -------------------------


def test_dedup_join_keys(duck):
    a, b = _a(), _b()
    _register(duck, a, b)
    base = bt.from_arrow(a).join(bt.from_arrow(b), on="k", how="inner")
    join = base._plan
    dup = dataclasses.replace(
        join,
        left_keys=join.left_keys + join.left_keys,
        right_keys=join.right_keys + join.right_keys,
    )
    out = Dataset(dup, base._sources).collect()
    expected = duck.sql("SELECT a.k AS k, a.v AS v, b.w AS w FROM a JOIN b ON a.k = b.k")
    assert_same(out, expected)
