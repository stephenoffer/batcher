"""`LIKE 'prefix%'` → range rewrite vs DuckDB.

The rewrite must be exact for a pure prefix (and a no-op for patterns it can't
safely convert), so results match DuckDB across empties, the bare-prefix match,
non-ASCII data, and patterns with mid-string or `_` wildcards (left as LIKE).
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _t(duck, names):
    t = pa.table({"name": names, "v": list(range(len(names)))})
    duck.register("t", t)
    return bt.from_arrow(t)


def test_prefix_match(duck):
    from conftest import assert_same

    ds = _t(duck, ["apple", "apricot", "banana", "applet", "ap", "BANANA", "az"])
    assert_same(
        ds.filter(col("name").str.like("ap%")).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'ap%'"),
    )


def test_full_string_prefix(duck):
    from conftest import assert_same

    ds = _t(duck, ["cat", "category", "cab", "dog"])
    assert_same(
        ds.filter(col("name").str.like("cat%")).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'cat%'"),
    )


def test_non_ascii_data(duck):
    from conftest import assert_same

    ds = _t(duck, ["café", "cabana", "cat", "caña", "dog"])
    assert_same(
        ds.filter(col("name").str.like("ca%")).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'ca%'"),
    )


def test_underscore_wildcard_left_as_like(duck):
    from conftest import assert_same

    ds = _t(duck, ["a1x", "a2x", "abx", "a1", "b1x"])
    assert_same(
        ds.filter(col("name").str.like("a_x%")).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'a_x%'"),
    )


def test_mid_string_wildcard_left_as_like(duck):
    from conftest import assert_same

    ds = _t(duck, ["axb", "ayb", "axc", "azb"])
    assert_same(
        ds.filter(col("name").str.like("a%b")).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'a%b'"),
    )


def test_prefix_in_conjunction(duck):
    from conftest import assert_same

    ds = _t(duck, ["apple", "apricot", "avocado", "banana"])
    assert_same(
        ds.filter(col("name").str.like("ap%") & (col("v") > 0)).collect(),
        duck.sql("SELECT * FROM t WHERE name LIKE 'ap%' AND v > 0"),
    )
