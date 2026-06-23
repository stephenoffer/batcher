"""Phase-2 type/value hygiene: null-safe equality, value remap, fill strategies.

All lower to existing IR (CASE / window-aggregate + coalesce), so each is checked
against the equivalent DuckDB expression.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_eq_missing_is_not_distinct_from(duck):
    """`eq_missing` matches SQL ``IS NOT DISTINCT FROM`` (two nulls equal, null vs
    value is false)."""
    from conftest import assert_same

    t = pa.table(
        {
            "a": pa.array([1, 2, None, None, 5], pa.int64()),
            "b": pa.array([1, 9, None, 4, None], pa.int64()),
        }
    )
    duck.register("em", t)
    out = bt.from_arrow(t).select(eq=col("a").eq_missing(col("b"))).collect()
    assert_same(out, duck.sql("SELECT (a IS NOT DISTINCT FROM b) eq FROM em"))


def test_replace_maps_values(duck):
    """`replace({old: new})` remaps listed values, keeping the rest unchanged."""
    from conftest import assert_same

    t = pa.table({"c": ["US", "UK", "FR", "US", None]})
    duck.register("rp", t)
    out = bt.from_arrow(t).select(c=col("c").replace({"US": "USA", "UK": "GBR"})).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN c = 'US' THEN 'USA' WHEN c = 'UK' THEN 'GBR' ELSE c END c FROM rp"
        ),
    )


def test_replace_with_default(duck):
    """A `default` replaces every unmapped (incl. unmatched) value."""
    from conftest import assert_same

    t = pa.table({"c": ["a", "b", "z", None]})
    duck.register("rpd", t)
    out = bt.from_arrow(t).select(c=col("c").replace({"a": "1", "b": "2"}, default="?")).collect()
    assert_same(
        out,
        duck.sql("SELECT CASE WHEN c = 'a' THEN '1' WHEN c = 'b' THEN '2' ELSE '?' END c FROM rpd"),
    )


def test_fill_null_strategy_mean(duck):
    """`fill_null(strategy="mean")` fills nulls with the column mean (whole relation)."""
    from conftest import assert_same

    t = pa.table({"v": pa.array([1.0, 3.0, None, 5.0, None], pa.float64())})
    duck.register("fm", t)
    out = bt.from_arrow(t).fill_null(strategy="mean").collect()
    assert_same(out, duck.sql("SELECT COALESCE(v, AVG(v) OVER ()) v FROM fm"))


def test_normalize_whitespace(duck):
    """`str.normalize_whitespace` collapses whitespace runs and trims (vs DuckDB)."""
    from conftest import assert_same

    t = pa.table({"s": ["  hello   world ", "a  b", "no-change", None]})
    duck.register("nw", t)
    out = bt.from_arrow(t).select(s=col("s").str.normalize_whitespace()).collect()
    assert_same(out, duck.sql(r"SELECT trim(regexp_replace(s, '\s+', ' ', 'g')) s FROM nw"))


def test_fill_null_strategy_max_subset(duck):
    """A strategy fill restricted to `subset`."""
    from conftest import assert_same

    t = pa.table(
        {
            "a": pa.array([1, None, 3], pa.int64()),
            "b": pa.array([None, 2, None], pa.int64()),
        }
    )
    duck.register("fmx", t)
    out = bt.from_arrow(t).fill_null(strategy="max", subset=["a"]).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, MAX(a) OVER ()) a, b FROM fmx"))
