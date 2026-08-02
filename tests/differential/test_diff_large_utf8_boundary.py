"""Differential tests for LargeUtf8 normalization at the FFI type boundary.

A `large_string` (LargeUtf8) column must behave identically to a `string` (Utf8)
column — DuckDB treats both as `VARCHAR`. Before the fix, the boundary normalized
narrow numerics and decoded dictionaries but left `LargeUtf8` un-normalized, so the
engine's string kernels (which accept only `Utf8`) crashed on it:

- ``filter(col("s") == "a")`` → "Invalid comparison operation: LargeUtf8 == Utf8"
- ``col("s").str.contains("a")`` / ``str.upper()`` → "expected a Utf8 argument, got LargeUtf8"
- ``join`` a LargeUtf8 key against a Utf8 key → "join key type mismatch"

The boundary now normalizes ``LargeUtf8 → Utf8`` (value-preserving; offsets only), so the
identical column succeeds. See docs/architecture/internals/bug_hunt_ledger.md.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def test_large_utf8_filter_against_string_literal(duck):
    """A LargeUtf8 column compares against a Utf8 literal (was: crash)."""
    t = pa.table({"s": pa.array(["a", "a", "b", "c"], pa.large_utf8()), "v": [1, 2, 3, 4]})
    out = bt.from_arrow(t).filter(bt.col("s") == "a").collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE s = 'a'"))


def test_large_utf8_string_functions(duck):
    """`str.contains` / `str.upper` on a LargeUtf8 column (was: crash)."""
    t = pa.table({"s": pa.array(["ab", "bc", "ca"], pa.large_utf8())})
    out = (
        bt.from_arrow(t)
        .select(
            u=bt.col("s").str.upper(),
            has_a=bt.col("s").str.contains("a"),
        )
        .collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql("SELECT upper(s) AS u, contains(s, 'a') AS has_a FROM t"),
    )


def test_large_utf8_join_against_utf8_key(duck):
    """Joining a LargeUtf8 key to a Utf8 key must match, not raise a type mismatch."""
    left = pa.table({"k": pa.array(["a", "b", "c"], pa.large_utf8()), "x": [1, 2, 3]})
    right = pa.table({"k": pa.array(["a", "c"], pa.utf8()), "y": [10, 30]})
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k").collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(out, duck.sql("SELECT l.k, l.x, r.y FROM l JOIN r USING (k)"))


def test_large_utf8_group_by_matches_utf8(duck):
    """Grouping on a LargeUtf8 key agrees with grouping on the decoded Utf8 key."""
    t = pa.table({"k": pa.array(["a", "b", "a", "c", "b"], pa.large_utf8()), "v": [1, 2, 3, 4, 5]})
    out = bt.from_arrow(t).group_by("k").agg(s=bt.col("v").sum()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT k, sum(v) AS s FROM t GROUP BY k"))


def test_large_utf8_round_trips_to_utf8_schema():
    """A LargeUtf8 column normalizes to a value-identical Utf8 column."""
    t = pa.table({"s": pa.array(["x", None, "zz"], pa.large_utf8())})
    out = bt.from_arrow(t).collect()
    assert out.schema.field("s").type == pa.utf8()
    assert out.to_pydict()["s"] == ["x", None, "zz"]


def test_large_utf8_column_compared_to_string_literal(duck):
    """`largeutf8_col = 'literal'` must compare, not raise a type mismatch. Every SQL string
    literal is Utf8, so a LargeUtf8 column filtered by equality once hit the comparison
    kernel's "identical types only" rule and errored `LargeUtf8 == Utf8` (ledger B257)."""
    t = pa.table({"r": [0, 1, 2, 3], "s": pa.array(["a", "b", "a", None], pa.large_utf8())})
    out = bt.from_arrow(t).filter(bt.col("s") == "a").select("r").collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT r FROM t WHERE s = 'a'"))
