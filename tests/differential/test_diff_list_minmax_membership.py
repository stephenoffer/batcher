"""Differential tests: list min/max type-generality + precision, and float membership.

Each test pins a bug-hunt defect. Before the fix the engine diverged from DuckDB;
after it they agree. See docs/internals/bug_hunt_ledger.md.

- ``list_min``/``list_max`` cast the child to Float64, which nulled non-numeric
  elements (strings/bools/dates) and lost integer precision above 2^53 — returning a
  value not even in the list. They must preserve the element type and be exact.
- ``list_contains``/``list_position`` compared floats raw, so ``-0.0`` and ``0.0``
  (one value under DuckDB and the engine's GROUP BY/join keys) failed to match.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same

pytestmark = pytest.mark.differential


def test_list_min_max_over_strings(duck):
    """list_min/list_max on a string list return the lexical extreme, not null."""
    t = pa.table(
        {
            "a": pa.array(
                [["banana", "apple", "cherry"], [], None, ["z"]],
                type=pa.list_(pa.string()),
            )
        }
    )
    out = bt.from_arrow(t).select(mn=bt.col("a").list.min(), mx=bt.col("a").list.max()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT list_min(a) AS mn, list_max(a) AS mx FROM t"))


def test_list_min_max_integer_precision(duck):
    """list_min/list_max stay exact for i64 elements above 2^53 (no f64 rounding)."""
    t = pa.table(
        {
            "a": pa.array(
                [
                    [(1 << 53) + 1, (1 << 53) + 2, 9223372036854775807],
                    [9223372036854775807, 1],
                    [100, 200],
                ],
                type=pa.list_(pa.int64()),
            )
        }
    )
    out = bt.from_arrow(t).select(mn=bt.col("a").list.min(), mx=bt.col("a").list.max()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT list_min(a) AS mn, list_max(a) AS mx FROM t"))


def test_list_contains_position_fold_negative_zero(duck):
    """list_contains/list_position treat -0.0 and 0.0 as one value, matching DuckDB."""
    t = pa.table({"a": pa.array([[0.0], [-0.0], [1.0, -0.0]], type=pa.list_(pa.float64()))})
    out = (
        bt.from_arrow(t)
        .select(
            c=bt.col("a").list.contains(0.0),
            p=bt.col("a").list.position(-0.0),
        )
        .collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql("SELECT list_contains(a, 0.0) AS c, list_position(a, -0.0) AS p FROM t"),
    )
