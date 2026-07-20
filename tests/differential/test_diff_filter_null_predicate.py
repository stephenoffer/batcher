"""Differential tests for WHERE / filter three-valued-logic on NULL predicates.

A SQL filter keeps a row only when the predicate is TRUE; a NULL predicate drops
the row (three-valued logic). These guard a fixed zone-map pruning defect: over a
column whose min/max prove an inner predicate empty, ``NOT (that predicate)`` was
folded to *always-true* and the whole filter dropped — which wrongly KEPT the NULL-
predicate rows the filter must drop. See docs/internals/bug_hunt_ledger.md.

Root cause: `kyber/rules/zonemap_pruning.py::_predicate_status` negated its tri-state
with a plain `not inner`, turning an "always-empty" (`_FALSE`, which conflates FALSE
and NULL rows) into "always-true" (`_TRUE`). A null row negates to null (still
dropped), so `Not(_FALSE)` is NOT provably always-true. The negation is sound only in
the `_TRUE -> _FALSE` direction, which `_not` now enforces.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def test_not_between_out_of_range_keeps_null_dropped(duck):
    """`i NOT BETWEEN -2 AND 0` over a column whose only non-null value is out of range.

    ``i BETWEEN -2 AND 0`` is provably empty from min/max ([-3,-3] ∩ [-2,0] = ∅), but the
    NULL row makes ``NOT BETWEEN`` NULL, not TRUE — so the filter must drop it. The plan
    must not fold the filter away.
    """
    t = pa.table({"i": pa.array([None, -3], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").between(-2, 0)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -2 AND 0"))


def test_not_in_out_of_range_keeps_null_dropped(duck):
    """`i NOT IN (1, 2)` where the only non-null value (-3) is outside the list, plus a NULL."""
    t = pa.table({"i": pa.array([None, -3], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").is_in([1, 2])).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT IN (1, 2)"))


def test_not_between_with_nulls_and_passing_rows(duck):
    """Mixed nulls and out-of-range values: only the genuinely-passing rows survive."""
    t = pa.table(
        {
            "i": pa.array([None, None, -3, 5], pa.int64()),
            "row": pa.array([0, 1, 2, 3], pa.int64()),
        }
    )
    out = bt.from_arrow(t).filter(~bt.col("i").between(-2, 0)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -2 AND 0"))


def test_not_between_always_true_no_nulls_still_prunes(duck):
    """The sound direction is preserved: with no nulls, `NOT BETWEEN` a range covering all
    rows is genuinely always-false → empty result (matches DuckDB)."""
    t = pa.table({"i": pa.array([-3, -2], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").between(-100, 100)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -100 AND 100"))


def test_between_out_of_range_prune_still_correct(duck):
    """The always-empty pruning direction is unaffected: `i BETWEEN 50 AND 60` over [-3,-2]
    is empty, and the NULL-safe path must still prune it (no rows, no NULL kept)."""
    t = pa.table(
        {"i": pa.array([None, -3, -2], pa.int64()), "row": pa.array([0, 1, 2], pa.int64())}
    )
    out = bt.from_arrow(t).filter(bt.col("i").between(50, 60)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i BETWEEN 50 AND 60"))
