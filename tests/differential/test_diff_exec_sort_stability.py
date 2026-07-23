"""Single-key sort stability across execution paths (invariant #7: seq == par == spill).

A full single-key ORDER BY takes a stable specialized path for the common key types
(string, integer/temporal, non-NaN float). Every *other* single key — boolean, decimal,
a NaN-bearing float — had no such path and fell through to arrow's UNSTABLE
`sort_to_indices`, whose tie order is arbitrary and input-size-dependent. That made the
in-memory sort (one comparison over the whole relation) and the spilling sort (one
comparison per ~16 k-row run, then a merge) resolve rows equal on the key into *different*
row orders — a `collect()` vs `collect(spill=True)` divergence on the payload columns.

SQL leaves the tie order of an under-determined ORDER BY unspecified, so DuckDB is not the
oracle here; the binding contract is Batcher's own: the spilled path must reproduce the
in-memory path row-for-row. These assertions are ordered — an unordered compare is blind to
a tie-order bug (CLAUDE.md calls this out).
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

# Large enough to morselize into several runs when spilling (>2 x the 16,384-row morsel),
# with heavy ties on the sort key so the tie-break is exercised, and a distinct payload so a
# tie-order disagreement surfaces as a row-order mismatch.
_N = 40_000


@pytest.mark.parametrize("descending", [False, True])
def test_boolean_sort_spill_equals_in_memory(descending):
    """A boolean single-key sort is stable — the spilled path equals the in-memory path."""
    table = pa.table(
        {
            "b": pa.array([i % 3 == 0 for i in range(_N)], pa.bool_()),
            "p": pa.array(list(range(_N)), pa.int64()),
        }
    )
    plan = bt.from_arrow(table).sort(bt.col("b"), descending=descending)
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


@pytest.mark.parametrize("descending", [False, True])
def test_decimal_sort_spill_equals_in_memory(descending):
    """A decimal single-key sort (no radix/string fast path) is stable across the spill split."""
    vals = [decimal.Decimal(i % 5) for i in range(_N)]
    table = pa.table(
        {
            "d": pa.array(vals, pa.decimal128(10, 0)),
            "p": pa.array(list(range(_N)), pa.int64()),
        }
    )
    plan = bt.from_arrow(table).sort(bt.col("d"), descending=descending)
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


@pytest.mark.parametrize("nulls_first", [False, True])
def test_nan_float_sort_spill_equals_in_memory(nulls_first):
    """A NaN-bearing float key (the radix declines it) is stable across the spill split."""
    fvals = [float("nan") if i % 7 == 0 else float(i % 4) for i in range(_N)]
    table = pa.table(
        {
            "f": pa.array(fvals, pa.float64()),
            "p": pa.array(list(range(_N)), pa.int64()),
        }
    )
    plan = bt.from_arrow(table).sort(bt.col("f"), nulls_first=nulls_first)
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)
