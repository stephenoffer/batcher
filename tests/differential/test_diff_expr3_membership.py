"""Differential tests for `is_in`/`InList` membership and `sequence` generation.

Two regressions are pinned here:

* A float column filtered by >= 5 integer-equality disjuncts is folded (Kyber
  `fold_in_list`) into an `InList` over a `Float64` column. Integer literals are
  foldable and the fold does not inspect the column's dtype, so `WHERE x IN (1, 3, 5,
  7, 9)` on a `DOUBLE` column reaches `eval_in_list` with a `Float64` array — which
  previously errored ("in_list unsupported for Float64"), turning a working filter into
  a crash. The fix compares by total-order (bit) equality, matching the `col = lit`
  path the fold replaces.
* `sequence(start, stop, step)` over an unbounded range must error rather than attempt
  a multi-gigabyte allocation / overflow its 32-bit list offsets.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, sequence


def _int_in_predicate(values: list[int]):
    """`col("x") == v0 | ... == vN` with *integer* literals (the foldable shape)."""
    pred = col("x") == values[0]
    for v in values[1:]:
        pred = pred | (col("x") == v)
    return pred


def test_float_column_int_in_list_fold_matches_duckdb(duck):
    """A float column filtered by >= 5 int-equality disjuncts folds to InList<Float64>."""
    from conftest import assert_same

    tbl = pa.table({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, None]})
    duck.register("t", tbl)
    # Five int-literal disjuncts pass the fold threshold -> InList over Float64.
    out = bt.from_arrow(tbl).filter(_int_in_predicate([1, 3, 5, 7, 9])).collect()
    assert_same(out, duck.sql("SELECT x FROM t WHERE x IN (1, 3, 5, 7, 9)"))


def test_float_column_int_in_list_fold_large_set(duck):
    """Past LINEAR_SCAN_MAX the Float64 set takes the hashed branch; still matches."""
    from conftest import assert_same

    tbl = pa.table({"x": [float(i) for i in range(20)] + [None]})
    duck.register("t", tbl)
    members = list(range(0, 20, 2))  # 10 members > LINEAR_SCAN_MAX
    out = bt.from_arrow(tbl).filter(_int_in_predicate(members)).collect()
    in_sql = ", ".join(str(m) for m in members)
    assert_same(out, duck.sql(f"SELECT x FROM t WHERE x IN ({in_sql})"))


@pytest.mark.parametrize(
    "start,stop,step,expected",
    [
        (1, 5, 1, [1, 2, 3, 4, 5]),
        (5, 1, -1, [5, 4, 3, 2, 1]),
        (5, 1, 1, []),  # step points the wrong way -> empty
        (1, 5, -1, []),
        (1, 10, 3, [1, 4, 7, 10]),
        (3, 3, 1, [3]),  # single element
        (10, 1, -3, [10, 7, 4, 1]),
    ],
)
def test_sequence_inclusive_series(start, stop, step, expected):
    """`sequence` builds an inclusive integer series (Spark/DuckDB generate_series)."""
    ds = bt.from_pydict({"z": [0]})
    got = ds.select(r=sequence(bt.lit(start), bt.lit(stop), bt.lit(step))).to_pydict()
    assert got["r"] == [expected]


def test_sequence_over_large_range_errors_not_oom():
    """~10^10 elements must raise, not exhaust memory / overflow the 32-bit offsets."""
    ds = bt.from_pydict({"z": [0]})
    with pytest.raises(Exception):
        ds.select(r=sequence(bt.lit(1), bt.lit(10_000_000_000), bt.lit(1))).collect()


def test_sequence_zero_step_errors():
    ds = bt.from_pydict({"z": [0]})
    with pytest.raises(Exception):
        ds.select(r=sequence(bt.lit(1), bt.lit(5), bt.lit(0))).collect()
