"""Regression tests vs DuckDB for two NaN-handling bugs in the NORMALIZE rewrites.

Both were found by fuzzing the optimizer against its own un-optimized plan (and confirmed
against DuckDB). ``bc-expr`` orders NaN as the *maximum* float value — ``NaN > x`` and
``NaN == NaN`` are TRUE, ``NaN < x`` is FALSE — the opposite of Python, and ``col = NaN``
matches a NaN row. Two rewrites reasoned about NaN with Python semantics and changed a
result:

* ``normalize.ranges.or_to_in_and_range`` added ``c >= min(vs) AND c <= max(vs)`` to a
  ``c = v1 OR c = v2 OR …`` disjunction. With a NaN among the values, that range dropped the
  NaN rows the ``c = NaN`` disjunct keeps (``NaN >= lo`` is FALSE), and Python's
  ``min``/``max`` over a NaN-bearing list is order-dependent (a leading NaN yields
  ``c >= NaN``, which rejects every row).

* ``normalize.fold.constant_folding`` folded a constant comparison with Python's operators,
  so ``NaN > -0.0`` folded to FALSE where the engine computes TRUE. Reachable in a plain
  query once ``constant_propagation`` substitutes a ``col = NaN`` equality into a sibling
  comparison.

Each query runs through the full optimizer (so the rules fire) and must match DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, lit

pytestmark = pytest.mark.differential

_NAN = float("nan")


def _t(duck):
    t = pa.table(
        {"f": pa.array([_NAN, 1.0, -0.0, 0.0, 2.0, _NAN, -1.0], pa.float64())},
        schema=pa.schema([("f", pa.float64())]),
    )
    duck.register("nant", t)
    return bt.from_arrow(t)


def test_or_to_in_range_keeps_nan_disjunct(duck):
    """`f = 0.0 OR f = NaN OR f = 2.0` must keep NaN rows (the added range must not drop them)."""
    ds = _t(duck)
    out = ds.filter((col("f") == lit(0.0)) | (col("f") == lit(_NAN)) | (col("f") == lit(2.0)))
    out = out.select("f").collect()
    assert_same(
        out,
        duck.sql("SELECT f FROM nant WHERE f = 0.0 OR f = CAST('nan' AS DOUBLE) OR f = 2.0"),
    )


def test_or_to_in_range_leading_nan(duck):
    """A NaN first in the OR list must not collapse the whole filter (Python min([nan,…])=nan)."""
    ds = _t(duck)
    out = ds.filter((col("f") == lit(_NAN)) | (col("f") == lit(0.0)) | (col("f") == lit(2.0)))
    out = out.select("f").collect()
    assert_same(
        out,
        duck.sql("SELECT f FROM nant WHERE f = CAST('nan' AS DOUBLE) OR f = 0.0 OR f = 2.0"),
    )


def test_constant_fold_nan_comparison_via_propagation(duck):
    """`f > -0.0 AND f = NaN` must keep the NaN row (fold of `NaN > -0.0` must not be FALSE)."""
    ds = _t(duck)
    out = ds.filter((col("f") > lit(-0.0)) & (col("f") == lit(_NAN))).select("f").collect()
    assert_same(
        out,
        duck.sql("SELECT f FROM nant WHERE f > -0.0 AND f = CAST('nan' AS DOUBLE)"),
    )
