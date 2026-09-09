"""What a device has to read for a join is both its inputs, not the join's cardinality.

`decide_gpu_backend` sizes the fan-out from `_estimate`, which used to descend the run of
*reducing* nodes on top of a plan and estimate the first node below them. For a chain over one
scan that is right — a filter or a projection outputs about what it reads. For a **join** it is
two errors at once: a projection sits above the outermost join in essentially every real
analytical plan, so the walk stopped there; and the thing it then estimated was the join's
output cardinality, which is not what a join processes and is what the estimator is least
reliable about.

Measured on TPC-H sf10, that reported **one row** for q14 and q17 and ten for q3. All three were
therefore routed to a *single* device — q14 took 11.4 s on one board against the CPU engine's
1.9 s, with five devices idle.

The chain case must be untouched by the fix, which is half of what these tests pin.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.gpu.policy import _first_branch, _processed

pytestmark = pytest.mark.unit

ROW_BYTES = 8


def _estimator(ds):
    return CardinalityEstimator(sources=ds._sources, learned=None)


def _left():
    return bt.from_pydict({"k": list(range(64)), "a": list(range(64))})


def _right():
    return bt.from_pydict({"k": list(range(32)), "b": list(range(32))})


# --- finding the branch ------------------------------------------------------


def test_a_chain_has_no_branch():
    ds = _left().filter(bt.col("a") > 1).select("k")
    assert _first_branch(ds._plan) is None


def test_a_projection_above_a_join_does_not_hide_it():
    """The shape that made this matter: the optimizer puts one there in every real plan."""
    ds = _left().join(_right(), on="k").select("a")
    assert _first_branch(ds._plan) is not None


def test_an_aggregate_above_a_join_does_not_hide_it():
    ds = _left().join(_right(), on="k").group_by("k").agg(s=bt.col("a").sum())
    assert _first_branch(ds._plan) is not None


# --- what it measures --------------------------------------------------------


def test_a_join_processes_both_of_its_inputs():
    ds = _left().join(_right(), on="k")
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert rows == 64 + 32


def test_a_join_under_an_aggregate_still_processes_both_inputs():
    """The aggregate's output is a handful of rows; the device still reads 96."""
    ds = _left().join(_right(), on="k").group_by("k").agg(s=bt.col("a").sum())
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert rows == 64 + 32


def test_a_three_way_join_sums_every_leaf():
    third = bt.from_pydict({"k": list(range(16)), "c": list(range(16))})
    ds = _left().join(_right(), on="k").join(third, on="k")
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert rows == 64 + 32 + 16


def test_a_union_sums_its_inputs():
    ds = bt.concat([_left().select("k"), _right().select("k")])
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert rows == 64 + 32


def test_bytes_track_rows():
    ds = _left().join(_right(), on="k")
    rows, nbytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert nbytes > 0
    assert nbytes >= rows  # at least a byte a row, whatever the widths worked out to


# --- the chain case is unchanged ---------------------------------------------


def test_a_reducing_chain_is_still_measured_below_its_reducers():
    """`group_by().agg().sort().limit(10)` outputs ten rows over a scan of sixty-four."""
    ds = _left().group_by("k").agg(s=bt.col("a").sum()).sort("k").limit(10)
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert rows == 64


def test_a_map_chain_is_measured_at_its_own_output():
    """A filter's output is what the operators above it hold, and that rule is not changed here."""
    ds = _left().filter(bt.col("a") > 1)
    rows, _bytes = _processed(ds._plan, _estimator(ds), ROW_BYTES)
    assert 0 < rows <= 64
