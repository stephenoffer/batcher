"""A predicate that answers *whether* a path applies must never raise when the answer is no.

`supports_spilling_sort`, `supports_spilling_window`, `supports_spilling_join` and
`supports_ordered_bucket_offsets` are each asked, on every `collect(spill=True)` and every
`iter_batches()`, whether the out-of-core path covers this plan. A `False` costs memory: the
caller runs the ordinary in-memory kernel and returns the right answer. A raise costs the
query — and it is a raise the caller has no way to anticipate, because a plan that answered
`collect()` perfectly well is what reaches these.

That has happened three times, always the same way and always found one predicate at a time
after it shipped:

* `supports_spilling_window` did not check that its input named a single source, so a window
  above a join reached `_relabel_single_source`'s assertion. `supports_spilling_join`'s own
  docstring asserted "Sort and Window already gate this way"; Sort did, Window did not.
* `supports_ordered_bucket_offsets` did not check the order key's *type*, so `rank()` over a
  Boolean column passed the shape test and then died inside the range partitioner with a bare
  ``RuntimeError: range-partition key must be a numeric column`` — a query that worked in
  batch failing in streaming.
* the same predicate did not check for a multi-source input either, and was fixed for it
  separately, after the type check.

So this asks all four, on a matrix of plan shapes built to land on each of those edges plus
the ones nobody has hit yet. It asserts only that a `bool` comes back. Which shapes are
*supported* is the business of the differential suites, which compare the spilled and
streamed results against the in-memory ones; what is pinned here is the weaker property that
every one of these predicates is total.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_LEFT = pa.table(
    {
        "k": pa.array([1, 2, 3, 1], pa.int64()),
        "t": pa.array([10, 20, 30, 40], pa.int64()),
        "f": pa.array([1.5, None, 3.0, -0.0], pa.float64()),
        "s": pa.array(["a", "b", None, "a"]),
        "flag": pa.array([True, False, None, True], pa.bool_()),
    }
)
_RIGHT = pa.table({"k": pa.array([1, 2], pa.int64()), "w": pa.array(["x", "y"])})


def _shapes():
    """Plan shapes x the edges the three historical raises came from.

    Each entry is `(name, builder)` where the builder takes the two bound datasets and
    returns a `LogicalPlan`. The builders deliberately include shapes no spill path covers
    (a window above a join, a Boolean order key, a multi-source input), because those are
    precisely the ones that must return `False` rather than raise.
    """
    left, right = bt.from_arrow(_LEFT), bt.from_arrow(_RIGHT)
    joined = left.join(right, on="k")
    return {
        # --- sorts ---------------------------------------------------------------------
        "sort_plain": left.sort("k")._plan,
        "sort_string_key": left.sort("s")._plan,
        "sort_bool_key": left.sort("flag")._plan,
        "sort_float_key": left.sort("f")._plan,
        "sort_computed_key": left.sort(bt.col("k") + bt.col("t"))._plan,
        "sort_multi_key": left.sort(["k", "t"])._plan,
        "sort_above_join": joined.sort("k")._plan,
        "sort_above_aggregate": left.group_by("k").agg(n=bt.col("t").count()).sort("k")._plan,
        "sort_with_limit": left.sort("k").limit(2)._plan,
        # --- windows -------------------------------------------------------------------
        "window_partitioned": left.window(
            partition_by=["k"], order_by=["t"], functions={"r": "row_number"}
        )._plan,
        "window_computed_key": left.window(
            partition_by=[bt.col("k") % 2], order_by=["t"], functions={"r": "row_number"}
        )._plan,
        "window_global_ordered": left.window(order_by=["t"], functions={"r": "row_number"})._plan,
        "window_global_bool_key": left.window(order_by=["flag"], functions={"r": "rank"})._plan,
        "window_global_string_key": left.window(order_by=["s"], functions={"r": "rank"})._plan,
        "window_global_fold": left.window(
            order_by=["t"], functions={"w": ("bit_or", bt.col("k"))}
        )._plan,
        "window_global_lag": left.window(
            order_by=["t"], functions={"w": ("lag", bt.col("k"))}
        )._plan,
        "window_above_join": joined.window(order_by=["t"], functions={"r": "row_number"})._plan,
        "window_partitioned_above_join": joined.window(
            partition_by=["k"], order_by=["t"], functions={"r": "row_number"}
        )._plan,
        # --- joins ---------------------------------------------------------------------
        "join_inner": joined._plan,
        "join_left": left.join(right, on="k", how="left")._plan,
        "join_over_aggregate": left.group_by("k")
        .agg(n=bt.col("t").count())
        .join(right, on="k")
        ._plan,
    }


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_every_spill_predicate_answers_rather_than_raising(name):
    from batcher.dist.global_window import supports_ordered_bucket_offsets
    from batcher.dist.spill_breakers import (
        supports_spilling_join,
        supports_spilling_sort,
        supports_spilling_window,
    )
    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Join, Sort, Window

    plan = _shapes()[name]
    sources = [InMemorySource(_LEFT.to_batches()), InMemorySource(_RIGHT.to_batches())]

    # Each predicate is typed for one node kind, so ask only the ones that apply — plus, for
    # the sort, the `sources=None` spelling the streaming dispatcher also uses.
    if isinstance(plan, Sort):
        assert isinstance(supports_spilling_sort(plan, sources), bool)
        assert isinstance(supports_spilling_sort(plan, None), bool)
    if isinstance(plan, Window):
        assert isinstance(supports_spilling_window(plan), bool)
        assert isinstance(supports_ordered_bucket_offsets(plan), bool)
    if isinstance(plan, Join):
        assert isinstance(supports_spilling_join(plan), bool)
