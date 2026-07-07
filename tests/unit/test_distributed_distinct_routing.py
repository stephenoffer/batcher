"""A distributed aggregate over a DISTINCT must distribute the DISTINCT first, never map-local.

The map/shuffle aggregate path runs the aggregate's input sub-plan as a per-partition,
breaker-free map prefix. A `Distinct` in that input has GLOBAL semantics: run it map-local
and each partition dedups independently, then the reducer sums the per-partition counts —
double-counting any value that spans two source partitions (an exact `COUNT(DISTINCT)` /
`distinct().count()` overcounts on the distributed path). The dispatch must therefore route
such a plan through the distinct shuffle, not the map-local aggregate. This guards the
routing decision without a live cluster; the numeric equivalence is covered distributed.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.dist.executors.plan_analysis import _has_breaker, _split_at
from batcher.kyber import optimize_logical
from batcher.plan.logical import Aggregate, Distinct

pytestmark = pytest.mark.unit

_T = pa.table({"x": [1, 2, 2, 3], "g": [1, 1, 2, 2]})


def _opt(ds):
    """The optimized logical plan the distributed executor dispatches (as the conductor does)."""
    return optimize_logical(ds._plan, sources=ds._sources)


def _agg_over_distinct(ds) -> bool:
    split = _split_at(_opt(ds), Aggregate)
    if split is None:
        return False
    _above, agg = split
    # The map-local aggregate path is guarded by `not _has_breaker(agg.input)`; a DISTINCT
    # input must trip that guard (so the dispatch falls through to the distinct handler).
    return isinstance(agg.input, Distinct) and _has_breaker(agg.input)


def test_lone_count_distinct_routes_through_distinct():
    # The `count_distinct → distinct + count` rewrite makes a lone n_unique an
    # Aggregate-over-Distinct; the map-local aggregate path must be bypassed.
    ds = bt.from_arrow(_T).agg(n=col("x").n_unique())
    assert _agg_over_distinct(ds)


def test_user_distinct_then_count_routes_through_distinct():
    ds = bt.from_arrow(_T).select("x").distinct().agg(c=count())
    assert _agg_over_distinct(ds)


def test_plain_groupby_still_uses_maplocal_path():
    # A normal aggregate over a breaker-free input keeps the fast map-local path.
    ds = bt.from_arrow(_T).group_by("g").agg(s=col("x").sum())
    split = _split_at(_opt(ds), Aggregate)
    assert split is not None
    _above, agg = split
    assert not _has_breaker(agg.input)  # breaker-free ⇒ map-local aggregate path is correct
