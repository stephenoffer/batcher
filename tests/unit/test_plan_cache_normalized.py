"""The NORMALIZE memo (`kyber.plan_cache.normalized`): a re-plan reuses the normalized plan.

What is pinned: a second optimization of the same plan is served the memoized NORMALIZE output
and ends at the plan a cold optimization produces; `plan_cache.clear` empties it; and a plan
whose key cannot see everything NORMALIZE reads (a streaming watermark) is never memoized.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber import plan_cache
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.plan_cache import normalized


@pytest.fixture(autouse=True)
def _fresh():
    plan_cache.clear()
    yield
    plan_cache.clear()


def _ds():
    t = pa.table({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    return bt.from_arrow(t).filter((col("k") > lit(1)) & (lit(1) == lit(1))).select("k", "v")


def _optimized(ds):
    return Optimizer(sources=ds._sources).optimize_full(ds._plan)[1]


def test_a_second_optimization_is_served_and_ends_at_the_same_plan():
    ds = _ds()
    cold = _optimized(ds)
    assert len(normalized._MEMO) == 1, "the first optimization stored its normalized plan"
    served = []
    real = normalized.lookup

    def spy(key):
        out = real(key)
        served.append(out is not None)
        return out

    normalized.lookup = spy
    try:
        warm = _optimized(ds)
    finally:
        normalized.lookup = real
    assert served == [True]
    assert warm.content_key() == cold.content_key()


def test_clear_empties_the_memo():
    _optimized(_ds())
    assert normalized._MEMO
    plan_cache.clear()
    assert not normalized._MEMO


def test_a_watermarked_plan_is_never_memoized():
    t = pa.table({"ts": pa.array([1, 2, 3], pa.timestamp("ms")), "v": [1, 2, 3]})
    ds = bt.from_arrow(t).with_watermark("ts", "1s").group_by("ts").agg(n=col("v").count())
    from batcher.plan.logical import Aggregate
    from batcher.plan.visitor import walk

    assert any(isinstance(n, Aggregate) and n.watermark for n in walk(ds._plan)), "control"
    _optimized(ds)
    assert not normalized._MEMO
