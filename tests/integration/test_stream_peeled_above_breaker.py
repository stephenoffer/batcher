"""Row-wise operators stacked *above* a breaker stream instead of materializing.

`api/terminal/stream/dispatch.py` dispatches on the exact top-level node, so
`group_by().agg().select()`, SQL `HAVING` (`Filter(Aggregate)`), and `sort().select()`
matched no streaming branch and fell through to `_collect` — materializing the whole
result even though the breaker underneath already streams and `Project`/`Filter` are
per-batch valid. The dispatcher now peels those ops, streams the breaker, and re-applies
them per emitted batch.

`Limit` is deliberately NOT peeled: applied per batch it would keep n rows from EVERY
batch. `test_limit_above_breaker_is_not_peeled` pins that.

The peelable set is the neutral `is_partition_independent` predicate rather than a
hand-written tuple. The tuple listed only `Project`/`Filter` while the predicate already
admitted the row-multiplying reshapers, so `group_by().agg().explode()` and
`sort().unpivot()` materialized for no reason —
`test_row_multiplying_reshapers_are_peeled` pins that they no longer do.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt


def _rows(ds) -> list[dict]:
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches).to_pylist() if batches else []


@pytest.fixture
def big():
    random.seed(0)
    n = 50_000
    return bt.from_pydict(
        {
            "g": [random.choice("abcde") for _ in range(n)],
            "v": [random.randint(0, 1000) for _ in range(n)],
        }
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "shape",
    [
        lambda d: d.group_by("g").agg(s=bt.col("v").sum()).select("s"),
        lambda d: d.group_by("g").agg(s=bt.col("v").sum()).filter(bt.col("s") > 0),
        lambda d: d.group_by("g").agg(s=bt.col("v").sum()).with_columns(x=bt.col("s") * 2),
        lambda d: d.select("g").distinct().select("g"),
        lambda d: d.sort("v").select("v"),
        lambda d: d.sort("v", descending=True).filter(bt.col("v") > 500),
    ],
    ids=["agg-select", "having", "agg-derived", "distinct-select", "sort-select", "sort-filter"],
)
def test_streamed_result_equals_collected(big, shape):
    ds = shape(big)
    collected = ds.collect().to_pylist()
    streamed = _rows(ds)
    # Order-DEPENDENT comparison on purpose: an order-independent one cannot see a sort
    # bug, and two of these shapes are sorts (CLAUDE.md's explicit warning).
    assert streamed == collected


@pytest.mark.integration
def test_sort_order_survives_per_batch_reapply(big):
    # The risky shape: the breaker below establishes a global order, and the peeled
    # `Project` is re-applied per batch. Descending, over many batches.
    ds = big.sort("v", descending=True).select("v")
    values = [r["v"] for r in _rows(ds)]
    assert len(values) == 50_000
    assert values == sorted(values, reverse=True)


@pytest.mark.integration
def test_these_shapes_no_longer_materialize(big, monkeypatch):
    """The point of the change: `_collect` is not reached for a peelable plan."""
    import batcher.api.terminal.core as tc

    calls: list[int] = []
    original = tc._collect
    monkeypatch.setattr(tc, "_collect", lambda *a, **k: (calls.append(1), original(*a, **k))[1])

    ds = big.group_by("g").agg(s=bt.col("v").sum()).select("s")
    assert len(_rows(ds)) == 5
    assert calls == []


@pytest.mark.integration
def test_limit_above_breaker_is_not_peeled(big):
    """`LIMIT` is not row-wise: peeling it would keep n rows from every batch."""
    ds = big.sort("v", descending=True).limit(10)
    streamed = _rows(ds)
    assert len(streamed) == 10
    assert streamed == ds.collect().to_pylist()


@pytest.mark.integration
@pytest.mark.parametrize(
    "shape",
    [
        lambda d: d.group_by("g").agg(vs=bt.col("v").array_agg()).explode("vs"),
        lambda d: (
            d.group_by("g").agg(vs=bt.col("v").array_agg()).explode("vs").filter(bt.col("vs") > 500)
        ),
        lambda d: d.sort("v").unpivot(index=["g"], on=["v"]),
        lambda d: d.group_by("g").agg(s=bt.col("v").sum()).unpivot(index=["g"], on=["s"]),
    ],
    ids=["agg-explode", "agg-explode-filter", "sort-unpivot", "agg-unpivot"],
)
def test_row_multiplying_reshapers_are_peeled(big, shape, monkeypatch):
    """`Unnest`/`Unpivot` multiply rows but hold no state, so a batch's output does not
    depend on how the input was split — they are peelable, and used not to be."""
    import batcher.api.terminal.core as tc

    calls: list[int] = []
    original = tc._collect
    monkeypatch.setattr(tc, "_collect", lambda *a, **k: (calls.append(1), original(*a, **k))[1])

    ds = shape(big)
    assert _rows(ds) == ds.collect().to_pylist()
    assert calls == []


@pytest.mark.integration
def test_peeling_a_reshaper_preserves_the_breakers_order(big):
    """The risky shape: the breaker below establishes a global order and the peeled
    reshaper fans each batch out. Row order must survive the fan-out."""
    ds = big.sort("v", descending=True).unpivot(index=["g"], on=["v"])
    values = [r["value"] for r in _rows(ds)]
    assert len(values) == 50_000
    assert values == sorted(values, reverse=True)


@pytest.mark.integration
def test_empty_result_keeps_its_schema(big):
    ds = big.group_by("g").agg(s=bt.col("v").sum()).filter(bt.col("s") > 10**12)
    assert _rows(ds) == []
    assert ds.collect().num_rows == 0
