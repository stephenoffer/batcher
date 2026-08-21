"""Which operations carry an event-time watermark, and what happens where one is dropped.

`with_watermark` is what makes a downstream `groupby().agg()` *bounded*: the aggregate emits
and evicts each group as the watermark passes it (`core/streaming/folds.py` reads
``agg.watermark.time_col`` to build that fold). An aggregate with ``watermark=None`` is not an
error -- it is a valid, **unbounded** aggregate whose state is never evicted.

That is what makes the drop worth a test. Every single-input transform carries the watermark,
so `join` and `union` losing it is invisible at the call site and only shows up as a stream
whose memory grows without limit. The loss is not fixed here (carrying it through a
stream-to-stream join needs the *minimum* of the two watermarks, a streaming-semantics change
this project's gate cannot execute) -- it is *announced*, with the one-line repair named.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pyarrow as pa
import pytest

import batcher as bt

TS = dt.datetime(2024, 1, 1)


def _watermarked() -> bt.Dataset:
    table = pa.table(
        {
            "k": ["a", "b", "a"],
            "x": [1.0, 2.0, 3.0],
            "ts": pa.array([TS] * 3, type=pa.timestamp("us")),
        }
    )
    return bt.from_arrow(table).with_watermark("ts", "5 minutes")


def _right() -> bt.Dataset:
    return bt.from_arrow(pa.table({"k": ["a", "b"], "y": [10, 20]}))


@pytest.mark.parametrize(
    "transform",
    [
        lambda d: d.filter(bt.col("x") > 0),
        lambda d: d.select("k", "x", "ts"),
        lambda d: d.with_columns(z=bt.col("x") * 2),
        lambda d: d.limit(2),
        lambda d: d.sort("x"),
        lambda d: d.distinct(),
    ],
    ids=["filter", "select", "with_columns", "limit", "sort", "distinct"],
)
def test_single_input_transforms_carry_the_watermark(transform):
    """Including the pipeline breakers -- `sort` and `distinct` keep it, so it is not a
    breaker-vs-streaming distinction the caller could have predicted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert transform(_watermarked())._watermark is not None


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("join", lambda d, r: d.join(r, on="k")),
        ("union", lambda d, r: d.union(d)),
    ],
)
def test_join_and_union_announce_that_they_drop_the_watermark(name, build):
    """The drop stays (for now); the silence does not."""
    with pytest.warns(UserWarning, match="does not carry the event-time watermark"):
        result = build(_watermarked(), _right())
    assert result._watermark is None


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("join", lambda d, r: d.join(r, on="k")),
        ("union", lambda d, r: d.union(d)),
    ],
)
def test_nothing_is_announced_when_there_is_no_watermark_to_lose(name, build):
    """The announcement tracks the loss, not the operator -- otherwise every join warns."""
    plain = bt.from_arrow(pa.table({"k": ["a", "b"], "x": [1.0, 2.0]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(plain, _right())
    assert [str(w.message) for w in caught if "watermark" in str(w.message)] == []


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("join", lambda d, r: d.join(r, on="k")),
        ("union", lambda d, r: d.union(d)),
    ],
)
def test_the_repair_the_warning_names_actually_works(name, build):
    """A warning that recommends a fix is only useful while the fix works, so it is pinned.

    Re-applying `with_watermark` to the result restores the aggregate's bound exactly, which
    is why announcing the loss is a real remedy rather than a shrug.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joined = build(_watermarked(), _right())
        repaired = joined.with_watermark("ts", "5 minutes")
        lost = joined.groupby("k").agg(s=bt.col("x").sum())._plan
        kept = repaired.groupby("k").agg(s=bt.col("x").sum())._plan
    assert lost.watermark is None
    assert kept.watermark is not None
    assert kept.watermark.time_col == "ts"
