"""Invariant #7 at the streaming gate: distributed must never mean *different*.

`single-node == distributed` is a hard invariant, and the distributed streaming gate is where
it is enforced for streams: a plan the cluster cannot fold with single-node semantics is
**refused**, not run with different ones.

One shape slipped through. A watermarked windowed aggregation is an `Aggregate` over a
streamable input, so the old gate (`isinstance(plan, Aggregate) and is_streamable(plan.input)`)
waved it past — while `python/batcher/dist/` contains no watermark at all: no window eviction,
no late-row drop, no append output mode. The query silently degraded to an unbounded
complete-mode aggregate that re-emits the whole running result every epoch and grows state
without bound. Same query, `distributed=True` vs `False`, different answers, no error.

These tests pin the gate's decision directly. Reaching it end to end needs a partitionable
source, but the decision *is* the contract.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.api.io_namespace.writer import (
    _undistributable_stream_reason,
    distributed_stream_refusal,
)
from batcher.plan.streaming import Trigger

pytestmark = pytest.mark.unit


def _stream():
    """An unbounded source — the only kind a streaming write folds."""
    import datetime as dt

    import pyarrow as pa

    schema = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
    base = dt.datetime(2024, 1, 1)

    def batches():
        yield pa.RecordBatch.from_pylist([{"ts": base, "v": 1}], schema=schema)

    return bt.from_batches(batches, schema, bounded=False)


def test_plain_streaming_aggregate_is_distributable():
    """The shape the mergeable algebra genuinely covers: partial → combine → finalize."""
    plan = _stream().group_by("v").agg(total=col("v").sum())._plan
    assert _undistributable_stream_reason(plan) is None


def test_watermarked_aggregate_is_refused_not_silently_degraded():
    plan = (
        _stream()
        .with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
        ._plan
    )

    reason = _undistributable_stream_reason(plan)

    assert reason is not None, (
        "a watermarked windowed aggregate must be refused: the distributed runner has no "
        "watermark, so running it would return a different result than single-node"
    )
    assert "watermark" in reason
    # The message must point at the way out, not just say no.
    assert "distributed=False" in reason


def test_a_second_pipeline_breaker_is_still_refused():
    """The case the gate always caught — it must keep catching it."""
    plan = _stream().sort("v").group_by("v").agg(total=col("v").sum())._plan
    reason = _undistributable_stream_reason(plan)
    assert reason is not None
    assert "pipeline breaker" in reason


# --- the gate as the write path actually reaches it --------------------------------


def _watermarked():
    """The shape whose distributed answer differs from its single-node one."""
    return (
        _stream()
        .with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
        ._plan
    )


@pytest.mark.parametrize(
    "trigger",
    [None, Trigger.processing_time(0), Trigger.processing_time(5)],
    ids=["default", "asap", "paced"],
)
def test_the_refusal_holds_for_every_processing_time_trigger_including_the_default(trigger):
    """`trigger=None` is the *default*, not "no streaming" — and it skipped the whole gate.

    The refusals above were correct and were tested directly, so they stayed green while the
    call site consulted them only under `trigger is not None`. Since a `None` trigger means
    "processing time, as soon as possible", `ds.write(path, "delta", distributed=True)` on a
    watermarked aggregation ran the exact shape the refusal exists to prevent — silently
    degrading to an unbounded complete-mode aggregate on the cluster while one node produced
    windowed results. A gate reachable only through an optional argument is not a gate.
    """
    reason = distributed_stream_refusal(_watermarked(), trigger)
    assert reason is not None and "watermark" in reason


def test_a_continuous_trigger_still_names_the_micro_batch_boundary_first():
    """A stateful plan under a continuous trigger has two problems; report the trigger one.

    It is the one the caller can fix without changing the query, so naming the watermark
    instead would send them to restructure a plan that only needed a different trigger.
    """
    reason = distributed_stream_refusal(_watermarked(), Trigger.continuous(1))
    assert reason is not None and "continuous trigger" in reason


@pytest.mark.parametrize(
    "trigger",
    [None, Trigger.processing_time(0), Trigger.continuous(1)],
    ids=["default", "asap", "continuous"],
)
def test_a_stateless_pipeline_is_distributable_under_every_trigger(trigger):
    """The gate must not have grown teeth for the shape that was always fine."""
    plan = _stream().filter(col("v") > 0).select("v")._plan
    assert distributed_stream_refusal(plan, trigger) is None


def test_a_plain_aggregate_is_distributable_with_no_trigger():
    """Closing the hole must not close the door on the shape the algebra covers."""
    plan = _stream().group_by("v").agg(total=col("v").sum())._plan
    assert distributed_stream_refusal(plan, None) is None


# --- distinct() is the same mergeable fold, and used to be refused on a cluster -----


def test_a_streaming_distinct_is_distributable():
    """A whole-column `distinct()` IS the aggregate — a group-by over every column.

    The single-node processor has folded it that way since it was written, while the
    distributed gate tested `isinstance(plan, Aggregate)` and so refused the identical node
    on a cluster. That was a capability gap with no semantic cause: both paths would have
    been running the same operator over the same spec.
    """
    assert distributed_stream_refusal(_stream().distinct()._plan, None) is None


def test_a_streaming_distinct_folds_the_same_on_one_node_and_many():
    """Invariant #7 for the shape this change newly admits, proved rather than assumed.

    `combine(partial(p0), partial(p1))` must finalize to what one fold over both partitions
    produces, and both must be the batch answer. Runs the real `RunningAggregate` — the same
    state the distributed runner ships between workers — so this is the operator, not a model
    of it.
    """
    import pyarrow as pa

    from batcher.core.mergeable import RunningAggregate
    from batcher.plan.logical import streaming_fold_target

    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    partitions = [
        pa.RecordBatch.from_pylist(
            [{"k": 1, "v": 10}, {"k": 1, "v": 10}, {"k": 2, "v": 20}], schema
        ),
        pa.RecordBatch.from_pylist(
            [{"k": 1, "v": 10}, {"k": 2, "v": 21}, {"k": 3, "v": 30}], schema
        ),
    ]
    ds = bt.from_batches(lambda: iter(partitions), schema, bounded=False)
    agg = streaming_fold_target(ds.distinct()._plan)
    assert agg is not None and [p.alias for p in agg.group_keys] == ["k", "v"]

    single = RunningAggregate(agg)
    single.push(partitions)

    combined = RunningAggregate(agg)
    for part in partitions:  # each "worker" partials its own share
        combined.absorb(RunningAggregate(agg).partial([part]))

    def rows(batch):
        return sorted(map(str, pa.Table.from_batches([batch]).to_pylist()))

    oracle = sorted(
        map(str, bt.from_arrow(pa.Table.from_batches(partitions)).distinct().collect().to_pylist())
    )
    assert rows(single.finalize()) == oracle
    assert rows(combined.finalize()) == oracle


def test_a_keyed_distinct_is_still_refused_and_says_why():
    """`DISTINCT ON` has no group-by form, so it must not ride in on the whole-column case."""
    from batcher.plan.logical import Distinct

    # Built directly: `Dataset.distinct(subset=...)` refuses an unbounded source up front,
    # so this node is the only way to ask the gate the question.
    plan = Distinct(_stream()._plan, keys=("v",))
    reason = distributed_stream_refusal(plan, None)
    assert reason is not None and "drop_duplicates_within_watermark" in reason
    assert "pipeline breaker" not in reason, (
        "a keyed dedup has no breaker in it; sending the reader to look for one is the "
        "same misdirection transform_with_state was given its own message to avoid"
    )
