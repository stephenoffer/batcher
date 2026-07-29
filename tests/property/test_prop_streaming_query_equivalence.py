"""Property: a *streaming query* writes what the batch collect returns.

`test_prop_path_equivalence` covers `{collect, spill, iter_batches, distributed}`. It does
not cover the fifth path — the micro-batch engine behind `ds.write(..., trigger=...)`,
which has its own loop, its own checkpoint, and its own sink protocol. That path is where
the bugs live: an epoch whose output was several batches lost everything after the first;
the end-of-stream flush collided with the next run's first epoch; a drained source's final
position was never recorded, so a restart replayed its last window as duplicate output.
Every one of those was a *silent* wrong answer that a per-shape unit test missed and a
random micro-batch chunking would have caught.

So this asserts, for a random table, a random pipeline, and a random source chunking:

    collect(pipeline)  ==  everything the streaming query wrote to its sink

as multisets, and does it again over a *restart* against the same checkpoint, where the
second run must add exactly the rows the second half of the stream carries and no more.
"""

from __future__ import annotations

import math
import uuid

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]

_SCHEMA = pa.schema([("k", pa.int64()), ("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
_floats = st.one_of(st.none(), st.sampled_from([0.0, -0.0, 1.5, -1.5, float("nan")]))
_ints = st.one_of(st.none(), st.sampled_from([-3, -1, 0, 1, 2, 7]))
_strs = st.one_of(st.none(), st.sampled_from(["a", "b", "", "c"]))


def _coerce(v: object) -> object:
    if isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        if v != v:
            return "<nan>"
        if not math.isfinite(v):
            return float(v)
        r = round(v, 9)
        return int(r) if r == int(r) else r
    return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = sorted(table.column_names)
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


@st.composite
def _rows(draw: st.DrawFn) -> pa.Table:
    n = draw(st.integers(min_value=0, max_value=30))
    return pa.table(
        {
            "k": pa.array(draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)), pa.int64()),
            "i": pa.array(draw(st.lists(_ints, min_size=n, max_size=n)), pa.int64()),
            "f": pa.array(draw(st.lists(_floats, min_size=n, max_size=n)), pa.float64()),
            "s": pa.array(draw(st.lists(_strs, min_size=n, max_size=n)), pa.string()),
        },
        schema=_SCHEMA,
    )


# Append-mode shapes: stateless pipelines, which is what a streaming write supports for
# `append`. Each is breaker-free so the micro-batch processor runs it per batch.
_APPEND = {
    "identity": lambda ds: ds,
    "filter": lambda ds: ds.filter(col("i") > bt.lit(0)),
    "project": lambda ds: ds.select("k", "i"),
    "derive": lambda ds: ds.with_columns(d=col("i") * bt.lit(2)),
    "filter_then_project": lambda ds: ds.filter(col("k") > bt.lit(0)).select("k", "f"),
}

# Complete-mode shapes: a running aggregate, whose sink holds the *whole* running result.
_COMPLETE = {
    "grouped_sum": lambda ds: ds.group_by("k").agg(s=col("i").sum(), n=bt.count()),
    "grouped_minmax": lambda ds: ds.group_by("k").agg(lo=col("i").min(), hi=col("i").max()),
    "global_count": lambda ds: ds.agg(n=bt.count()),
}

_HC = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
_PROP = settings(max_examples=25, deadline=None, suppress_health_check=_HC)


def _chunked(table: pa.Table, size: int):
    """The table as a `from_batches` source cut into `size`-row batches."""
    batches = table.to_batches(max_chunksize=size) if table.num_rows else []

    def gen():
        yield from batches

    return bt.from_batches(gen, table.schema, bounded=True)


def _drain_to_memory(ds, name: str, output_mode: str) -> pa.Table:
    query = ds.write.memory(name, trigger=bt.Trigger.available_now(), output_mode=output_mode)
    assert query.await_termination(60) is True
    assert query.exception() is None, query.exception()
    return bt.read_memory(name).collect()


@_PROP
@given(_rows(), st.sampled_from(sorted(_APPEND)), st.sampled_from([1, 3, 8]))
def test_an_append_stream_writes_what_collect_returns(table, shape, chunk):
    """The micro-batch loop, its sink protocol and its chunking must not change the answer."""
    name = f"prop-append-{uuid.uuid4().hex}"
    build = _APPEND[shape]
    streamed = _drain_to_memory(build(_chunked(table, chunk)), name, "append")
    expected = build(bt.from_arrow(table)).collect()
    assert _rowset(streamed) == _rowset(expected)


@_PROP
@given(_rows(), st.sampled_from(sorted(_COMPLETE)), st.sampled_from([1, 3, 8]))
def test_a_complete_stream_ends_at_the_batch_aggregate(table, shape, chunk):
    """`complete` emits the running result each micro-batch, so the sink's final contents
    are the aggregate over everything consumed — the mergeable fold's whole claim."""
    name = f"prop-complete-{uuid.uuid4().hex}"
    build = _COMPLETE[shape]
    streamed = _drain_to_memory(build(_chunked(table, chunk)), name, "complete")
    expected = build(bt.from_arrow(table)).collect()
    assert _rowset(streamed) == _rowset(expected)


@pytest.fixture(scope="module")
def ckpt_root(tmp_path_factory):
    """A module-scoped root: `@given` runs many examples, each needing its own directory."""
    return tmp_path_factory.mktemp("stream-ckpt")


@_PROP
@given(st.integers(min_value=1, max_value=12), st.integers(min_value=1, max_value=12))
def test_a_checkpointed_restart_over_a_replayable_source_neither_replays_nor_skips(
    ckpt_root, first_rows, extra_rows
):
    """Two drains against one checkpoint: the second adds the rows the source grew by.

    Uses the `rate` source, which reports a position and honours `seek` — the property only
    means anything over a source that *can* be resumed. A restart that replays its last
    window shows up here as duplicate values; one that skips shows up as a gap.
    """
    name = f"prop-restart-{uuid.uuid4().hex}"
    location = str(ckpt_root / name)
    total = first_rows + extra_rows

    first = bt.read.rate(rows_per_second=2, num_rows=first_rows, pace=False).write.memory(
        name, trigger=bt.Trigger.available_now(), output_mode="append", checkpoint=location
    )
    assert first.await_termination(60) is True
    assert first.exception() is None, first.exception()
    seen = bt.read_memory(name).collect().column("value").to_pylist()
    assert sorted(seen) == list(range(first_rows))

    second = bt.read.rate(rows_per_second=2, num_rows=total, pace=False).write.memory(
        name, trigger=bt.Trigger.available_now(), output_mode="append", checkpoint=location
    )
    assert second.await_termination(60) is True
    assert second.exception() is None, second.exception()
    # `memory` clears on open, so the second sink holds exactly what the second run
    # consumed: the values beyond the checkpoint, each once.
    resumed = bt.read_memory(name).collect().column("value").to_pylist()
    assert sorted(resumed) == list(range(first_rows, total))


@_PROP
@given(_rows(), st.sampled_from([1, 4]))
def test_a_source_that_cannot_report_a_position_replays_rather_than_skipping(
    ckpt_root, table, chunk
):
    """A source with no `snapshot_position` records none, so recovery has nothing to seek
    to and the restart re-reads from the start. That is at-least-once — the direction the
    whole design chooses on purpose, because a duplicate is absorbable and a gap is not —
    and it is worth pinning so the *other* direction can never be mistaken for correct.
    """
    name = f"prop-noresume-{uuid.uuid4().hex}"
    location = str(ckpt_root / name)
    for _ in range(2):
        query = _chunked(table, chunk).write.memory(
            name, trigger=bt.Trigger.available_now(), output_mode="append", checkpoint=location
        )
        assert query.await_termination(60) is True
        assert query.exception() is None, query.exception()
        assert _rowset(bt.read_memory(name).collect()) == _rowset(table)
