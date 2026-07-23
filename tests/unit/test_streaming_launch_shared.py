"""Pin the behavior the two streaming launchers duplicate, so they cannot drift again.

Two derivations exist twice in the tree, each carrying a comment saying so:

1. "optimize the breaker-free plan once, then run it per micro-batch" —
   `api/streaming/_launch.py::_build_run_batch` and
   `api/terminal/stream/dispatch.py::_iter_streaming`. These already drifted once: the
   launcher returned the runner *without* the source projection, so `LocalRunner` read
   the source with `iter_batches(None)` while the identical `iter_batches` pipeline
   pushed the projection down — single-node streaming to a sink decoded every column.
2. `Distinct` → `Aggregate` over all columns —
   `core/streaming_query.py::_distinct_as_aggregate` and the derivation inlined in
   `core/streaming.py::stream_distinct` (and a third copy in
   `dist/executors/distinct.py`).

Neither could be collapsed to one definition without editing a file outside this
change's scope, so these tests are the guard: they assert the *observable* equality of
the two copies rather than trusting a comment.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.streaming._launch import _build_run_batch
from batcher.api.terminal.stream.dispatch import _iter_streaming
from batcher.core.streaming_query import _distinct_as_aggregate

pytestmark = pytest.mark.unit


class _SpySource:
    """An in-memory source that records the projection/predicate it was read with."""

    supports_predicate = True

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches
        self.calls: list[tuple[list[str] | None, dict | None]] = []

    def schema(self) -> pa.Schema:
        return self._batches[0].schema

    def row_count(self) -> int:
        return sum(b.num_rows for b in self._batches)

    def statistics(self):
        return None

    def iter_batches(self, projection=None, predicate=None):
        self.calls.append((list(projection) if projection else projection, predicate))
        for b in self._batches:
            yield b.select(projection) if projection else b


def _batches() -> list[pa.RecordBatch]:
    return [
        pa.record_batch({"a": [1, 2, 3], "b": [10, 20, 30], "c": [7, 8, 9]}),
        pa.record_batch({"a": [4, 5], "b": [40, 50], "c": [1, 2]}),
    ]


def _pipeline(build):
    """Return `(plan, sources)` for a pipeline built over a fresh in-memory relation."""
    ds = build(bt.from_arrow(pa.Table.from_batches(_batches())))
    return ds._plan, list(ds._sources)


# --- (1) the launcher and the iter_batches path must push down identically ----------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.select("a"), id="project"),
        pytest.param(lambda ds: ds.filter(bt.col("a") > 2).select("b"), id="filter+project"),
        pytest.param(lambda ds: ds.select("a", "b").filter(bt.col("b") > 15), id="project+filter"),
        pytest.param(lambda ds: ds.with_columns(d=bt.col("a") * 2), id="with_columns"),
    ],
)
def test_launcher_pushes_the_same_projection_and_predicate_as_iter_batches(build):
    """The launcher's `(projection, predicate)` == what the `iter_batches` path reads with.

    This is the exact drift that shipped once: a launcher that pushed nothing while the
    streaming-collect path pushed a projection down.
    """
    plan, sources = _pipeline(build)

    dispatch_source = _SpySource(_batches())
    list(_iter_streaming(plan, [dispatch_source], None))
    assert len(dispatch_source.calls) == 1
    dispatch_projection, dispatch_predicate = dispatch_source.calls[0]

    _run_batch, launch_projection, launch_predicate = _build_run_batch(plan, sources)

    assert launch_projection == dispatch_projection
    assert launch_predicate == dispatch_predicate


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.select("a"), id="project"),
        pytest.param(lambda ds: ds.filter(bt.col("a") > 2).select("b"), id="filter+project"),
        pytest.param(lambda ds: ds.with_columns(d=bt.col("a") * 2), id="with_columns"),
    ],
)
def test_launcher_and_iter_batches_produce_the_same_rows(build):
    """Driving the launcher's runner over the pushed-down read equals the streamed rows."""
    plan, sources = _pipeline(build)

    expected = pa.Table.from_batches(list(_iter_streaming(plan, [_SpySource(_batches())], None)))

    run_batch, projection, predicate = _build_run_batch(plan, sources)
    source = _SpySource(_batches())
    out: list[pa.RecordBatch] = []
    for batch in source.iter_batches(projection, predicate):
        out.extend(b for b in run_batch(batch) if b.num_rows)
    actual = pa.Table.from_batches(out)

    assert actual.schema.names == expected.schema.names
    assert actual.to_pydict() == expected.to_pydict()


def test_launcher_leaves_a_map_batches_pipeline_unpushed():
    """A `map_batches` UDF is opaque to Kyber, so both paths read every column."""
    ds = bt.from_arrow(pa.Table.from_batches(_batches())).map_batches(lambda b: b)
    run_batch, projection, predicate = _build_run_batch(ds._plan, list(ds._sources))
    assert projection is None
    assert predicate is None
    assert run_batch is not None

    dispatch_source = _SpySource(_batches())
    list(_iter_streaming(ds._plan, [dispatch_source], None))
    assert all(call == (None, None) for call in dispatch_source.calls)


# --- (2) the two Distinct -> Aggregate derivations must agree -----------------------


def test_distinct_as_aggregate_matches_the_stream_distinct_derivation(monkeypatch):
    """`_distinct_as_aggregate` builds the same `Aggregate` `stream_distinct` folds."""
    from batcher.core import streaming

    ds = bt.from_arrow(pa.Table.from_batches(_batches())).select("a", "b").distinct()
    distinct = ds._plan

    seen: list[object] = []

    def _capture(agg, source, batch_size=None):
        seen.append(agg)
        return iter(())

    monkeypatch.setattr(streaming, "stream_aggregate", _capture)
    list(streaming.stream_distinct(distinct, _SpySource(_batches())))

    assert len(seen) == 1
    # `Expr.__eq__` builds an expression, so dataclass equality on a plan is unusable —
    # compare the IR, which is the wire contract both derivations must agree on.
    assert _distinct_as_aggregate(distinct).to_ir() == seen[0].to_ir()
