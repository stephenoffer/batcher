"""A streaming query pushes source projection down, exactly as `iter_batches` does.

`_build_run_batch` optimized the plan (so it *had* Kyber's `source_projections`) but
returned only the runner, and `LocalRunner.stage` read with `iter_batches(None)`. So the
same pipeline pushed its projection down under `iter_batches` and decoded every column
under `ds.write(...)` streaming — and the distributed launcher, which already threaded the
projection, disagreed with the single-node one.

These tests pin the runner's contract at the seam where it broke.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.streaming_runner import LocalRunner

_SCHEMA = pa.schema([("a", pa.int64()), ("b", pa.int64()), ("c", pa.int64())])


class _SpySource:
    """Records the projection each `iter_batches` call was handed."""

    def __init__(self) -> None:
        self.asked: list[list[str] | None] = []

    def schema(self) -> pa.Schema:
        return _SCHEMA

    def iter_batches(self, projection=None):
        self.asked.append(projection)
        yield pa.RecordBatch.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, schema=_SCHEMA)


class _Processor:
    def process(self, batch):
        return [batch]


class _Sink:
    def __init__(self) -> None:
        self.written: list[pa.Table] = []

    def write_batch(self, batch_id: int, table: pa.Table) -> None:
        self.written.append(table)


@pytest.mark.integration
def test_projection_reaches_the_source():
    source = _SpySource()
    runner = LocalRunner(source, _Processor(), _Sink(), projection=["a"])
    runner.stage(0)
    assert source.asked == [["a"]], "the streaming runner must read through Kyber's pushdown"


@pytest.mark.integration
def test_no_pushdown_still_reads_everything():
    # A `map_batches` pipeline has no projection to push (the UDF is opaque to Kyber);
    # that case must keep reading every column.
    source = _SpySource()
    LocalRunner(source, _Processor(), _Sink()).stage(0)
    assert source.asked == [None]


@pytest.mark.integration
def test_launcher_returns_projection_for_a_relational_plan():
    import batcher as bt
    from batcher.api.streaming._launch import _build_run_batch

    ds = bt.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).select("a")
    run_batch, projection, _predicate = _build_run_batch(ds._plan, ds._sources)
    assert run_batch is not None
    # The whole point: a `select("a")` must not ask the source for b and c.
    assert projection is not None
    assert set(projection) == {"a"}


@pytest.mark.integration
def test_map_batches_plan_has_no_pushdown():
    import batcher as bt
    from batcher.api.streaming._launch import _build_run_batch

    ds = bt.from_pydict({"a": [1, 2]}).map_batches(lambda b: b)
    run_batch, projection, predicate = _build_run_batch(ds._plan, ds._sources)
    assert run_batch is not None
    assert projection is None and predicate is None
