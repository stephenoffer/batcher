"""`InferencePool` releases the models it built, rather than waiting for the collector.

The pool's whole reason to exist is that each of `num_workers` slots holds one loaded
model, and a model holds a CUDA context, an HTTP session, or a database handle. Those were
released only whenever the garbage collector happened to reach them, so a script running two
pools back to back could hold both generations in VRAM at once.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.ml.inference import InferencePool

pytestmark = pytest.mark.unit


class _Model:
    """A worker that records its own construction and teardown."""

    built = 0
    closed = 0

    def __init__(self) -> None:
        _Model.built += 1

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch

    def close(self) -> None:
        _Model.closed += 1


@pytest.fixture(autouse=True)
def _reset() -> None:
    _Model.built = _Model.closed = 0


def _batches(n: int = 4) -> list[pa.RecordBatch]:
    return [pa.record_batch({"x": [i]}) for i in range(n)]


def test_every_built_worker_is_closed() -> None:
    pool = InferencePool(_Model, num_workers=3, target_batch_rows=1)
    assert sum(b.num_rows for b in pool.run(iter(_batches()))) == 4
    assert _Model.built == 3
    assert _Model.closed == 3


def test_a_shared_worker_is_closed_once() -> None:
    """A factory may hand every slot the same object; closing it N times is not a teardown."""
    shared = _Model()
    pool = InferencePool(lambda: shared, num_workers=4, target_batch_rows=1)
    list(pool.run(iter(_batches())))
    assert _Model.closed == 1


def test_teardown_runs_even_when_a_worker_raises() -> None:
    class _Boom(_Model):
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            raise RuntimeError("model died")

    pool = InferencePool(_Boom, num_workers=2, target_batch_rows=1)
    with pytest.raises(RuntimeError, match="model died"):
        list(pool.run(iter(_batches())))
    assert _Model.closed == 2


def test_a_worker_without_close_is_fine() -> None:
    pool = InferencePool(lambda: lambda batch: batch, num_workers=2, target_batch_rows=1)
    assert sum(b.num_rows for b in pool.run(iter(_batches()))) == 4


def test_a_factory_that_fails_partway_releases_what_it_already_built() -> None:
    """The ordinary multi-GPU failure: the second of several models finds the device full.

    Leaking the ones already loaded turns a recoverable "size the pool smaller" into an OOM
    that outlives the failure and takes the retry with it.
    """
    calls = {"n": 0}

    def factory() -> _Model:
        calls["n"] += 1
        if calls["n"] == 3:
            raise MemoryError("CUDA out of memory")
        return _Model()

    pool = InferencePool(factory, num_workers=4, target_batch_rows=1)
    with pytest.raises(MemoryError, match="out of memory"):
        list(pool.run(iter(_batches())))
    assert _Model.built == 2
    assert _Model.closed == 2


def test_a_consumer_that_stops_early_does_not_pay_for_the_whole_inflight_window() -> None:
    """Abandoning the iterator cancels the batches that have not started.

    A `limit` over a streamed inference otherwise waited on every submitted forward pass
    before returning, because the executor's shutdown waits for its whole queue.
    """
    seen: list[int] = []

    class _Slow(_Model):
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            seen.append(batch.num_rows)
            return batch

    pool = InferencePool(_Slow, num_workers=1, target_batch_rows=1, max_inflight=1)
    stream = pool.run(iter(_batches(64)))
    assert next(stream).num_rows == 1
    stream.close()
    # One worker with an in-flight bound of one cannot have started more than a couple; the
    # point is that it is bounded rather than the whole 64-batch input.
    assert len(seen) < 8
    assert _Model.closed == 1
