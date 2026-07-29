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
