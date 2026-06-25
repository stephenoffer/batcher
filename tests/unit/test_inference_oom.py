"""Phase-1 fault tolerance: CUDA-OOM survival in the inference pool.

A transient VRAM spike must degrade the batch (halve and retry), not kill the job —
while a genuine per-row over-allocation, or any non-OOM error, still surfaces.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.ml.inference import _is_cuda_oom, _run_with_oom_retry


class _FakeCudaOOM(RuntimeError):
    """Mimics torch's message-bearing OOM without importing torch."""


def _batch(n: int) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict({"x": list(range(n))})


def test_is_cuda_oom_recognizes_message_and_type():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert not _is_cuda_oom(RuntimeError("some other runtime error"))
    assert not _is_cuda_oom(ValueError("nope"))

    class OutOfMemoryError(Exception):
        pass

    assert _is_cuda_oom(OutOfMemoryError("device oom"))  # matched by type name


def test_oom_retry_halves_until_it_fits():
    # A worker that OOMs on any batch larger than 2 rows; succeeds at <= 2.
    seen_sizes: list[int] = []

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen_sizes.append(batch.num_rows)
        if batch.num_rows > 2:
            raise RuntimeError("CUDA out of memory")
        x = batch.column("x").to_pylist()
        return pa.RecordBatch.from_pydict({"x": x, "y": [v * 10 for v in x]})

    out, latency_ms = _run_with_oom_retry(worker, _batch(8))
    rows = pa.Table.from_batches([out]).to_pylist()
    # Every input row produced exactly once, in order, despite the splits.
    assert [r["x"] for r in rows] == list(range(8))
    assert all(r["y"] == r["x"] * 10 for r in rows)
    assert max(seen_sizes) == 8 and min(seen_sizes) <= 2  # it did split down
    assert latency_ms >= 0


def test_oom_on_a_single_row_reraises():
    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        raise RuntimeError("CUDA out of memory")  # OOMs even at 1 row

    with pytest.raises(RuntimeError, match="out of memory"):
        _run_with_oom_retry(worker, _batch(4))


def test_non_oom_error_is_not_retried():
    calls = []

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        calls.append(batch.num_rows)
        raise ValueError("a real bug")

    with pytest.raises(ValueError, match="a real bug"):
        _run_with_oom_retry(worker, _batch(8))
    assert calls == [8]  # no halving — surfaced immediately
