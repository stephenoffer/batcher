"""Wave-3 regression: the CUDA-OOM halving path must not silently drop rows.

`_run_with_oom_retry` survives a transient OOM by splitting a batch, running each
half, and concatenating the halves back into one batch. The pre-fix merge used
``pa.Table.from_batches([left, right]).combine_chunks().to_batches()[0]``. When the
concatenated column overflows the 32-bit Arrow offset limit (~2 GiB — reachable when
inference emits a large binary/string/list column), ``combine_chunks().to_batches()``
splits into *multiple* batches and ``[0]`` kept only the first, silently discarding
the rest (data loss). The fix uses ``pa.concat_batches`` which preserves every row and
raises a clear error on a genuine >2 GiB overflow instead of dropping data.

The large-output test is heavy (~2.2 GiB) and skips when memory is tight.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

from batcher.ml.inference import _run_with_oom_retry

pytestmark = pytest.mark.unit


class _FakeOOM(Exception):
    """Structurally a CUDA OOM (``_is_cuda_oom`` matches on the class name)."""


_FakeOOM.__name__ = "OutOfMemoryError"


def test_oom_retry_concatenates_all_halves_small() -> None:
    # The ordinary split path (no overflow) must return EVERY row, in order.
    calls = {"n": 0}

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        calls["n"] += 1
        if batch.num_rows > 1:
            raise _FakeOOM("CUDA out of memory")
        return batch  # identity: one row at a time

    inp = pa.record_batch({"x": list(range(8))})
    out, _ms = _run_with_oom_retry(worker, inp)
    assert out.num_rows == 8
    assert out.column("x").to_pylist() == list(range(8))
    assert calls["n"] > 1  # the batch really did split


def _available_bytes() -> int:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return 0


def test_oom_retry_does_not_silently_drop_on_offset_overflow() -> None:
    # Two halves whose binary outputs concatenate to >2 GiB: the pre-fix code returned
    # only the FIRST half's row (silent data loss). The fix must NOT return a truncated
    # single row — it raises a clear ArrowInvalid instead.
    if _available_bytes() < 3_500_000_000:
        pytest.skip("needs ~2.2 GiB of free memory to build a >2 GiB Arrow column")

    per_row = 1_090_000_000  # two of these overflow the 32-bit binary offset (2**31)

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        if batch.num_rows > 1:
            raise _FakeOOM("CUDA out of memory")
        return pa.record_batch([pa.array([b"a" * per_row], pa.binary())], names=["y"])

    inp = pa.record_batch({"x": [0, 1]})  # two tiny input rows -> two large outputs
    with pytest.raises(pa.lib.ArrowInvalid):
        out, _ms = _run_with_oom_retry(worker, inp)
        # The pre-fix path reached here with a single row (the second was dropped).
        assert out.num_rows == 2, "silently dropped a half's rows"
