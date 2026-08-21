"""The adaptive machinery, end to end: it may change the shape of the work, never the answer.

`InferencePool` re-chunks batches, grows and shrinks the size online, halves and retries on an
out-of-memory, and pulls its source ahead on a background thread. Every one of those is a
*scheduling* change, so the row multiset it produces must be identical to the sequential
answer no matter which of them fire. These drive a simulated device that fails above a
threshold — the failure a real VRAM ceiling produces — and hold the output to that.
"""

from __future__ import annotations

import threading
from collections import Counter

import pyarrow as pa
import pytest

from batcher.ml import InferencePool

pytestmark = pytest.mark.integration


def _batch(values: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays([pa.array(values, type=pa.int64())], names=["x"])


def _values(batches: list[pa.RecordBatch]) -> list[int]:
    return [v for b in batches for v in b.column(0).to_pylist()]


class FragileModel:
    """A model that refuses any batch above `limit` rows, the way a full device does."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.failures = 0
        self.max_seen = 0
        self.dispatched: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        with self._lock:
            self.max_seen = max(self.max_seen, batch.num_rows)
            self.dispatched.append(batch.num_rows)
        if batch.num_rows > self.limit:
            with self._lock:
                self.failures += 1
            raise RuntimeError(f"CUDA out of memory. Tried to allocate {batch.num_rows} rows")
        doubled = pa.array([v * 2 for v in batch.column(0).to_pylist()], type=pa.int64())
        return pa.RecordBatch.from_arrays([doubled], names=["x"])


@pytest.mark.parametrize("limit", [1, 3, 8, 64, 1000])
def test_every_row_survives_the_oom_ladder(limit):
    # The halving retry re-runs the halves and concatenates them, so the answer is the whole
    # batch's answer. A dropped or duplicated row here would be invisible in aggregate.
    model = FragileModel(limit)
    pool = InferencePool(lambda: model, num_workers=2, target_batch_rows=128)
    out = _values(list(pool.run(_batch(list(range(i * 10, i * 10 + 10))) for i in range(20))))
    assert sorted(out) == [v * 2 for v in range(200)]


@pytest.mark.parametrize("objective", ["latency", "throughput"])
def test_the_adaptive_objectives_do_not_change_the_answer(objective):
    kwargs = {"target_latency_ms": 5.0} if objective == "latency" else {}
    pool = InferencePool(
        lambda: lambda b: b,
        num_workers=3,
        target_batch_rows=7,
        objective=objective,
        **kwargs,
    )
    out = _values(list(pool.run(_batch([i]) for i in range(200))))
    assert out == list(range(200))  # in order, every row, exactly once


@pytest.mark.parametrize("prefetch", [0, 1, 2, 8])
def test_the_prefetch_depth_does_not_change_the_answer(prefetch):
    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=5, prefetch=prefetch)
    out = _values(list(pool.run(_batch([i, i + 1]) for i in range(0, 100, 2))))
    assert out == list(range(100))


def test_a_batch_that_cannot_shrink_far_enough_still_reports_the_failure():
    # A genuine over-allocation — one row that does not fit — is not a too-large batch, and
    # halving it forever would turn a clear error into a hang.
    class _AlwaysFails:
        def __call__(self, batch):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 row")

    pool = InferencePool(lambda: _AlwaysFails(), num_workers=1, target_batch_rows=4)
    with pytest.raises(RuntimeError, match="out of memory"):
        list(pool.run(iter([_batch([1, 2, 3, 4])])))


def test_a_non_memory_error_is_not_retried_as_one():
    # The halving ladder must not be spent on an error a smaller batch cannot fix.
    class _Broken:
        calls = 0

        def __call__(self, batch):
            type(self).calls += 1
            raise ValueError("column 'y' is missing")

    pool = InferencePool(lambda: _Broken(), num_workers=1, target_batch_rows=4)
    with pytest.raises(ValueError, match="missing"):
        list(pool.run(iter([_batch([1, 2, 3, 4])])))
    assert _Broken.calls == 1, f"retried a non-memory error {_Broken.calls} times"


def test_the_dispatch_size_converges_on_what_fits_rather_than_collapsing():
    """The ladder recovers the rows; what it *reports* decides whether the run recovers.

    Two failures lived here, and the simulation is what exposed both. Reporting every level of
    the bisection inflated the controller's consecutive-failure streak by the ladder's depth,
    which drove the ceiling to **1 row** against a model that accepts 16 — permanently, so the
    rest of the run dispatched one row at a time at a sixteenth of the achievable throughput.
    Reporting only the outermost size fixed the collapse but converged at the backoff ratio
    over eight more failed batches. Reporting the smallest size that failed hands over the
    bracket the bisection actually established.
    """
    model = FragileModel(limit=16)
    pool = InferencePool(
        lambda: model, num_workers=1, target_batch_rows=256, objective="throughput"
    )
    out = _values(list(pool.run(_batch(list(range(i * 32, i * 32 + 32))) for i in range(50))))
    assert sorted(out) == [v * 2 for v in range(1600)]
    # The most common size in the tail, which is the size the run settled on. The *last* few
    # dispatches include the flush remainder, which is legitimately a partial batch and says
    # nothing about the target.
    settled = Counter(model.dispatched[-20:]).most_common(1)[0][0]
    # Just under what fits is the goal: a ceiling at or below the size that failed, and not
    # the collapse to a handful of rows that reporting every ladder level produced.
    assert settled <= model.limit, f"settled above the limit at {settled}"
    # The lower bound is deliberately loose. Several oversized batches are already in flight
    # when the first report lands, so two or three genuinely independent failures land in a
    # row, and the controller's streak backoff — which exists precisely for repeats — deepens
    # on each. Measured over 30 runs the settled size is 11 (73%), 8 (17%) or 5 (10%) against a
    # limit of 16. What must not happen is the collapse to a single row that reporting every
    # ladder level produced, and that is what this bound guards.
    assert settled >= model.limit // 4, f"collapsed well below what fits: {settled}"


def test_the_bisection_reports_one_event_carrying_its_tightest_bound():
    # Directly, without the pool: the ladder over 64 rows against a limit of 8 discovers that
    # 16 fails and 8 fits, and 16 is the number worth reporting.
    from batcher.ml.inference.pool import _run_with_oom_retry

    reported: list[int] = []
    model = FragileModel(limit=8)
    _run_with_oom_retry(model, _batch(list(range(64))), reported.append)
    assert len(reported) == 1, f"one bisection reported {len(reported)} separate failures"
    assert reported == [16], f"reported {reported} rather than the tightest failing size"


def test_a_ladder_that_cannot_recover_still_reports_what_it_learned():
    # The bound is worth keeping even when this batch is lost: the next one should not repeat
    # the whole descent.
    from batcher.ml.inference.pool import _run_with_oom_retry

    class _AlwaysFails:
        def __call__(self, batch):
            raise RuntimeError("CUDA out of memory. Tried to allocate rows")

    reported: list[int] = []
    with pytest.raises(RuntimeError):
        _run_with_oom_retry(_AlwaysFails(), _batch(list(range(32))), reported.append)
    assert reported == [1]
