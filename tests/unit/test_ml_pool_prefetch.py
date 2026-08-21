"""`InferencePool`'s source prefetch — overlap, bounds, error handover, and teardown.

The pool already measured `blocked_ms` to tell a saturated pool from a starved one; these
cover the fix for the starved case. Everything drives `_prefetched` directly, so there is no
model, no device, and no timing dependence.
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

from batcher.ml.inference.pool import InferencePool, _prefetched

pytestmark = pytest.mark.unit


def _batch(values: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays([pa.array(values, type=pa.int64())], names=["x"])


def test_zero_depth_passes_the_source_straight_through():
    assert list(_prefetched(iter([1, 2, 3]), 0)) == [1, 2, 3]


def test_every_item_arrives_in_order():
    assert list(_prefetched(iter(range(50)), 4)) == list(range(50))


def test_an_empty_source_terminates():
    assert list(_prefetched(iter([]), 2)) == []


def test_the_source_runs_ahead_of_the_consumer():
    # The whole point: while the consumer is working, the source is already producing.
    produced: list[int] = []

    def source():
        for i in range(10):
            produced.append(i)
            yield i

    stream = _prefetched(source(), 3)
    first = next(stream)
    assert first == 0
    deadline = time.monotonic() + 5.0
    while len(produced) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(produced) >= 4  # depth 3 in the queue plus the one handed over
    stream.close()


def test_the_queue_bounds_how_far_ahead_the_source_runs():
    produced: list[int] = []

    def source():
        for i in range(1000):
            produced.append(i)
            yield i

    stream = _prefetched(source(), 2)
    next(stream)
    time.sleep(0.15)
    # Bounded: the depth, the item handed over, and the one the producer is parked on.
    assert len(produced) <= 8, f"prefetch ran {len(produced)} ahead of a depth of 2"
    stream.close()


def test_a_source_error_surfaces_on_the_consumers_thread():
    # A traceback printed on a daemon thread is not a failure anyone sees; the query must fail.
    def source():
        yield 1
        raise ValueError("bad shard")

    stream = _prefetched(source(), 2)
    assert next(stream) == 1
    with pytest.raises(ValueError, match="bad shard"):
        next(stream)


def test_an_error_on_the_very_first_item_still_surfaces():
    def source():
        raise ValueError("no source")
        yield  # pragma: no cover - unreachable, present to make this a generator

    with pytest.raises(ValueError, match="no source"):
        list(_prefetched(source(), 2))


def test_abandoning_the_stream_does_not_leave_a_thread_running():
    started = threading.active_count()

    def source():
        yield from range(10_000)

    stream = _prefetched(source(), 2)
    next(stream)
    stream.close()  # the `limit` / `break` case
    deadline = time.monotonic() + 5.0
    while threading.active_count() > started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= started


def test_the_pool_still_yields_every_row_in_order_with_prefetch_on():
    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=2)
    out = list(pool.run(_batch([i]) for i in range(9)))
    values = [v for batch in out for v in batch.column(0).to_pylist()]
    assert values == list(range(9))


def test_prefetch_can_be_turned_off_and_the_pool_is_unchanged():
    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=2, prefetch=0)
    out = list(pool.run(_batch([i]) for i in range(9)))
    values = [v for batch in out for v in batch.column(0).to_pylist()]
    assert values == list(range(9))


def test_a_source_error_fails_the_pool_rather_than_hanging_it():
    def source():
        yield _batch([1])
        raise RuntimeError("read failed")

    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=1)
    with pytest.raises(RuntimeError, match="read failed"):
        list(pool.run(source()))


class TestDynamicBatcherRetargetRace:
    """Retargeting mid-drain must not change which rows come out.

    `_target` is the one field two threads touch: the consumer drains on the thread driving
    `run`, and whichever worker hits a device out-of-memory lowers the target so the next batch
    is built smaller. `_drain` read that field three times per iteration — the loop condition,
    the slice, and the offset advance — so a retarget landing between them advanced the offset
    by a different number than the slice had just consumed.

    Nothing raises when that happens. Rows are **duplicated or dropped**, and the only symptom
    is a wrong answer on a run that also had an out-of-memory. Reproduced at 4 in 40 trials.
    """

    @staticmethod
    def _rows(batches) -> list[int]:
        return [v for b in batches for v in b.column(0).to_pylist()]

    def test_a_retarget_during_a_drain_cannot_change_the_row_sequence(self):
        from batcher.ml.inference.pool import _DynamicBatcher

        for _ in range(40):
            batcher = _DynamicBatcher(256)
            stop = threading.Event()

            def churn(b=batcher, s=stop):
                size = 8
                while not s.is_set():
                    b.set_target(size)
                    size = 256 if size == 8 else 8

            churner = threading.Thread(target=churn, daemon=True)
            churner.start()
            try:
                emitted: list[int] = []
                for i in range(40):
                    emitted += self._rows(batcher.push(_batch(list(range(i * 64, i * 64 + 64)))))
                emitted += self._rows(batcher.flush())
            finally:
                stop.set()
                churner.join(timeout=2)
            assert emitted == list(range(40 * 64)), "a mid-drain retarget corrupted the rows"

    def test_a_retarget_still_takes_effect_on_the_next_drain(self):
        # The snapshot must not turn the retarget into a no-op: it is what stops the next
        # batch walking into the out-of-memory that caused it.
        from batcher.ml.inference.pool import _DynamicBatcher

        batcher = _DynamicBatcher(64)
        assert [b.num_rows for b in batcher.push(_batch(list(range(64))))] == [64]
        batcher.set_target(16)
        assert [b.num_rows for b in batcher.push(_batch(list(range(64))))] == [16, 16, 16, 16]

    def test_every_row_survives_a_target_that_never_divides_evenly(self):
        from batcher.ml.inference.pool import _DynamicBatcher

        batcher = _DynamicBatcher(7)
        emitted: list[int] = []
        for i in range(10):
            emitted += self._rows(batcher.push(_batch(list(range(i * 10, i * 10 + 10)))))
        emitted += self._rows(batcher.flush())
        assert emitted == list(range(100))
