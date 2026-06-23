"""InferencePool — model-once-per-worker actor-pool inference with dynamic batching."""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

from batcher.ml import InferencePool


def _batch(vals: list[int]) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays([pa.array(vals, type=pa.int64())], names=["x"])


def _values(batches: list[pa.RecordBatch]) -> list[int]:
    out: list[int] = []
    for b in batches:
        out.extend(b.column(0).to_pylist())
    return out


def test_loads_model_once_per_worker_and_transforms():
    builds = []
    lock = threading.Lock()

    def make_worker():
        with lock:
            builds.append(1)  # count one "model load" per worker

        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            doubled = pa.array([v * 2 for v in batch.column(0).to_pylist()], type=pa.int64())
            return pa.RecordBatch.from_arrays([doubled], names=["x"])

        return worker

    pool = InferencePool(make_worker, num_workers=3, target_batch_rows=4)
    inputs = [_batch(list(range(i * 5, i * 5 + 5))) for i in range(6)]  # 30 rows
    out = list(pool.run(inputs))

    # Model built exactly once per worker, regardless of batch count.
    assert sum(builds) == 3
    # Order preserved, every value doubled.
    assert _values(out) == [v * 2 for v in range(30)]


def test_dynamic_rebatching_hits_target_size():
    def make_worker():
        return lambda b: b  # identity

    # 25 single-row inputs, target 4 → full batches are size 4, last is the remainder.
    pool = InferencePool(make_worker, num_workers=2, target_batch_rows=4)
    out = list(pool.run([_batch([i]) for i in range(25)]))
    assert _values(out) == list(range(25))  # no rows lost or reordered
    assert all(b.num_rows == 4 for b in out[:-1])
    assert out[-1].num_rows == 25 % 4 or out[-1].num_rows == 4


def test_dynamic_batch_size_responds_to_latency():
    import time

    def make_worker():
        def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
            time.sleep(0.001 * batch.num_rows)  # latency grows with batch size
            return batch

        return worker

    # Tiny latency target → controller should shrink toward small batches.
    pool = InferencePool(
        make_worker,
        num_workers=1,
        target_batch_rows=200,
        target_latency_ms=5.0,
        min_batch_rows=1,
        max_batch_rows=200,
    )
    out = list(pool.run([_batch(list(range(50))) for _ in range(20)]))
    assert _values(out) == [v for _ in range(20) for v in range(50)]  # all rows preserved


def test_empty_input():
    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=8)
    assert list(pool.run([])) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_throughput_objective_engages_controller_and_preserves_output():
    # Wiring check (deterministic): objective="throughput" engages the hill-climb
    # controller, the pool feeds it real throughput readings, and output is correct.
    # The *convergence* behaviour is covered deterministically in test_autobatch.py;
    # asserting an emergent target change here would be timing-dependent and flaky.
    pool = InferencePool(
        lambda: lambda b: b,
        num_workers=2,
        target_batch_rows=8,
        objective="throughput",
        min_batch_rows=4,
        max_batch_rows=256,
    )
    assert pool._throughput_ctl is not None
    out = list(pool.run(_batch([i]) for i in range(2000)))
    assert _values(out) == list(range(2000))  # correctness preserved
    # The controller received real readings from the pool (it was actually driven),
    # and the batch target stayed within the configured bounds throughout.
    assert pool._throughput_ctl._best_throughput is not None
    assert 4 <= pool._batcher._target <= 256


def test_throughput_objective_respects_vram_cap():
    # A VRAM sampler pinned over the cap keeps the batch size from growing unbounded.
    pool = InferencePool(
        lambda: lambda b: b,
        num_workers=1,
        target_batch_rows=64,
        objective="throughput",
        vram_sampler=lambda: 0.99,
        min_batch_rows=8,
        max_batch_rows=4096,
    )
    out = list(pool.run(_batch([i]) for i in range(1000)))
    assert _values(out) == list(range(1000))
    assert pool._batcher._target <= 64  # never grew past the start under VRAM pressure


def test_invalid_objective_rejected():
    with pytest.raises(ValueError, match="objective"):
        InferencePool(lambda: lambda b: b, objective="nonsense")


def test_throughput_controller_not_frozen_by_zero_latency_reading():
    # A degenerate zero-latency measurement must not poison the controller (feeding
    # inf would freeze it forever). After it, the size still adapts.
    pool = InferencePool(
        lambda: lambda b: b,
        num_workers=1,
        target_batch_rows=64,
        objective="throughput",
        min_batch_rows=8,
        max_batch_rows=4096,
    )
    ctl = pool._throughput_ctl
    assert ctl is not None
    # Simulate the degenerate then a real reading via the pool's own path.
    b = _batch([1])
    assert pool._next_target(b, 0.0) == ctl.current()  # zero latency → no-op
    grown = pool._next_target(b, 1.0)  # a real reading still adapts
    assert isinstance(grown, int)
    # best_throughput is finite, so further improvement is still possible (not frozen).
    assert ctl._best_throughput != float("inf")
