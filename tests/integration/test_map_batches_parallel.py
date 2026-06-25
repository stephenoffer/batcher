"""Parallel `map_batches` — concurrent per-batch inference, order preserved."""

from __future__ import annotations

import os
import threading
import time

import pyarrow as pa
import pytest

import batcher as bt


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    arr = pa.array([x * 2 for x in batch.column("v").to_pylist()], type=pa.int64())
    return pa.RecordBatch.from_arrays([arr], names=["v"])


def _double_np(d: dict) -> dict:
    return {"v": d["v"] * 2}


def _pid_col(batch: pa.RecordBatch) -> pa.RecordBatch:
    pid = pa.array([os.getpid()] * batch.num_rows, type=pa.int64())
    return pa.RecordBatch.from_arrays([batch.column("v"), pid], names=["v", "pid"])


class _Doubler:
    """A factory/class `fn` — load-once-per-worker pattern (forces a threads fallback)."""

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return _double(batch)


def test_parallel_preserves_order_and_values():
    t = pa.table({"v": list(range(1000))})
    out = bt.from_arrow(t).map_batches(_double, batch_size=100, num_workers=4).collect()
    assert out.column("v").to_pylist() == [x * 2 for x in range(1000)]


def test_parallel_matches_sequential():
    t = pa.table({"v": list(range(1000))})
    seq = bt.from_arrow(t).map_batches(_double, batch_size=100, num_workers=1).collect()
    par = bt.from_arrow(t).map_batches(_double, batch_size=100, num_workers=4).collect()
    assert seq.to_pydict() == par.to_pydict()


def test_actually_runs_concurrently():
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def slow(batch: pa.RecordBatch) -> pa.RecordBatch:
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.02)  # hold the slot so others overlap
        with lock:
            state["active"] -= 1
        return batch

    t = pa.table({"v": list(range(400))})
    bt.from_arrow(t).map_batches(slow, batch_size=50, num_workers=4).collect()
    assert state["max"] >= 2  # multiple batches ran at once


def test_single_worker_default_is_sequential():
    # num_workers=1 (default) must keep the simple sequential path.
    t = pa.table({"v": [1, 2, 3]})
    out = bt.from_arrow(t).map_batches(_double).collect()
    assert out.column("v").to_pylist() == [2, 4, 6]


# --- multiprocessing path: GIL-bound pure-Python UDFs across cores --------------


def test_multiprocessing_matches_sequential_and_threads():
    t = pa.table({"v": list(range(1000))})
    seq = bt.from_arrow(t).map_batches(_double, batch_size=100, num_workers=1).collect()
    thr = bt.from_arrow(t).map_batches(_double, batch_size=100, num_workers=4).collect()
    proc = (
        bt.from_arrow(t)
        .map_batches(_double, batch_size=100, num_workers=4, multiprocessing=True)
        .collect()
    )
    assert seq.to_pydict() == thr.to_pydict() == proc.to_pydict()


def test_multiprocessing_preserves_order():
    t = pa.table({"v": list(range(1000))})
    out = (
        bt.from_arrow(t)
        .map_batches(_double, batch_size=100, num_workers=4, multiprocessing=True)
        .collect()
    )
    assert out.column("v").to_pylist() == [x * 2 for x in range(1000)]


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs multiple cores")
def test_multiprocessing_actually_uses_processes():
    # The per-batch calls run in child processes → more than one distinct PID.
    t = pa.table({"v": list(range(1000))})
    out = (
        bt.from_arrow(t)
        .map_batches(_pid_col, batch_size=50, num_workers=4, multiprocessing=True)
        .collect()
    )
    pids = set(out.column("pid").to_pylist())
    assert pids and os.getpid() not in pids  # ran off the driver
    assert len(pids) >= 2  # spread across multiple workers


def test_multiprocessing_lambda_falls_back_and_is_correct():
    # A lambda is unpicklable → silently falls back to threads, still correct.
    t = pa.table({"v": list(range(300))})
    out = (
        bt.from_arrow(t)
        .map_batches(lambda b: _double(b), batch_size=50, num_workers=4, multiprocessing=True)
        .collect()
    )
    assert out.column("v").to_pylist() == [x * 2 for x in range(300)]


def test_multiprocessing_factory_falls_back_and_is_correct():
    # A class/factory `fn` would reload per process → falls back to threads.
    t = pa.table({"v": list(range(300))})
    out = (
        bt.from_arrow(t)
        .map_batches(_Doubler, batch_size=50, num_workers=4, multiprocessing=True)
        .collect()
    )
    assert out.column("v").to_pylist() == [x * 2 for x in range(300)]


def test_multiprocessing_non_pyarrow_format_falls_back_and_is_correct():
    # A non-pyarrow batch_format needs an unpicklable closure → threads fallback.
    t = pa.table({"v": list(range(300))})
    out = (
        bt.from_arrow(t)
        .map_batches(
            _double_np, batch_size=50, num_workers=4, batch_format="numpy", multiprocessing=True
        )
        .collect()
    )
    assert out.column("v").to_pylist() == [x * 2 for x in range(300)]


def test_multiprocessing_single_batch_is_sequential():
    # One batch never spawns a pool, even when opted in.
    t = pa.table({"v": [1, 2, 3]})
    out = bt.from_arrow(t).map_batches(_double, num_workers=4, multiprocessing=True).collect()
    assert out.column("v").to_pylist() == [2, 4, 6]
