"""Parallel `map_batches` — concurrent per-batch inference, order preserved."""

from __future__ import annotations

import threading
import time

import pyarrow as pa

import batcher as bt


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    arr = pa.array([x * 2 for x in batch.column("v").to_pylist()], type=pa.int64())
    return pa.RecordBatch.from_arrays([arr], names=["v"])


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
