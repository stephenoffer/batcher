"""Async (`async def`) `map_batches` UDFs: concurrent I/O-bound calls on one event loop.

An `async def` fn (or a class whose `__call__` is async) is detected from the callable and
routed to an asyncio runner instead of the thread/process pools. Its batches run concurrently,
bounded by `max_concurrency`, in input order; `timeout` there is a real coroutine cancel.
"""

from __future__ import annotations

import asyncio
import time

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")


def _add(batch: pa.RecordBatch, k: int) -> pa.RecordBatch:
    return batch.set_column(0, "x", pc.add(batch.column("x"), k))


def test_async_fn_transforms_and_preserves_order():
    async def enrich(batch: pa.RecordBatch) -> pa.RecordBatch:
        await asyncio.sleep(0.005)
        return _add(batch, 100)

    out = bt.from_pydict({"x": list(range(10))}).map_batches(enrich, batch_size=3).to_pydict()
    assert out["x"] == [v + 100 for v in range(10)]  # order preserved across concurrent batches


def test_async_class_udf_loads_once():
    class Model:
        def __init__(self) -> None:
            self.bias = 1000

        async def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            await asyncio.sleep(0.001)
            return _add(batch, self.bias)

    out = bt.from_pydict({"x": [1, 2, 3]}).map_batches(Model).to_pydict()
    assert out == {"x": [1001, 1002, 1003]}


def test_async_bounds_concurrency():
    """No more than `max_concurrency` coroutines run at once."""
    state = {"cur": 0, "max": 0}

    async def track(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.02)
        state["cur"] -= 1
        return batch

    bt.from_pydict({"x": list(range(40))}).map_batches(
        track, batch_size=5, max_concurrency=4
    ).collect()
    assert state["max"] <= 4  # semaphore held the line
    assert state["max"] >= 2  # and it genuinely overlapped


def test_async_overlaps_faster_than_serial():
    """Concurrent awaits finish in far less than the serial sum."""

    async def slow(batch: pa.RecordBatch) -> pa.RecordBatch:
        await asyncio.sleep(0.1)
        return batch

    t0 = time.perf_counter()
    bt.from_pydict({"x": list(range(16))}).map_batches(
        slow, batch_size=2, max_concurrency=8
    ).collect()  # 8 batches * 0.1s serial = 0.8s; concurrent ~0.1s
    assert time.perf_counter() - t0 < 0.5


def test_async_timeout_cancels_and_raises():
    async def hang(batch: pa.RecordBatch) -> pa.RecordBatch:
        await asyncio.sleep(5)
        return batch

    t0 = time.perf_counter()
    with pytest.raises(Exception, match="timeout"):
        bt.from_pydict({"x": [1]}).map_batches(hang, timeout=0.2).collect()
    assert time.perf_counter() - t0 < 2.0  # cancelled, not waited out


def test_async_retry_recovers():
    state = {"n": 0}

    async def flaky(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("429 rate limited")
        return _add(batch, 1)

    out = (
        bt.from_pydict({"x": [5]}).map_batches(flaky, max_retries=3, retry_backoff=0.0).to_pydict()
    )
    assert out == {"x": [6]}
    assert state["n"] == 3


def test_async_numpy_batch_format():
    import numpy as np

    async def scale(batch: dict) -> dict:
        await asyncio.sleep(0.001)
        return {"x": (batch["x"] * 2).astype(np.int64)}

    out = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(scale, batch_format="numpy", output_columns=["x"])
        .to_pydict()
    )
    assert out == {"x": [2, 4, 6]}


def test_async_empty_input():
    async def enrich(batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch

    out = (
        bt.from_pydict({"x": [1, 2, 3]})
        .filter(bt.col("x") > 100)  # empties the input
        .map_batches(enrich)
        .to_pydict()
    )
    assert out == {"x": []}


def test_async_per_row_map():
    async def enrich(row: dict) -> dict:
        await asyncio.sleep(0.002)
        return {"x": row["x"], "y": row["x"] * 10}

    out = (
        bt.from_pydict({"x": list(range(8))})
        .ml.map(enrich, output_columns=["x", "y"], max_concurrency=8)
        .to_pydict()
    )
    assert out["y"] == [v * 10 for v in range(8)]  # order preserved


def test_async_per_row_flat_map():
    async def dup(row: dict) -> list[dict]:
        await asyncio.sleep(0.001)
        return [{"x": row["x"]}, {"x": row["x"]}]

    out = bt.from_pydict({"x": [1, 2, 3]}).ml.flat_map(dup, output_columns=["x"]).to_pydict()
    assert out == {"x": [1, 1, 2, 2, 3, 3]}


def test_async_udf_inside_a_running_event_loop():
    """An async `map_batches` / `map` driven from inside a running loop (Jupyter, async apps)
    must not raise `asyncio.run() cannot be called from a running event loop`."""

    async def enrich(batch: pa.RecordBatch) -> pa.RecordBatch:
        await asyncio.sleep(0.001)
        return _add(batch, 1)

    async def per_row(row: dict) -> dict:
        await asyncio.sleep(0.001)
        return {"x": row["x"] + 10}

    async def driver() -> tuple[dict, dict]:
        a = bt.from_pydict({"x": [1, 2, 3]}).map_batches(enrich).to_pydict()
        b = bt.from_pydict({"x": [1, 2]}).ml.map(per_row, output_columns=["x"]).to_pydict()
        return a, b

    batch_out, row_out = asyncio.run(driver())
    assert batch_out == {"x": [2, 3, 4]}
    assert row_out == {"x": [11, 12]}


def test_async_class_with_constructor_kwargs():
    """An async class UDF bound with fn_constructor_kwargs stays async through the wrapper
    (its coroutine `__call__` is awaited), instead of being coerced as an un-awaited coroutine."""

    class AsyncModel:
        def __init__(self, bias: int = 0) -> None:
            self.bias = bias

        async def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            await asyncio.sleep(0.001)
            return _add(batch, self.bias)

    out = (
        bt.from_pydict({"x": [1, 2]})
        .ml.map_batches(AsyncModel, fn_constructor_kwargs={"bias": 100})
        .to_pydict()
    )
    assert out == {"x": [101, 102]}


def test_async_fn_with_kwargs():
    async def af(batch: pa.RecordBatch, k: int = 1) -> pa.RecordBatch:
        return _add(batch, k)

    out = bt.from_pydict({"x": [1, 2]}).ml.map_batches(af, fn_kwargs={"k": 50}).to_pydict()
    assert out == {"x": [51, 52]}


def test_async_max_errored_rows_isolates_bad_rows():
    """An async `fn` gets the same row-level dirty-data tolerance as a sync one: a failing row
    is retried, then isolated by bisection and dropped up to the budget."""

    async def flaky(batch: pa.RecordBatch) -> pa.RecordBatch:
        xs = batch.column("x").to_pylist()
        if any(v % 5 == 0 for v in xs):
            raise ValueError("bad row")
        return batch

    out = (
        bt.from_pydict({"x": list(range(1, 21))})
        .map_batches(flaky, batch_size=4, max_errored_rows=10)
        .to_pydict()
    )
    assert sorted(out["x"]) == [v for v in range(1, 21) if v % 5 != 0]  # 5,10,15,20 dropped


def test_async_max_errored_rows_budget_exhausted_raises():
    async def flaky(batch: pa.RecordBatch) -> pa.RecordBatch:
        if any(v % 5 == 0 for v in batch.column("x").to_pylist()):
            raise ValueError("bad row")
        return batch

    with pytest.raises(Exception, match="bad row"):
        bt.from_pydict({"x": list(range(1, 21))}).map_batches(
            flaky, batch_size=4, max_errored_rows=1
        ).collect()


def test_async_per_row_bounds_concurrency_within_a_batch():
    state = {"cur": 0, "max": 0}

    async def track(row: dict) -> dict:
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return {"x": row["x"]}

    bt.from_pydict({"x": list(range(30))}).ml.map(
        track, output_columns=["x"], batch_size=30, max_concurrency=5
    ).collect()
    assert state["max"] <= 5
    assert state["max"] >= 2  # genuinely overlapped
