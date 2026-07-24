"""Class-UDF lifecycle: a model loaded once per owned call is torn down via `close()`.

A class `fn` is a load-once factory (`__init__` loads the model, `__call__` scores each batch).
When the call that built the instance owns it, its optional `close()` runs at the end of the
stage so a GPU allocation / HTTP session / DB connection is released deterministically. A
prebuilt instance (passed in by a long-lived owner) is that owner's to tear down, not here.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")


def test_class_udf_loads_once_and_closes():
    events: list[str] = []

    class Model:
        def __init__(self) -> None:
            events.append("load")

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch.set_column(0, "x", pc.add(batch.column("x"), 1))

        def close(self) -> None:
            events.append("close")

    out = bt.from_pydict({"x": [1, 2, 3]}).map_batches(Model).to_pydict()
    assert out == {"x": [2, 3, 4]}
    assert events == ["load", "close"]  # loaded once, torn down once


def test_close_that_raises_does_not_fail_query():
    class Bad:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

        def close(self) -> None:
            raise RuntimeError("teardown boom")

    # Results are already produced; a failing close() must be swallowed.
    assert bt.from_pydict({"x": [1]}).map_batches(Bad).to_pydict() == {"x": [1]}


def test_class_without_close_is_fine():
    class NoClose:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

    assert bt.from_pydict({"x": [9]}).map_batches(NoClose).to_pydict() == {"x": [9]}


def test_prebuilt_instance_is_not_closed_here():
    """Passing an instance (not a class) means the caller owns the lifetime — `close()` is the
    caller's responsibility and must NOT be called by the stage."""
    closed: list[str] = []

    class Model:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

        def close(self) -> None:
            closed.append("close")

    inst = Model()
    bt.from_pydict({"x": [1]}).map_batches(inst).collect()
    assert closed == []  # the stage did not own the instance, so it did not close it


def test_teardown_survives_a_raising_udf():
    """Even when the `fn` raises, the model this call built is still closed (finally)."""
    events: list[str] = []

    class Model:
        def __init__(self) -> None:
            events.append("load")

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            raise ValueError("inference failed")

        def close(self) -> None:
            events.append("close")

    with pytest.raises(Exception, match="inference failed"):
        bt.from_pydict({"x": [1]}).map_batches(Model).collect()
    assert events == ["load", "close"]  # torn down despite the failure
