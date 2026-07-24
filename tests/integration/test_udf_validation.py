"""Eager, actionable validation of a `map_batches` / `map` / `flat_map` `fn` at the API edge.

A non-callable `fn`, or a model class that forgot `__call__`, used to fail deep in a worker
with an opaque error. These raise a `PlanError` naming the problem before any work starts.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3]})


@pytest.mark.parametrize("bad", [42, "a string", None, 3.14, object()])
def test_non_callable_fn_rejected(bad):
    with pytest.raises(PlanError, match="must be callable"):
        _ds().map_batches(bad)


def test_class_without_call_rejected():
    class NoCall:
        def __init__(self) -> None:
            self.model = None

    with pytest.raises(PlanError, match="not callable"):
        _ds().map_batches(NoCall)


def test_class_with_call_accepted():
    class Good:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch.set_column(0, "x", pc.add(batch.column("x"), 1))

    assert _ds().map_batches(Good).to_pydict() == {"x": [2, 3, 4]}


def test_inherited_call_accepted():
    class Base:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

    class Sub(Base):
        pass

    assert _ds().map_batches(Sub).to_pydict() == {"x": [1, 2, 3]}


def test_map_validates_inner_fn():
    # The row adapter is itself callable, so a non-callable user fn must be caught first.
    with pytest.raises(PlanError, match="must be callable"):
        _ds().map(42)


def test_flat_map_validates_inner_fn():
    with pytest.raises(PlanError, match="must be callable"):
        _ds().flat_map("not callable")


@pytest.mark.parametrize(
    "cols,msg",
    [
        (["x", "x"], "duplicate"),
        (["x", ""], "non-empty"),
        ([], "cannot be empty"),
    ],
)
def test_bad_output_columns_rejected(cols, msg):
    with pytest.raises(PlanError, match=msg):
        _ds().map_batches(lambda b: b, output_columns=cols)


def test_valid_output_columns_accepted():
    assert _ds().map_batches(lambda b: b, output_columns=["x"]).to_pydict() == {"x": [1, 2, 3]}


def test_async_with_multiprocessing_warns():
    import warnings

    from batcher._internal.errors import PerformanceWarning

    async def af(batch):
        return batch

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ds().map_batches(af, multiprocessing=True)
    assert any(
        isinstance(w.message, PerformanceWarning) and "multiprocessing is ignored" in str(w.message)
        for w in caught
    )


def test_async_with_num_gpus_warns():
    import warnings

    from batcher._internal.errors import PerformanceWarning

    async def af(batch):
        return batch

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ds().map_batches(af, num_gpus=1)
    assert any(
        isinstance(w.message, PerformanceWarning) and "GPU auto-batching" in str(w.message)
        for w in caught
    )
