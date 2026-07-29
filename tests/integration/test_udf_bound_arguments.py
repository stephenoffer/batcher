"""Configuring a UDF: `fn_args`/`fn_kwargs` and the class-constructor pair.

A model class is almost never zero-argument — ``Classifier("bert-base", device="cuda")`` is
the shape every framework uses — so a load-once UDF that could only be constructed with no
arguments forced users back onto a per-batch closure, reloading the model every batch. These
pin the four binding forms, that binding preserves load-once semantics, and that a bound
model still gets its `close()` called.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


class Scale:
    """A load-once 'model': scale ``x`` by a constructor factor."""

    builds = 0
    closes = 0

    def __init__(self, factor: int, *, bias: int = 0) -> None:
        Scale.builds += 1
        self.factor = factor
        self.bias = bias

    def __call__(self, batch: pa.RecordBatch, extra: int = 0) -> pa.RecordBatch:
        scaled = pc.add(pc.multiply(batch.column("x"), self.factor), self.bias + extra)
        return batch.set_column(0, "x", scaled)

    def close(self) -> None:
        Scale.closes += 1


@pytest.fixture(autouse=True)
def _reset() -> None:
    Scale.builds = Scale.closes = 0


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3]})


def test_constructor_positional_args(ds: bt.Dataset) -> None:
    assert ds.map_batches(Scale, fn_constructor_args=(10,)).to_pydict() == {"x": [10, 20, 30]}


def test_constructor_positional_and_keyword(ds: bt.Dataset) -> None:
    out = ds.map_batches(Scale, fn_constructor_args=(10,), fn_constructor_kwargs={"bias": 1})
    assert out.to_pydict() == {"x": [11, 21, 31]}


def test_per_call_kwargs_reach_the_model(ds: bt.Dataset) -> None:
    out = ds.map_batches(Scale, fn_constructor_args=(10,), fn_kwargs={"extra": 5})
    assert out.to_pydict() == {"x": [15, 25, 35]}


def test_per_call_positional_args_go_after_the_batch(ds: bt.Dataset) -> None:
    """`functools.partial` binds positionals in front of the batch, which is why this exists."""

    def scale(batch: pa.RecordBatch, factor: int) -> pa.RecordBatch:
        return batch.set_column(0, "x", pc.multiply(batch.column("x"), factor))

    assert ds.map_batches(scale, fn_args=(3,)).to_pydict() == {"x": [3, 6, 9]}


def test_binding_preserves_load_once(ds: bt.Dataset) -> None:
    """A bound class must stay a class, or every batch reloads the model."""
    many = bt.from_pydict({"x": list(range(10_000))})
    many.map_batches(Scale, fn_constructor_args=(2,), batch_size=1_000).to_pydict()
    assert Scale.builds == 1


def test_bound_model_still_gets_closed(ds: bt.Dataset) -> None:
    """The wrapper used to swallow `close`, so a bound model never released its GPU memory."""
    ds.map_batches(Scale, fn_constructor_args=(2,)).to_pydict()
    assert Scale.closes == 1


def test_constructor_args_on_a_plain_function_are_rejected(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="class fn"):
        ds.map_batches(lambda b: b, fn_constructor_args=(1,)).to_pydict()


def test_async_model_stays_async_when_bound(ds: bt.Dataset) -> None:
    """A sync wrapper around an async `__call__` silently coerces an un-awaited coroutine."""

    class AsyncScale:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        async def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch.set_column(0, "x", pc.multiply(batch.column("x"), self.factor))

    out = ds.map_batches(AsyncScale, fn_constructor_args=(4,)).to_pydict()
    assert out == {"x": [4, 8, 12]}


def test_sugar_forwards_the_full_option_set(ds: bt.Dataset) -> None:
    """`ds.map_batches` used to drop half of what `ds.ml.map_batches` accepts."""
    import inspect

    sugar = set(inspect.signature(bt.Dataset.map_batches).parameters)
    full = set(inspect.signature(type(ds.ml).map_batches).parameters)
    assert full - sugar == set()
