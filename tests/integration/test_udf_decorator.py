"""The `@bt.udf` decorator: identity, reuse at a second scale, and direct callability.

A decorator that makes the decorated function unusable everywhere except through itself is a
trap, and an anonymous one makes every stage in a profile read as the same wrapper class.
These pin the three properties that keep a decorated UDF an ordinary Python function.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt


@bt.udf
def add_one(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Add one to ``x``."""
    return batch.set_column(0, "x", pc.add(batch.column("x"), 1))


@bt.udf(batch_format="pandas", output_columns=["x"])
def double(df):
    return df.assign(x=df["x"] * 2)


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3]})


def test_applies_to_a_dataset(ds: bt.Dataset) -> None:
    assert add_one(ds).to_pydict() == {"x": [2, 3, 4]}


def test_carries_the_wrapped_function_identity() -> None:
    """The profile names stages by qualname and the strategy probe caches by it."""
    assert add_one.__name__ == "add_one"
    assert add_one.__qualname__ == "add_one"
    assert add_one.__doc__ == "Add one to ``x``."
    assert add_one.__module__ == __name__


def test_repr_shows_the_function_and_its_options() -> None:
    assert "add_one" in repr(add_one)
    assert "batch_format='pandas'" in repr(double)


def test_runs_directly_on_a_batch() -> None:
    """A decorated fn stays unit-testable and composable inside another UDF."""
    batch = pa.record_batch({"x": [1, 2]})
    assert add_one(batch).column("x").to_pylist() == [2, 3]


def test_passes_to_map_batches_by_hand(ds: bt.Dataset) -> None:
    """Its own options are not applied this way, but it must not raise."""
    assert ds.map_batches(add_one).to_pydict() == {"x": [2, 3, 4]}


def test_options_overrides_without_mutating_the_original(ds: bt.Dataset) -> None:
    tuned = double.options(batch_size=1)
    assert tuned.config["batch_size"] == 1
    assert tuned.config["batch_format"] == "pandas"
    assert "batch_size" not in double.config
    assert tuned(ds).to_pydict() == {"x": [2, 4, 6]}


def test_per_row_udf_applies_and_runs_directly(ds: bt.Dataset) -> None:
    @bt.udf(per_row=True)
    def shout(row: dict) -> dict:
        return {"x": row["x"] * 10}

    assert shout(ds).to_pydict() == {"x": [10, 20, 30]}
    assert shout({"x": 4}) == {"x": 40}
