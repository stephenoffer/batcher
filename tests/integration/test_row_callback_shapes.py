"""A per-row callback that returns the wrong shape is told so, by name.

`map` and `flat_map` are where a migrating user writes their first Python callback, and a
wrong return shape used to surface from inside pyarrow: ``AttributeError: 'int' object has
no attribute 'keys'`` for `map`, and — worse — ``'str' object has no attribute 'keys'`` for a
`flat_map` that returned a single dict, because iterating a dict yields its keys.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3]})


@pytest.mark.parametrize("bad", [lambda row: row["x"], lambda row: None, lambda row: [1]])
def test_map_rejects_a_non_dict_row(ds: bt.Dataset, bad) -> None:
    with pytest.raises(PlanError, match="per-row callback must return one"):
        ds.ml.map(bad).to_pydict()


def test_flat_map_rejects_a_single_dict(ds: bt.Dataset) -> None:
    """The mistake that used to blame a string for not having `.keys()`."""
    with pytest.raises(PlanError, match="returned a single dict"):
        ds.ml.flat_map(lambda row: {"y": 1}).to_pydict()


def test_flat_map_rejects_none(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="iterable of"):
        ds.ml.flat_map(lambda row: None).to_pydict()


def test_flat_map_rejects_non_dict_elements(ds: bt.Dataset) -> None:
    with pytest.raises(PlanError, match="list of int"):
        ds.ml.flat_map(lambda row: [1, 2]).to_pydict()


def test_the_message_names_the_callback(ds: bt.Dataset) -> None:
    def my_transform(row: dict) -> int:
        return row["x"]

    with pytest.raises(PlanError, match="my_transform"):
        ds.ml.map(my_transform).to_pydict()


def test_map_still_works(ds: bt.Dataset) -> None:
    assert ds.ml.map(lambda row: {"y": row["x"] * 2}).to_pydict() == {"y": [2, 4, 6]}


def test_flat_map_still_works(ds: bt.Dataset) -> None:
    out = ds.ml.flat_map(lambda row: [{"y": row["x"]}, {"y": row["x"] * 10}])
    assert out.to_pydict() == {"y": [1, 10, 2, 20, 3, 30]}


def test_flat_map_may_drop_a_row(ds: bt.Dataset) -> None:
    assert ds.ml.flat_map(lambda row: []).to_pydict() == {"x": []}


def test_flat_map_accepts_a_generator(ds: bt.Dataset) -> None:
    """A generator must not be consumed by the check, nor rejected by it."""
    out = ds.ml.flat_map(lambda row: ({"y": i} for i in range(2)))
    assert out.to_pydict() == {"y": [0, 1, 0, 1, 0, 1]}


def test_an_async_callback_is_checked_too(ds: bt.Dataset) -> None:
    async def bad(row: dict) -> int:
        return row["x"]

    with pytest.raises(PlanError, match="per-row callback"):
        ds.ml.map(bad).to_pydict()


@pytest.mark.parametrize("terminal", ["collect", "iter_batches"])
def test_the_check_reaches_the_streaming_path_too(ds: bt.Dataset, terminal: str) -> None:
    """`map` lowers to `map_batches`, which `iter_batches` runs through a different executor."""
    plan = ds.ml.map(lambda row: row["x"])
    with pytest.raises(PlanError, match="per-row callback"):
        plan.to_pydict() if terminal == "collect" else list(plan.iter_batches())


def test_an_empty_batch_is_not_checked() -> None:
    """There is no first row to inspect, and nothing to be wrong about."""
    empty = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 99)
    assert empty.ml.map(lambda row: {"y": row["x"]}).count() == 0
