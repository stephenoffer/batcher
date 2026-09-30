"""`from_huggingface` over a streaming `IterableDataset` and over a map-style *view*.

`datasets` is not a test dependency, so the two HuggingFace shapes are stood in for by the
minimum the adapter touches: a map-style dataset exposes ``.data.table`` and an ``_indices``
mapping, an iterable one exposes neither and is read through ``with_format('arrow').iter``.
A fake ``datasets`` module satisfies the optional-dependency guard. The real library is
covered by `tests/io/test_interop.py::test_from_huggingface_roundtrip` where installed.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.interop import from_huggingface


@pytest.fixture(autouse=True)
def _fake_datasets(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))


class _ArrowView:
    def __init__(self, table: pa.Table, pulls: list[int]) -> None:
        self._table = table
        self._pulls = pulls

    def iter(self, batch_size: int):
        for start in range(0, self._table.num_rows, batch_size):
            self._pulls.append(start)
            yield self._table.slice(start, batch_size)


class _FakeIterable:
    """An `IterableDataset`: no ``.data``, re-streamable, counts every batch pulled."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table
        self.pulls: list[int] = []

    def with_format(self, fmt: str) -> _ArrowView:
        assert fmt == "arrow"
        return _ArrowView(self._table, self.pulls)


class _FakeView:
    """A map-style dataset after ``select``: the parent's full table plus an indices map."""

    def __init__(self, parent: pa.Table, rows: list[int]) -> None:
        self.data = types.SimpleNamespace(table=parent)
        self._indices = pa.table({"indices": rows})
        self._selected = parent.take(rows)

    def with_format(self, fmt: str) -> _ArrowView:
        return _ArrowView(self._selected, [])


def test_a_view_reads_its_own_rows_not_its_parents():
    parent = pa.table({"x": list(range(10))})
    view = _FakeView(parent, [7, 2, 5])
    got = bt.from_arrow(from_huggingface(view).read()).to_pydict()
    # Taking `.data.table` returned all ten parent rows.
    assert got == {"x": [7, 2, 5]}


def test_an_iterable_dataset_streams_instead_of_materializing():
    table = pa.table({"x": list(range(40_000))})
    stream = _FakeIterable(table)
    ds = bt.from_huggingface(stream)
    # Building the Dataset pulls one batch to read the schema. Draining the stream into a
    # table here pulled all three 16,384-row batches.
    assert len(stream.pulls) == 1
    iterator = ds.iter_batches()
    first = next(iterator)
    assert first.num_rows > 0
    assert len(stream.pulls) < 1 + 3
    rest = sum(b.num_rows for b in iterator)
    assert first.num_rows + rest == 40_000


def test_an_iterable_dataset_can_be_read_twice():
    stream = _FakeIterable(pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]}))
    ds = bt.from_huggingface(stream)
    assert ds.count() == 3
    assert sorted(ds.to_pydict()["x"]) == [1, 2, 3]
