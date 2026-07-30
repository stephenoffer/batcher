"""No file source may hand the engine a batch bigger than a morsel.

A reader that parses a whole file into one Arrow chunk — numpy, XML, point clouds, several
SQL drivers — emitted that chunk as a *single* `RecordBatch` of however many rows the file
held. The engine's memory model assumes a batch is a morsel: the read-ahead budgets by
batch, every operator holds one, and a spill is measured in them. A 100M-row file arriving
as one batch defeats all three at once, and nothing failed — it simply used the memory.

The cut lives in `FileSource._normalize`, the one funnel both `read()` and `iter_batches()`
pass through, so it holds for every format rather than for the ones somebody remembered.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

from batcher.config import Config, config_context
from batcher.io.formats.ml.numpy import NumpySource


@pytest.fixture
def small_morsels():
    base = Config()
    cfg = base.replace(execution=dataclasses.replace(base.execution, morsel_rows=16))
    with config_context(cfg):
        yield


def _npy(tmp_path, rows: int) -> str:
    path = tmp_path / "a.npy"
    np.save(path, np.arange(rows, dtype=np.int64))
    return str(path)


def test_read_cuts_a_whole_file_chunk_into_morsels(tmp_path, small_morsels):
    sizes = [b.num_rows for b in NumpySource(_npy(tmp_path, 50)).read()]
    assert sizes == [16, 16, 16, 2]


def test_iter_batches_cuts_it_the_same_way(tmp_path, small_morsels):
    """`read()` and `iter_batches()` must agree, or a distributed read returns different
    batches from a single-node one."""
    src = NumpySource(_npy(tmp_path, 50))
    assert [b.num_rows for b in src.iter_batches()] == [b.num_rows for b in src.read()]


def test_the_rows_survive_the_cut_in_order(tmp_path, small_morsels):
    """A zero-copy slice is still a slice: every row exactly once, still in file order."""
    src = NumpySource(_npy(tmp_path, 50))
    values = [v for b in src.iter_batches() for v in b.column("data").to_pylist()]
    assert values == list(range(50))


def test_a_batch_already_within_the_morsel_is_passed_through_untouched(tmp_path, small_morsels):
    """The common case — a reader that already chunks — must not pay for the check."""
    src = NumpySource(_npy(tmp_path, 5))
    batches = src.read()
    assert [b.num_rows for b in batches] == [5]


def test_the_cut_follows_the_configured_morsel(tmp_path):
    """Not a constant: a deployment that raises `morsel_rows` gets bigger batches."""
    path = _npy(tmp_path, 50)
    base = Config()
    for rows, expected in ((8, [8] * 6 + [2]), (64, [50])):
        cfg = base.replace(execution=dataclasses.replace(base.execution, morsel_rows=rows))
        with config_context(cfg):
            assert [b.num_rows for b in NumpySource(path).read()] == expected


def test_slicing_never_swallows_a_zero_row_batch(tmp_path, small_morsels):
    """A reader that emits an empty batch does so to carry its types, and the cut must
    pass it through rather than treating "nothing to slice" as "nothing to yield"."""
    schema = pa.schema([pa.field("data", pa.int64())])
    empty = pa.RecordBatch.from_pylist([], schema=schema)
    src = NumpySource(_npy(tmp_path, 1))
    out = list(src._normalize([empty], None, "e.npy"))
    assert [b.num_rows for b in out] == [0]
    assert out[0].schema == schema


def test_an_empty_file_reads_to_no_rows(tmp_path, small_morsels):
    """And an empty `.npy` — whose one Arrow chunk is zero-length — still reads cleanly."""
    path = tmp_path / "e.npy"
    np.save(path, np.zeros(0, dtype=np.int64))
    src = NumpySource(str(path))
    assert sum(b.num_rows for b in src.read()) == 0
    assert src.schema() == pa.schema([pa.field("data", pa.int64())])
