"""No file source may hand the engine a batch bigger than `execution.read_batch_bytes`.

A reader that parses a whole file into one Arrow chunk — numpy, XML, point clouds, several
SQL drivers — emitted that chunk as a *single* `RecordBatch` of however many rows the file
held. The engine's memory model budgets by batch: the read-ahead counts them, every operator
holds one, and a spill is measured in them. A 100M-row file arriving as one batch defeats all
three at once, and nothing failed — it simply used the memory.

**The bound is bytes, not the morsel row count, and the difference is load-bearing.** The
engine re-morselizes whatever it is handed (`morsel_rows`/`morsel_bytes`, zero-copy), so
cutting to the morsel size here reduces no downstream work and costs one Arrow C Data
Interface import per extra batch — measured on TPC-H sf10 `lineitem` at 94.5 ms across 3,907
morsel-sized batches against 46.5 ms across 980 65,536-row ones, for identical work. The
morsel row count remains the *floor*, so a batch is never cut below the engine's unit.

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

# 16 rows of `int64` — so the cut lands at 16 rows and every expectation below is the
# *byte* bound biting, not the morsel floor.
_SIXTEEN_INT64_ROWS = 16 * 8


@pytest.fixture
def small_morsels():
    base = Config()
    cfg = base.replace(
        execution=dataclasses.replace(
            base.execution, morsel_rows=1, read_batch_bytes=_SIXTEEN_INT64_ROWS
        )
    )
    with config_context(cfg):
        yield


def _npy(tmp_path, rows: int) -> str:
    path = tmp_path / "a.npy"
    np.save(path, np.arange(rows, dtype=np.int64))
    return str(path)


def test_read_cuts_a_whole_file_chunk_down(tmp_path, small_morsels):
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


def test_the_cut_follows_the_configured_byte_budget(tmp_path):
    """Not a constant: a deployment that raises `read_batch_bytes` gets bigger batches."""
    path = _npy(tmp_path, 50)
    base = Config()
    for byte_budget, expected in ((8 * 8, [8] * 6 + [2]), (64 * 8, [50])):
        cfg = base.replace(
            execution=dataclasses.replace(
                base.execution, morsel_rows=1, read_batch_bytes=byte_budget
            )
        )
        with config_context(cfg):
            assert [b.num_rows for b in NumpySource(path).read()] == expected


def test_the_morsel_row_count_is_a_floor_the_byte_budget_cannot_cross(tmp_path):
    """A byte budget smaller than one morsel still yields whole morsels.

    The engine schedules in morsels, so cutting below one would hand it batches it can only
    pad back out — and a misconfigured budget must not be able to shred a read into
    single-row batches.
    """
    path = _npy(tmp_path, 50)
    base = Config()
    cfg = base.replace(
        execution=dataclasses.replace(base.execution, morsel_rows=20, read_batch_bytes=8)
    )
    with config_context(cfg):
        assert [b.num_rows for b in NumpySource(path).read()] == [20, 20, 10]


def test_a_reader_that_already_chunks_is_not_cut_finer_than_it_chunked(tmp_path):
    """The regression this bound exists for: batches within budget pass through whole.

    A reader returning 65,536-row batches used to have each one sliced into four
    morsel-sized ones, quadrupling the `RecordBatch` count crossing the FFI for no change
    downstream. At the default budget a narrow batch of that size is well inside it.
    """
    src = NumpySource(_npy(tmp_path, 50))
    chunked = [pa.record_batch({"data": pa.array(range(25), type=pa.int64())})] * 2
    out = list(src._normalize(chunked, None, "a.npy"))
    assert [b.num_rows for b in out] == [25, 25]


def test_a_wide_string_batch_is_cut_by_its_real_size_not_an_estimated_one(tmp_path):
    """The bound reads `RecordBatch.nbytes`, not a per-row width guessed from the schema.

    A variable-length column contributes a 32-byte default to `schema_row_bytes`, so a
    kilobyte-string column estimates ~30x narrower than it is — and a bound computed that way
    would pass through a batch 30x over budget, which is the opposite of what a memory cap is
    for. This holds a batch of 1 KiB strings to a 64 KiB budget and checks the pieces really
    are that size.
    """
    path = tmp_path / "wide.npy"
    np.save(path, np.array(["x" * 1024] * 512))
    src = NumpySource(str(path))
    wide = pa.record_batch({"data": pa.array(["x" * 1024] * 512)})
    budget = 64 * 1024
    base = Config()
    cfg = base.replace(
        execution=dataclasses.replace(base.execution, morsel_rows=1, read_batch_bytes=budget)
    )
    with config_context(cfg):
        out = list(src._normalize([wide], None, "w.npy"))
    assert sum(b.num_rows for b in out) == 512
    assert len(out) > 1, "a batch 8x over budget must be cut"
    assert max(b.nbytes for b in out) <= budget * 2, "each piece stays near the budget"


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
