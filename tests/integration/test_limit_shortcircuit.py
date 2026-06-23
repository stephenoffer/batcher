"""`limit(n)` / `head(n)` short-circuits the source read (O8.6).

Ray Data's `limit(n)` still processes the whole input. Batcher streams a plain
`Limit` over a breaker-free pipeline and stops reading once `n` rows are produced —
IO-bounded by `n + offset`, not the source size. Verified with a source that counts
how many batches it actually yields.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

from batcher import col
from batcher.api.dataset.frame import Dataset
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef


class _CountingSource:
    """An in-memory source that records how many batches were actually pulled."""

    bounded = True

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches
        self._schema = batches[0].schema
        self.batches_read = 0

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        self.batches_read = len(self._batches)  # a full read pulls everything
        return [b.select(projection) if projection is not None else b for b in self._batches]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for b in self._batches:
            self.batches_read += 1
            yield b.select(projection) if projection is not None else b

    def row_count(self) -> int | None:
        return sum(b.num_rows for b in self._batches)

    def identity(self) -> str:
        return "counting"

    def splits(self, target_size: int | None = None):
        return []


def _source(num_batches: int = 10, rows: int = 100) -> _CountingSource:
    batches = [
        pa.record_batch({"x": list(range(i * rows, i * rows + rows))}) for i in range(num_batches)
    ]
    return _CountingSource(batches)


def _ds(src: _CountingSource) -> Dataset:
    return Dataset(Scan(0, SchemaRef.from_arrow(src.schema())), [src])


def test_head_reads_only_the_needed_prefix():
    src = _source()  # 1000 rows across 10 batches
    out = _ds(src).head(50).collect()
    assert out.column("x").to_pylist() == list(range(50))
    assert src.batches_read == 1  # first 100-row batch covered 50


def test_limit_spanning_two_batches():
    src = _source()
    out = _ds(src).head(150).collect()
    assert out.column("x").to_pylist() == list(range(150))
    assert src.batches_read == 2


def test_limit_with_offset_skips_then_takes():
    src = _source()
    out = _ds(src).limit(10, offset=120).collect()
    assert out.column("x").to_pylist() == list(range(120, 130))
    assert src.batches_read == 2  # batch0 fully skipped, batch1 yields the slice


def test_full_collect_reads_everything_for_contrast():
    src = _source()
    out = _ds(src).collect()
    assert out.num_rows == 1000
    assert src.batches_read == 10


def test_limit_after_filter_is_correct_and_bounded():
    # filter keeps every row (x >= 0), so head(30) still equals the first 30.
    src = _source()
    out = _ds(src).filter(col("x") >= 0).head(30).collect()
    assert out.column("x").to_pylist() == list(range(30))
    assert src.batches_read == 1


def test_iter_batches_streams_limit():
    src = _source()
    batches = list(_ds(src).head(50).iter_batches())
    assert sum(b.num_rows for b in batches) == 50
    assert src.batches_read == 1
