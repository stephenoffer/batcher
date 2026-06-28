"""Unit tests for the distributed-scan split prefetch (`scan_read._prefetch_split_reads`).

Prefetch reads up to `depth` splits ahead on a thread pool to overlap object-store I/O
with the map-side fold. It MUST be invisible to results: the same batches in the same
(file) order as a sequential read, however the reads interleave.
"""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

from batcher.dist.executors import scan_read as pio

pytestmark = pytest.mark.unit


class _FakeSplit:
    """A split whose `read` returns one batch carrying its id, after a small delay so
    concurrent reads visibly overlap (the prefetch path must still preserve order)."""

    def __init__(self, i: int, delay: float = 0.0) -> None:
        self._i = i
        self._delay = delay

    def read(self, _projection=None):
        if self._delay:
            time.sleep(self._delay)
        return [pa.record_batch({"i": pa.array([self._i], type=pa.int64())})]

    def schema(self) -> pa.Schema:
        return pa.schema([pa.field("i", pa.int64())])


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("n", [0, 1, 3, 10])
def test_prefetch_preserves_order_and_completeness(depth, n):
    splits = [_FakeSplit(i) for i in range(n)]
    got = [b.column("i")[0].as_py() for b in pio._prefetch_split_reads(splits, None, None, depth)]
    assert got == list(range(n))


def test_dataset_scan_returns_none_for_non_rowgroup_splits():
    # The fast pyarrow dataset path is only for Parquet RowGroupSplits; anything else
    # (here a fake split) must return None so the caller falls back to the prefetch pool.
    assert pio._dataset_scan_batches([_FakeSplit(0), _FakeSplit(1)], None, None) is None


def test_read_split_batches_falls_back_and_preserves_order():
    # `_read_split_batches` dispatches: non-row-group splits take the prefetch fallback and
    # must yield the same batches in order as the direct prefetch read.
    splits = [_FakeSplit(i) for i in range(5)]
    got = [b.column("i")[0].as_py() for b in pio._read_split_batches(splits, None, None)]
    assert got == list(range(5))


def test_prefetch_overlaps_io():
    # 8 splits, each a 100ms read. Sequential ~800ms; with depth 8 the reads overlap, so
    # the whole thing finishes in roughly one read's time. Assert it is clearly < serial.
    splits = [_FakeSplit(i, delay=0.1) for i in range(8)]
    t0 = time.perf_counter()
    out = list(pio._prefetch_split_reads(splits, None, None, 8))
    elapsed = time.perf_counter() - t0
    assert [b.column("i")[0].as_py() for b in out] == list(range(8))
    assert elapsed < 0.5, f"prefetch did not overlap I/O (took {elapsed:.2f}s, serial ~0.8s)"


def test_prefetch_propagates_read_errors():
    class _Boom(_FakeSplit):
        def read(self, _projection=None):
            raise ValueError("read failed")

    with pytest.raises(ValueError, match="read failed"):
        list(pio._prefetch_split_reads([_FakeSplit(0), _Boom(1)], None, None, 4))
