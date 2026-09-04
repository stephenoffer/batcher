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


# --------------------------------------------------------------------------- #
# The native row-group reader reads its per-file windows concurrently
# --------------------------------------------------------------------------- #
# `_native_scan_batches` groups a partition's splits by file and reads each file's
# row-groups in its own native call. Those calls used to run one after another, so a
# partition spread over many files paid a full object-store round trip per file in series
# — the shape the balanced split assignment actually produces. These pin the fix: the
# windows are read concurrently, in order, with a row-group-bounded look-ahead.


def _row_group_splits(n_files: int, per_file: int = 1):
    """`n_files x per_file` `RowGroupSplit`s over fake paths (the read itself is stubbed)."""
    from batcher.io.splits import RowGroupSplit

    return [
        RowGroupSplit(path=f"/fake/f{f}.parquet", row_groups=[g])
        for f in range(n_files)
        for g in range(per_file)
    ]


def _stub_native(monkeypatch, read):
    """Point the native row-group read at `read(uri, row_groups)`."""
    from batcher.io.formats.structured import _parquet_native

    monkeypatch.setattr(
        _parquet_native,
        "read_row_groups_filtered",
        lambda uri, rgs, cols, pred, rows: read(uri, rgs),
        raising=True,
    )


def _batch(uri: str, rgs: list[int]) -> list[pa.RecordBatch]:
    return [
        pa.record_batch({"i": pa.array([int(uri.split("f")[-1].split(".")[0]) * 100 + g])})
        for g in rgs
    ]


@pytest.mark.parametrize("per_file", [1, 3])
def test_native_windows_are_read_in_file_order(monkeypatch, per_file):
    # Concurrency must be invisible to the result: same batches, same order as a serial read.
    _stub_native(monkeypatch, _batch)
    splits = _row_group_splits(6, per_file)
    got = [b.column("i")[0].as_py() for b in pio._native_scan_batches(splits, None, None)]

    monkeypatch.setattr(pio, "_native_read_depth", lambda _units: 1)  # force serial
    serial = [b.column("i")[0].as_py() for b in pio._native_scan_batches(splits, None, None)]
    assert got == serial
    assert got == [f * 100 + g for f in range(6) for g in range(per_file)]


def test_native_windows_are_actually_concurrent(monkeypatch):
    """The positive control: this FAILS if the windows are read one after another.

    Every stubbed read parks on a `Barrier` sized to the file count, so the barrier can
    only clear if all six reads are in flight at once. A serial reader deadlocks on the
    first one and the barrier times out — which is the pre-fix behaviour, and the reason
    this test is here rather than a wall-clock comparison that a slow box could fake.
    """
    import threading

    barrier = threading.Barrier(6, timeout=10.0)

    def _read(uri, rgs):
        barrier.wait()  # BrokenBarrierError on timeout -> a serial reader fails here
        return _batch(uri, rgs)

    _stub_native(monkeypatch, _read)
    got = list(pio._native_scan_batches(_row_group_splits(6), None, None))
    assert len(got) == 6


def test_native_read_depth_keeps_the_row_group_budget():
    # The per-split reader holds at most `_SCAN_PREFETCH` splits (normally one row-group
    # each). Expressing the native path's look-ahead in row-groups keeps its resident
    # footprint there instead of multiplying it by the window size.
    narrow = [("u", [0])] * 40
    wide = [("u", list(range(pio._NATIVE_RG_WINDOW)))] * 40
    assert pio._native_read_depth(narrow) == pio._SCAN_PREFETCH
    assert pio._native_read_depth(wide) == max(1, pio._SCAN_PREFETCH // pio._NATIVE_RG_WINDOW)
    assert pio._native_read_depth([]) >= 1


def test_native_read_falls_back_when_the_native_call_declines(monkeypatch):
    # A `None` from the native reader means "unavailable" — the caller must get `None` and
    # fall back to pyarrow, not a half-stream.
    _stub_native(monkeypatch, lambda uri, rgs: None)
    assert pio._native_scan_batches(_row_group_splits(3), None, None) is None


def test_the_native_reader_is_on_by_default_and_still_switchable(monkeypatch):
    """The default is ON, and `BATCHER_NATIVE_READER=0` still turns it off.

    Pinned because the default was flipped on a measurement (serial per-file reads were what
    made this reader trail PyArrow under distributed load, not its HTTP client), and the
    previous default was set on a measurement of the same kind. If a later run says PyArrow
    is faster again, that is a decision to record here — not a line to quietly change.
    """
    import importlib

    try:
        monkeypatch.delenv("BATCHER_NATIVE_READER", raising=False)
        assert importlib.reload(pio)._NATIVE_READER is True
        monkeypatch.setenv("BATCHER_NATIVE_READER", "0")
        assert importlib.reload(pio)._NATIVE_READER is False
    finally:
        monkeypatch.delenv("BATCHER_NATIVE_READER", raising=False)
        importlib.reload(pio)
