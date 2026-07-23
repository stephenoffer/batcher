"""`io.base._readahead` — order, bounding, error propagation and shutdown.

The read-ahead is the only thing standing between `iter_batches` and a peak memory
proportional to file size, and every property it must hold is invisible in a result: a
broken bound still returns the right rows, and a leaked producer only shows up as a
process that aborts at exit. So they are pinned here as arithmetic over a fake reader,
with no file system and no format involved.
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

from batcher.io.base._readahead import ordered_readahead

pytestmark = pytest.mark.unit


def _batch(file_idx: int, seq: int, nrows: int = 8) -> pa.RecordBatch:
    return pa.record_batch({"f": [file_idx] * nrows, "i": [seq] * nrows})


def _reader(batches_per_file: int, *, on_yield=None):
    def iter_file(path: str):
        idx = int(path)
        for seq in range(batches_per_file):
            if on_yield is not None:
                on_yield(idx, seq)
            yield _batch(idx, seq)

    return iter_file


def test_yields_every_batch_in_file_order() -> None:
    files = [str(i) for i in range(7)]
    out = list(ordered_readahead(files, _reader(3), depth=4, max_bytes=1 << 20))

    assert len(out) == 21
    seen = [(b.column("f")[0].as_py(), b.column("i")[0].as_py()) for b in out]
    assert seen == [(f, s) for f in range(7) for s in range(3)]


def test_order_holds_when_later_files_are_much_faster() -> None:
    """File 0 being slow must not let file 3's batches overtake it."""

    def iter_file(path: str):
        idx = int(path)
        for seq in range(2):
            if idx == 0:
                time.sleep(0.05)
            yield _batch(idx, seq)

    out = list(ordered_readahead([str(i) for i in range(4)], iter_file, depth=4, max_bytes=1 << 20))
    assert [b.column("f")[0].as_py() for b in out] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_producer_blocks_rather_than_running_ahead_unboundedly() -> None:
    """The bound is on undelivered bytes: a consumer that stops pulling stalls producers.

    Without this, a producer decodes its whole file into the queue and the 'bound' is
    the file size — exactly the bug this module exists to prevent.
    """
    produced = 0
    lock = threading.Lock()

    def count(_idx: int, _seq: int) -> None:
        nonlocal produced
        with lock:
            produced += 1

    one = _batch(0, 0).nbytes
    # Budget for ~2 batches per file across 2 in-flight files.
    it = ordered_readahead(
        [str(i) for i in range(4)], _reader(50, on_yield=count), depth=2, max_bytes=one * 4
    )
    next(it)
    time.sleep(0.2)  # let producers run as far as they are allowed
    with lock:
        ran_ahead = produced
    it.close()

    assert ran_ahead < 50, f"producer decoded {ran_ahead} batches without backpressure"


def test_oversized_batch_is_still_delivered() -> None:
    """A single batch larger than the whole budget must not deadlock."""
    big = pa.record_batch({"v": list(range(10_000))})

    def iter_file(path: str):
        yield big

    out = list(ordered_readahead(["0"], iter_file, depth=1, max_bytes=8))
    assert len(out) == 1
    assert out[0].num_rows == 10_000


def test_producer_error_propagates_to_the_consumer() -> None:
    def iter_file(path: str):
        if path == "2":
            raise ValueError("corrupt file")
        yield _batch(int(path), 0)

    with pytest.raises(ValueError, match="corrupt file"):
        list(ordered_readahead([str(i) for i in range(4)], iter_file, depth=4, max_bytes=1 << 20))


def test_early_exit_joins_every_producer() -> None:
    """A consumer that stops early must leave no thread parked on credit or on a queue.

    A surviving producer holds Arrow buffers into interpreter shutdown, which aborts the
    process rather than raising — so the leak is only observable as a thread count.
    """
    before = threading.active_count()
    it = ordered_readahead(
        [str(i) for i in range(8)], _reader(200), depth=4, max_bytes=_batch(0, 0).nbytes * 2
    )
    next(it)
    it.close()

    deadline = time.time() + 5.0
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.02)
    assert threading.active_count() == before


def test_empty_and_single_file_inputs() -> None:
    assert list(ordered_readahead([], _reader(3), depth=4, max_bytes=1 << 20)) == []
    assert len(list(ordered_readahead(["0"], _reader(3), depth=4, max_bytes=1 << 20))) == 3
    # A file that yields nothing must not stall the files behind it.
    assert len(list(ordered_readahead(["0", "1"], _reader(0), depth=2, max_bytes=1 << 20))) == 0
