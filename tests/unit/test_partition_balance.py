"""In-memory source partitioning fans out evenly across workers (no Ray needed).

`partition_descriptors` must split a non-splittable (in-memory / iterator) source into
per-worker partitions of near-equal ROW COUNT, so a source that arrives as one large
batch — a `from_arrow` table, an image/tensor set — is spread across every worker
instead of landing whole on worker 0 (which capped every in-memory distributed pipeline,
GPU or relational, to a single worker). The slicing is row-conserving and order-
preserving, so it is result-invariant for any downstream operator.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from batcher.dist.executors.partition_io import (
    _slice_rows_evenly,
    descriptor_rows,
    partition_descriptors,
)


def _batch(lo: int, hi: int) -> pa.RecordBatch:
    return pa.record_batch({"x": pa.array(np.arange(lo, hi, dtype=np.int64))})


def _rows(groups: list[list[pa.RecordBatch]]) -> list[int]:
    return [sum(b.num_rows for b in g) for g in groups]


def _values(groups: list[list[pa.RecordBatch]]) -> list[int]:
    return [v for g in groups for b in g for v in b.column("x").to_pylist()]


def test_single_large_batch_spreads_evenly():
    # The bug: one 2048-row batch used to land entirely on worker 0.
    g = _slice_rows_evenly([_batch(0, 2048)], 8)
    assert _rows(g) == [256] * 8
    assert _values(g) == list(range(2048))  # order preserved, every row present


def test_uneven_total_distributes_remainder_to_leading_workers():
    g = _slice_rows_evenly([_batch(0, 2050)], 8)
    assert _rows(g) == [257, 257, 256, 256, 256, 256, 256, 256]
    assert sum(_rows(g)) == 2050


def test_fewer_batches_than_workers_still_balances():
    g = _slice_rows_evenly([_batch(0, 100), _batch(100, 300)], 8)
    assert _rows(g) == [38, 38, 38, 38, 37, 37, 37, 37]
    assert _values(g) == list(range(300))


def test_more_rows_conserved_across_many_batches():
    batches = [_batch(i * 97, i * 97 + 97) for i in range(11)]  # 11 batches of 97
    total = 11 * 97
    g = _slice_rows_evenly(batches, 8)
    assert sum(_rows(g)) == total
    assert max(_rows(g)) - min(_rows(g)) <= 1  # near-equal
    assert _values(g) == list(range(total))  # order preserved


def test_empty_source_yields_empty_groups():
    g = _slice_rows_evenly([pa.record_batch({"x": pa.array([], pa.int64())})], 4)
    assert _rows(g) == [0, 0, 0, 0]


def test_single_worker_gets_everything():
    g = _slice_rows_evenly([_batch(0, 50), _batch(50, 90)], 1)
    assert _rows(g) == [90]


def test_partition_descriptors_balances_in_memory_source():
    from batcher.io.source import InMemorySource

    # A single 1600-row batch through the public descriptor API → 8 balanced partitions.
    src = InMemorySource([_batch(0, 1600)])
    descs = partition_descriptors(src, 8)
    assert [descriptor_rows(d) for d in descs] == [200] * 8
