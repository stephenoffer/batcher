"""Rebatching must never drop a row.

Three sites took `[0]` off a `combine_chunks().to_batches()` list, which splits into
MULTIPLE batches at Arrow's 32-bit offset limit. Each one silently lost rows for exactly
the wide binary/string/list outputs large-scale inference produces, and one of them lost
the first batch of every TensorFlow training run regardless of width.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.ml.converters import to_tf_dataset
from batcher.ml.inference import _DynamicBatcher


def _batches(sizes: list[int]) -> list[pa.RecordBatch]:
    out, n = [], 0
    for size in sizes:
        out.append(pa.record_batch({"x": list(range(n, n + size))}))
        n += size
    return out


@pytest.mark.unit
def test_dynamic_batcher_conserves_every_row_across_push_and_flush():
    batcher = _DynamicBatcher(target=4)
    emitted: list[pa.RecordBatch] = []
    for batch in _batches([3, 5, 2, 7, 1]):
        emitted.extend(batcher.push(batch))
    emitted.extend(batcher.flush())

    assert sum(b.num_rows for b in emitted) == 18
    assert pa.Table.from_batches(emitted).column("x").to_pylist() == list(range(18))


@pytest.mark.unit
def test_dynamic_batcher_flush_returns_a_list_so_no_tail_batch_is_dropped():
    batcher = _DynamicBatcher(target=1000)
    for batch in _batches([2, 3]):
        assert batcher.push(batch) == []
    tail = batcher.flush()
    assert isinstance(tail, list)
    assert sum(b.num_rows for b in tail) == 5
    assert batcher.flush() == []  # buffer is drained, not replayed


@pytest.mark.unit
def test_dynamic_batcher_tracks_a_remainder_row_count_not_just_its_first_batch():
    """`_rows` drives the emit threshold; undercounting it stalls the batcher."""
    batcher = _DynamicBatcher(target=4)
    batcher.push(pa.record_batch({"x": list(range(6))}))  # emits 4, keeps 2
    assert batcher._rows == 2
    assert sum(b.num_rows for b in batcher.push(pa.record_batch({"x": [6, 7]}))) == 4


@pytest.mark.unit
def test_to_tf_dataset_does_not_swallow_the_first_batch():
    """The signature probe must not consume a batch out of a one-shot source."""
    tf = pytest.importorskip("tensorflow")
    assert tf is not None

    source = iter(_batches([2, 2, 2]))
    got = [d["x"].numpy().tolist() for d in to_tf_dataset(source)]

    assert got == [[0, 1], [2, 3], [4, 5]], "batch 0 was consumed by the signature probe"
