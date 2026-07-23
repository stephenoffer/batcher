"""The out-of-core input tap normalizes source batches to a byte target.

`_iter_spill_morsels` is the single input every spill executor (aggregate / join /
sort / window) reads through. Its contract: split an over-large batch into pieces
within the target, coalesce a run of small batches up to it, and never drop or
reorder a row. Normalizing the batch the partition phase feeds the engine at once
keeps the parallel partial's per-thread hash tables — and therefore peak memory —
bounded by a morsel (the split half), *and* keeps every chunk wide enough to fan
across all cores so throughput doesn't collapse on a fine-grained source (the
coalesce half) — regardless of how the source happened to chunk its output.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.spill import _SPILL_INPUT_CHUNK_BYTES, _iter_spill_morsels

pytestmark = pytest.mark.unit


class _FakeSource:
    """Minimal `Source` stand-in: replays a fixed list of batches."""

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def iter_batches(self, projection: list[str] | None = None):
        del projection  # replayed verbatim; projection is applied by the map sub-plan
        return iter(self._batches)


def _wide_batch(rows: int) -> pa.RecordBatch:
    # Two int64 columns => 16 bytes/row, so a modest row count clears the ceiling.
    return pa.record_batch({"k": list(range(rows)), "v": list(range(rows))})


def test_small_batch_passes_through_unchanged() -> None:
    b = _wide_batch(1000)  # ~16 KiB, well under the ceiling
    out = list(_iter_spill_morsels(_FakeSource([b])))
    assert len(out) == 1
    assert out[0] is b  # forwarded verbatim, not copied


def test_empty_batches_are_dropped() -> None:
    empty = _wide_batch(0)
    b = _wide_batch(10)
    out = list(_iter_spill_morsels(_FakeSource([empty, b])))
    assert [o.num_rows for o in out] == [10]


def test_oversized_batch_is_split_within_the_ceiling() -> None:
    rows = (_SPILL_INPUT_CHUNK_BYTES // 16) * 4 + 123  # ~4 ceilings' worth + remainder
    big = _wide_batch(rows)
    assert big.nbytes > _SPILL_INPUT_CHUNK_BYTES
    out = list(_iter_spill_morsels(_FakeSource([big])))
    assert len(out) > 1
    # Every emitted chunk is within the byte ceiling (the bounded-memory guarantee).
    assert all(o.nbytes <= _SPILL_INPUT_CHUNK_BYTES for o in out)


def test_split_preserves_every_row_in_order() -> None:
    rows = (_SPILL_INPUT_CHUNK_BYTES // 16) * 2 + 7
    big = _wide_batch(rows)
    out = list(_iter_spill_morsels(_FakeSource([big])))
    assert sum(o.num_rows for o in out) == rows
    stitched = pa.Table.from_batches(out).column("k").to_pylist()
    assert stitched == list(range(rows))  # no row dropped, reordered, or duplicated


def test_split_pieces_are_zero_copy_views() -> None:
    # A slice shares the parent's buffers, so re-morselizing an over-large batch adds
    # no bulk copy — only the small downstream engine outputs are fresh allocations.
    rows = (_SPILL_INPUT_CHUNK_BYTES // 16) * 2
    big = _wide_batch(rows)
    out = list(_iter_spill_morsels(_FakeSource([big])))
    parent_buf = big.column("k").buffers()[1].address
    child_buf = out[0].column("k").buffers()[1].address
    assert parent_buf == child_buf  # same underlying buffer, offset view


def test_many_tiny_batches_are_coalesced() -> None:
    # A source that emits thousands of tiny batches must not become thousands of
    # engine dispatches: the run is coalesced into a handful of full-size chunks. This
    # is the fix for the ~30x out-of-core slowdown on fine-grained streaming sources.
    tiny = [_wide_batch(64) for _ in range(4000)]  # 64 rows * 16 B = 1 KiB each
    out = list(_iter_spill_morsels(_FakeSource(tiny)))
    assert len(out) < len(tiny) // 10  # coalesced to far fewer, larger chunks
    assert sum(o.num_rows for o in out) == 4000 * 64  # no row lost
    # No coalesced chunk runs away past the target (bar the trailing remainder).
    assert all(o.nbytes <= _SPILL_INPUT_CHUNK_BYTES * 2 for o in out)


def test_coalescing_preserves_row_order_across_mixed_sizes() -> None:
    # Interleave small runs with an over-large batch: the small run must flush before
    # the split of the large one, so the global row order is preserved end to end.
    per_small = 100
    small_a = pa.record_batch({"k": list(range(per_small)), "v": list(range(per_small))})
    big_start = per_small
    big_rows = (_SPILL_INPUT_CHUNK_BYTES // 16) + 500
    big = pa.record_batch(
        {
            "k": list(range(big_start, big_start + big_rows)),
            "v": list(range(big_start, big_start + big_rows)),
        }
    )
    tail_start = big_start + big_rows
    small_b = pa.record_batch(
        {
            "k": list(range(tail_start, tail_start + per_small)),
            "v": list(range(tail_start, tail_start + per_small)),
        }
    )
    out = list(_iter_spill_morsels(_FakeSource([small_a, big, small_b])))
    stitched = pa.Table.from_batches(out).column("k").to_pylist()
    assert stitched == list(range(tail_start + per_small))  # contiguous, in order
