"""`_rechunk` coalesces/splits a UDF's input batches to a coarse target — losslessly.

The morsel-sized native scan that feeds a per-batch Python `fn` produces hundreds of
small batches; running the `fn` once per morsel makes the fixed per-call overhead (FFI +
framework conversion + schema build) dominate. `_rechunk` merges them up to a coarse
target so every core gets a full batch. pyarrow's ``Table.to_batches(max_chunksize=n)``
only *splits* — it silently fails to *merge* — so this is the piece that makes coarsening
actually happen. It MUST be a relation-level no-op: same rows, same order.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.udf.apply import rechunk as _rechunk

pytestmark = pytest.mark.unit


def _batches(sizes: list[int]) -> list[pa.RecordBatch]:
    """One batch per size, values globally sequential so order is verifiable."""
    out, start = [], 0
    for n in sizes:
        out.append(pa.record_batch({"v": pa.array(range(start, start + n))}))
        start += n
    return out


def _flatten(batches: list[pa.RecordBatch]) -> list[int]:
    if not batches:
        return []
    return pa.Table.from_batches(batches).column("v").to_pylist()


def test_merges_small_batches_up_to_target():
    # 20 batches of 6_900 rows (the morsel-fed shape) must coalesce toward the target,
    # not stay 20 tiny batches (the bug `to_batches(max_chunksize=)` leaves behind).
    src = _batches([6_900] * 20)
    out = _rechunk(src, 62_500)
    assert len(out) < len(src)
    assert max(b.num_rows for b in out) >= 62_500  # actually coarsened
    assert _flatten(out) == list(range(6_900 * 20))  # rows + order preserved


def test_splits_a_single_oversized_batch():
    # A single 400-row batch with target 50 must become 8 batches of 50 (the explicit
    # `batch_size` case), so the per-batch call fans out across workers.
    (src,) = _batches([400])
    out = _rechunk([src], 50)
    assert [b.num_rows for b in out] == [50] * 8
    assert _flatten(out) == list(range(400))


def test_mixed_sizes_preserve_rows_and_order():
    src = _batches([10, 200, 5, 5, 300, 1])
    out = _rechunk(src, 100)
    assert _flatten(out) == list(range(sum([10, 200, 5, 5, 300, 1])))
    # every output batch is bounded by ~target (a merged run can reach <2*target)
    assert all(b.num_rows <= 2 * 100 for b in out)


def test_noop_when_already_one_bounded_batch():
    src = _batches([100])
    out = _rechunk(src, 128)
    assert out is src  # a single already-bounded batch is returned untouched (no copy)


@pytest.mark.parametrize("target", [0, -1])
def test_nonpositive_target_is_passthrough(target):
    src = _batches([10, 20])
    assert _rechunk(src, target) is src


def test_empty_and_zero_row_inputs():
    assert _rechunk([], 100) == []
    z = pa.record_batch({"v": pa.array([], type=pa.int64())})
    assert _flatten(_rechunk([z, z], 100)) == []
