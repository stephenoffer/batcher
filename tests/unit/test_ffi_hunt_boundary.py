"""FFI boundary regression tests for the `bc-py` zero-copy edge (wave-9 bug hunt).

Two classes of defect pinned here:

1. **UInt64 narrow-type normalization must not silently corrupt data.** The boundary
   widens narrow numerics to Int64/Float64; every recorded widening is lossless except
   ``UInt64 -> Int64`` (a value above ``i64::MAX`` has no Int64 form). Arrow's *safe*
   cast turned such a value into a **null**, silently replacing real data. The engine now
   refuses the batch with a clear error instead of handing back a corrupted column. A
   ``UInt64`` value that *fits* in Int64 must still round-trip exactly.

2. **The shuffle/bloom FFI partitioners must not panic on bad indices.** An out-of-range
   key index or a zero partition count used to index a column out of bounds deep in the
   runtime and *panic through PyO3* — a ``PanicException`` (a ``BaseException`` that a
   caller's ``except Exception`` misses, and a process-abort risk). The boundary now
   validates these and raises a clean, catchable ``Exception``.

These exercise ``batcher._native`` directly (the FFI surface), so they require the engine
to be built with the wave-9 fix; before the fix each ``raises`` assertion fails (the call
either returns corrupted data or raises a ``BaseException``).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.native import engine

pytestmark = pytest.mark.unit

_native = engine()

I64_MAX = 2**63 - 1


def _rb():
    return pa.record_batch({"k": pa.array([1, 2, 1]), "v": pa.array([10, 20, 30])})


# --- 1. UInt64 normalization: no silent corruption --------------------------------


def test_uint64_above_i64max_is_not_silently_nulled():
    """A UInt64 above i64::MAX must not cross the boundary as a silent null."""
    big = 2**63 + 5  # > i64::MAX, unrepresentable in Int64
    tbl = pa.table({"u": pa.array([1, big, 3], type=pa.uint64())})
    with pytest.raises(Exception) as exc:
        bt.from_arrow(tbl).collect()
    # The message must name the offending column/type, not fail opaquely.
    assert "u" in str(exc.value) or "Int64" in str(exc.value)


def test_uint64_within_i64max_roundtrips_exactly():
    """A UInt64 that fits in Int64 still widens losslessly (no regression)."""
    vals = [0, 1, 2**62, I64_MAX]
    tbl = pa.table({"u": pa.array(vals, type=pa.uint64())})
    out = bt.from_arrow(tbl).collect()
    assert out.column("u").to_pylist() == vals


def test_uint32_max_roundtrips_exactly():
    """UInt32's max is far below i64::MAX — always lossless."""
    tbl = pa.table({"u": pa.array([4294967295], type=pa.uint32())})
    out = bt.from_arrow(tbl).collect()
    assert out.column("u").to_pylist() == [4294967295]


# --- 2. Partitioner / bloom FFI: clean error, never a panic -----------------------


def test_partition_batches_bad_index_raises_clean_exception():
    with pytest.raises(Exception) as exc:
        _native.partition_batches([_rb()], [99], 4)
    assert isinstance(exc.value, Exception)  # not a bare BaseException/PanicException


def test_partition_batches_zero_partitions_raises_clean_exception():
    with pytest.raises(Exception) as exc:
        _native.partition_batches([_rb()], [0], 0)
    assert isinstance(exc.value, Exception)


def test_range_partition_batches_bad_index_raises_clean_exception():
    with pytest.raises(Exception) as exc:
        _native.range_partition_batches([_rb()], 99, [1.0], 2, True, False)
    assert isinstance(exc.value, Exception)


def test_salted_partition_batches_bad_index_raises_clean_exception():
    with pytest.raises(Exception) as exc:
        _native.salted_partition_batches([_rb()], [99], 4, [], 2, False)
    assert isinstance(exc.value, Exception)


def test_build_key_bloom_bad_index_raises_clean_exception():
    with pytest.raises(Exception) as exc:
        _native.build_key_bloom([_rb()], [99], 100)
    assert isinstance(exc.value, Exception)


def test_partition_batches_valid_index_still_works():
    """A valid key index partitions without error (no regression)."""
    parts = _native.partition_batches([_rb()], [0], 4)
    assert len(parts) == 4
