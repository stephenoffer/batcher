"""The native Parquet read batch is bounded by bytes, not only by rows.

A Parquet decode's working set is several times the batch it produces -- the physical column
buffer (a `uint8` tensor child is INT32 on disk, so 4x), the definition levels beside it, and
the cast to the Arrow type -- so the read batch size *is* the reader's residency. The flat
65,536-row ceiling is about 4 MB for an ordinary 64-byte row and **9.6 GB** for a decoded
224x224x3 image, which is how a GPU inference actor came to hold 24 GB on a 31 GB node and be
killed by the kernel rather than by anything the engine could see.

Measured on 4,000 such images (0.56 GB) read through `collect()`: **10.16 GB peak RSS in
5.82 s at 65,536 rows against 2.34 GB in 4.89 s at 64** -- 4.3x the memory and 1.2x the time,
for the same result and the same batch count out, because the engine re-morselizes either way.

The cap must never bind on an ordinary table, which is the half of this that a benchmark
regression would find first and these tests find now.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io.formats.structured._parquet_native import (
    NATIVE_READ_BATCH,
    NATIVE_READ_TARGET_BYTES,
    native_read_batch,
)

pytestmark = pytest.mark.unit

_PIX = 224 * 224 * 3
_IMAGES = pa.schema([("id", pa.int64()), ("img", pa.list_(pa.uint8(), _PIX))])
_NARROW = pa.schema([("a", pa.int64()), ("b", pa.float64()), ("c", pa.int32())])


def test_a_wide_row_is_capped_to_the_byte_target():
    rows = native_read_batch(_IMAGES)
    assert 1 <= rows < NATIVE_READ_BATCH
    assert rows * _PIX <= NATIVE_READ_TARGET_BYTES
    # And it is not capped so hard that the read degenerates to a row at a time.
    assert rows >= 64, "a 16 MiB budget holds ~111 images; a handful would be per-row I/O"


def test_an_ordinary_row_is_untouched():
    """The regression guard: TPC-H and every narrow table must read exactly as before."""
    assert native_read_batch(_NARROW) == NATIVE_READ_BATCH
    # 256 bytes/row is the break-even by construction (65,536 x 256 == 16 MiB).
    at_the_line = pa.schema([("f", pa.list_(pa.uint8(), 256))])
    assert native_read_batch(at_the_line) == NATIVE_READ_BATCH


def test_a_projection_is_sized_on_what_it_reads():
    """Projecting the narrow column out of a wide table lifts the cap, because it must.

    The image column is what makes the row wide; a query reading only `id` decodes none of
    it, and charging that read the tensor's width would cut its batch by three orders of
    magnitude for bytes it never touches.
    """
    assert native_read_batch(_IMAGES, ["id"]) == NATIVE_READ_BATCH
    assert native_read_batch(_IMAGES, ["img"]) < NATIVE_READ_BATCH


def test_the_cap_never_raises_the_callers_ceiling():
    """It is a cap. A caller asking for a small batch keeps it, whatever the row costs."""
    assert native_read_batch(_NARROW, None, ceiling=4096) == 4096
    assert native_read_batch(_IMAGES, None, ceiling=4096) < 4096
    assert native_read_batch(_IMAGES, None, ceiling=1) == 1


def test_an_unknown_schema_falls_back_to_the_row_ceiling():
    """No schema is a reason to keep the old behaviour, never to fail or guess small."""
    assert native_read_batch(None) == NATIVE_READ_BATCH
    assert native_read_batch(None, ["a"], ceiling=99) == 99


def test_an_empty_schema_does_not_divide_by_zero():
    assert native_read_batch(pa.schema([])) == NATIVE_READ_BATCH


def test_an_empty_projection_is_sized_conservatively_rather_than_as_nothing():
    """`projected_row_bytes` reads an empty projection as the whole schema, and that is fine.

    An empty column list is degenerate — a `count(*)` decodes no values — so the only thing
    that matters is that it cannot produce a *larger* batch than the honest width would. It
    does not: it produces the same one the full row does.
    """
    assert native_read_batch(_IMAGES, []) == native_read_batch(_IMAGES, None)
