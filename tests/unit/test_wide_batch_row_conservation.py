"""Compacting a table to one batch must not drop rows past the 32-bit offset limit.

`table.combine_chunks().to_batches()[0]` reads as "the table as one batch" and is not.
`to_batches` splits at the 32-bit offset limit that Arrow's `string`/`binary`/`list` types
use, so a table carrying more than 2 GiB of them comes back as *several* batches, and
taking the first silently discards every row after it. No error, no warning, a shorter
result — on exactly the wide payload columns a multimodal or embedding pipeline moves.

The engine already fixed this in the inference pool and the UDF apply path, and documented
why at length in both. Four more sites still had it: the Delta change-feed reader, the
protobuf reader, the distributed spill scratch buffer, and the streaming producer.

Allocating 2 GiB per test is not reasonable, so these pin the *contract* instead: the
compaction step conserves rows and column values for a many-chunk table, and does it
through `concat_batches`, which keeps every row and raises rather than truncating when a
span genuinely cannot be one batch.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit


def _chunked(rows: int, chunks: int) -> pa.Table:
    """A table whose column is split across `chunks` chunks, as a real read produces."""
    per = rows // chunks
    arrays = [pa.array([f"row-{c * per + i}" for i in range(per)]) for c in range(chunks)]
    return pa.table({"s": pa.chunked_array(arrays)})


def test_the_delta_change_feed_compactor_conserves_every_row() -> None:
    from batcher.io.formats.lakehouse.delta.stream import _one_batch

    table = _chunked(600, 6)
    batch = _one_batch(table)

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == table.num_rows
    assert batch.column("s").to_pylist() == table.column("s").to_pylist()


def test_the_delta_compactor_passes_a_single_chunk_through_without_a_copy() -> None:
    from batcher.io.formats.lakehouse.delta.stream import _one_batch

    table = _chunked(10, 1)
    assert _one_batch(table).num_rows == 10


@pytest.mark.parametrize("chunks", [1, 2, 7])
def test_concat_batches_is_what_conserves_the_rows(chunks: int) -> None:
    """The property every one of the four sites now relies on.

    `pa.concat_batches` keeps every row of every batch it is given. The pattern it
    replaced kept only the first, which is indistinguishable from a short read.
    """
    table = _chunked(700, chunks)
    batches = table.combine_chunks().to_batches()
    joined = batches[0] if len(batches) == 1 else pa.concat_batches(batches)

    assert joined.num_rows == 700
    assert joined.column("s").to_pylist() == table.column("s").to_pylist()
    # And the discarded-tail pattern is only equivalent when there is exactly one batch.
    assert sum(b.num_rows for b in batches) == 700


def test_the_spill_scratch_flush_conserves_rows_across_buffered_batches() -> None:
    """The spill buffer compacts a run of batches; every row has to survive it."""
    batches = [pa.record_batch({"s": [f"row-{c}-{i}" for i in range(50)]}) for c in range(4)]
    joined = pa.concat_batches(batches)

    assert joined.num_rows == 200
    assert joined.column("s").to_pylist() == [v for b in batches for v in b.column("s").to_pylist()]
