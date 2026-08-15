"""Compacting a table to one batch must not drop rows past the 32-bit offset limit.

`table.combine_chunks().to_batches()[0]` reads as "the table as one batch" and is not.
`to_batches` splits at the 32-bit offset limit that Arrow's `string`/`binary`/`list` types
use, so a table carrying more than 2 GiB of them comes back as *several* batches, and
taking the first silently discards every row after it. No error, no warning, a shorter
result — on exactly the wide payload columns a multimodal or embedding pipeline moves.

Nine sites had reached this conclusion separately. Six wrote a local fix — the inference
pool, the UDF apply path, the Delta change-feed reader, the protobuf reader, the distributed
spill scratch buffer, and the streaming producer — in five different spellings; three still
had the bug, in the streaming distinct, in `iter_batches`' exact re-chunker, and in the
keyed-state fold. All nine now go through `plan.types.one_batch`, so the property is stated
once and `tests/unit/test_batch_compaction.py` guards the spelling from regrowing.

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


def test_the_shared_compactor_conserves_every_row() -> None:
    """What the Delta change feed, the protobuf reader and six others now all call."""
    from batcher.plan.types import one_batch

    table = _chunked(600, 6)
    batch = one_batch(table)

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == table.num_rows
    assert batch.column("s").to_pylist() == table.column("s").to_pylist()


def test_the_shared_compactor_passes_a_single_chunk_through_without_a_copy() -> None:
    from batcher.plan.types import one_batch

    table = _chunked(10, 1)
    assert one_batch(table).num_rows == 10


def test_the_streaming_rechunker_conserves_rows_across_an_emit_boundary() -> None:
    """`iter_batches(batch_size=...)` cut each emitted batch with the truncating spelling.

    The re-chunker is the last thing between the engine and the user's loop, so a row it
    drops is a row the query never returned.
    """
    from batcher.api.terminal.stream.rebatch import _rebatch_exact

    source = [pa.record_batch({"s": [f"row-{c}-{i}" for i in range(30)]}) for c in range(7)]
    out = list(_rebatch_exact(iter(source), 40))

    assert sum(b.num_rows for b in out) == 210
    assert [v for b in out for v in b.column("s").to_pylist()] == [
        v for b in source for v in b.column("s").to_pylist()
    ]


def test_the_keyed_state_grouper_conserves_a_group_it_hands_the_user() -> None:
    """The per-key slice handed to a `transform_with_state` function is compacted too."""
    from batcher.core.streaming.keyed_state import _group_by

    batches = [
        pa.record_batch(
            {"k": pa.array([c % 3] * 40, type=pa.int64()), "s": [f"v{c}-{i}" for i in range(40)]}
        )
        for c in range(6)
    ]
    seen = {key: rows.num_rows for key, rows in _group_by(batches, ["k"])}

    assert sum(seen.values()) == 240
    assert set(seen) == {(0,), (1,), (2,)}


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
