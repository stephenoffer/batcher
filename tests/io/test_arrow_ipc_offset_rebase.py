"""Arrow IPC keeps variable-length columns intact when their offsets start mid-buffer.

`pyarrow.ipc` (reproduced on 19.0.1) serializes a string, binary or list array whose
offsets buffer does not begin at zero as garbage: the row count is right, no error is
raised, and the values come back as NUL bytes or invalid UTF-8. The array is valid Arrow,
so `validate(full=True)` passes and reading it in memory is correct -- the corruption
appears only after a round trip.

Batcher produces exactly that shape for the trailing partial batch of a `limit`, because
the batch is a window onto its morsel. `ds.head(50_000).write.arrow(path)` wrote 848
corrupt rows before `ArrowIPCSink` started rebasing offsets.

These tests pin both halves: the pyarrow behaviour that makes the workaround necessary,
so it can be removed when upstream fixes it, and the engine's round trip that must stay
clean regardless.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

import batcher as bt
from batcher.io.formats.structured.arrow_ipc import _offset_base, _rebase_offsets

WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]


def _offset_array(length: int, start: int) -> pa.Array:
    """A string array of `length` whose offsets buffer begins at element `start`."""
    parent = pa.array(WORDS * 4_000)
    _, offsets, values = parent.buffers()
    shifted = pa.py_buffer(bytes(memoryview(offsets)[start * 4 :]))
    return pa.StringArray.from_buffers(length, shifted, values, None, 0)


def test_pyarrow_still_corrupts_nonzero_offsets() -> None:
    """The upstream defect the sink works around. Delete the workaround when this fails."""
    array = _offset_array(848, 16_000)
    array.validate(full=True)  # the array itself is valid Arrow
    assert array.to_pylist()[:3] == WORDS[:3]  # and reads correctly in memory

    batch = pa.RecordBatch.from_arrays([array], names=["s"])
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    restored = ipc.open_stream(sink.getvalue()).read_all().column("s")

    # The row count survives, which is what makes this silent. The values do not: they
    # come back as NUL bytes here, and as invalid UTF-8 (a raising `to_pylist`) for other
    # offsets, so the assertion is that they differ rather than that they raise.
    assert len(restored) == len(array)
    try:
        corrupt = restored.to_pylist()
    except UnicodeDecodeError:
        return
    assert corrupt != array.to_pylist()


def test_rebase_normalizes_only_what_needs_it() -> None:
    """Offsets are moved to zero; a column already based there is passed through."""
    shifted = _offset_array(848, 16_000)
    assert _offset_base(shifted) > 0

    batch = pa.RecordBatch.from_arrays([shifted, pa.array(range(848))], names=["s", "n"])
    rebased = _rebase_offsets(batch)

    assert _offset_base(rebased.column("s")) == 0
    assert rebased.column("s").to_pylist() == shifted.to_pylist()
    assert rebased.column("n").to_pylist() == list(range(848))

    # A batch that needs nothing is returned unchanged, so the common path copies nothing.
    clean = pa.RecordBatch.from_arrays([pa.array(WORDS)], names=["s"])
    assert _rebase_offsets(clean) is clean


@pytest.mark.parametrize("rows", [49_153, 50_000, 65_537])
def test_limit_then_write_arrow_round_trips(tmp_path, rows: int) -> None:
    """A limited scan writes and reads back intact -- the case that was corrupt."""
    source = bt.from_pydict(
        {
            "key": list(range(200_000)),
            "word": [WORDS[index % len(WORDS)] for index in range(200_000)],
        }
    )
    staged = str(tmp_path / "source.parquet")
    source.write.parquet(staged)

    target = str(tmp_path / f"limited_{rows}.arrow")
    bt.read.parquet(staged).select("key", "word").limit(rows).write.arrow(target)

    restored = bt.read.arrow(target).to_pydict()
    assert len(restored["word"]) == rows
    assert set(restored["word"]) <= set(WORDS)


def test_full_scan_write_arrow_round_trips(tmp_path) -> None:
    """The unlimited case, which was already correct, stays correct."""
    source = bt.from_pydict({"word": [WORDS[index % len(WORDS)] for index in range(50_000)]})
    staged = str(tmp_path / "source.parquet")
    source.write.parquet(staged)

    target = str(tmp_path / "full.arrow")
    bt.read.parquet(staged).write.arrow(target)

    restored = bt.read.arrow(target).to_pydict()
    assert len(restored["word"]) == 50_000
    assert set(restored["word"]) <= set(WORDS)
