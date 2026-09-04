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


#: The oldest pyarrow `pyproject.toml` declares support for. The workaround below is
#: required for as long as *any* version in that range corrupts the round trip, so this is
#: what decides whether it can go -- not whichever version happens to be installed here.
SUPPORTED_FLOOR = (16,)


def test_the_workaround_is_still_required_at_the_supported_floor() -> None:
    """Why `_rebase_offsets` still exists, stated as the two facts it depends on.

    The defect was reproduced on pyarrow 19.0.1 and is fixed by 23.0.1, so on a current
    install the round trip below comes back clean. That does **not** retire the workaround:
    `pyproject.toml` declares `pyarrow>=16`, and a user on 19 gets NUL bytes with the right
    row count and no error. The sink cannot ask which defect its reader has.

    So this test asserts what is actually true on the installed version, and fails when the
    *floor* moves past the fix -- which is the event that makes the workaround dead code.
    """
    assert SUPPORTED_FLOOR < (20,), (
        "the declared pyarrow floor has passed the release that fixed nonzero-offset IPC; "
        "delete `_offset_base`/`_rebase_offsets` and their calls in `arrow_ipc.py`"
    )

    array = _offset_array(848, 16_000)
    array.validate(full=True)  # the array itself is valid Arrow
    assert array.to_pylist()[:3] == WORDS[:3]  # and reads correctly in memory

    batch = pa.RecordBatch.from_arrays([array], names=["s"])
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    restored = ipc.open_stream(sink.getvalue()).read_all().column("s")

    # The row count survives either way, which is what made the old behaviour silent.
    assert len(restored) == len(array)
    try:
        round_tripped = restored.to_pylist()
    except UnicodeDecodeError:
        return  # an unfixed pyarrow, corrupting loudly enough to raise
    if tuple(int(part) for part in pa.__version__.split(".")[:1]) >= (20,):
        assert round_tripped == array.to_pylist(), (
            "this pyarrow was expected to have the fix; if it does not, the version "
            "boundary in this test is wrong"
        )
    else:
        assert round_tripped != array.to_pylist(), "the defect this workaround exists for"


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
