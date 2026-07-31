"""Where the documents inside a packed sequence begin and end.

Packing puts unrelated documents in one sequence, and plain causal attention then lets every
token attend across the seams. The cost is invisible in the data and shows up in the model, so
the trainer has to be told where the seams are — `cu_seqlens` for FlashAttention's
variable-length path, or a position-id reset. The packed column is exactly as wide either way,
so if the boundaries are not emitted alongside it they are unrecoverable.

The invariant every test here rests on: the segment lengths of one row sum to `seq_len`. A row
whose segments sum to less has lost a seam, and one that sums to more has invented a token.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.ml.llm.packing import pack_sequences

pytestmark = pytest.mark.unit


def _packed(batches, **kwargs) -> list[tuple[list, list]]:
    rows: list[tuple[list, list]] = []
    for out in pack_sequences(batches, boundaries_column="seq_lens", **kwargs):
        tokens = out.column("tokens").to_pylist()
        seq_lens = out.column("seq_lens").to_pylist()
        rows.extend(zip(tokens, seq_lens, strict=True))
    return rows


def test_the_segments_of_a_sequence_sum_to_its_width() -> None:
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    rows = _packed([batch], seq_len=4)
    assert [t for t, _ in rows] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert [s for _, s in rows] == [[3, 1], [1, 3]]
    assert all(sum(s) == 4 for _, s in rows)


def test_a_document_spanning_a_boundary_is_a_segment_on_each_side() -> None:
    # The second document runs 4..5, straddling the cut at 4. Each piece is contiguous inside
    # its own sequence, which is exactly what a block-diagonal mask needs.
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    _, first_lens = _packed([batch], seq_len=4)[0]
    _, second_lens = _packed([batch], seq_len=4)[1]
    assert first_lens[-1] == 1
    assert second_lens[0] == 1


def test_the_separator_belongs_to_the_document_it_follows() -> None:
    # With an EOS the documents are 4, 3, 4 tokens wide, so the first sequence is exactly one
    # document and the second is the rest of the second plus the start of the third.
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    rows = _packed([batch], seq_len=4, eos_token=0)
    assert [s for _, s in rows] == [[4], [3, 1]]


def test_boundaries_carry_across_input_batches() -> None:
    # The seams are stream state, like the tokens themselves: a document split across two
    # input batches must not read as two documents.
    first = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3, 4, 5]]})
    second = pa.RecordBatch.from_pydict({"tokens": [[6, 7, 8]]})
    rows = _packed([first, second], seq_len=4)
    assert [t for t, _ in rows] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert [s for _, s in rows] == [[4], [1, 3]]


def test_the_padded_remainder_ends_on_a_segment() -> None:
    # The padding is its own segment, so the row still sums to seq_len and the trainer can mask
    # it by length rather than by scanning for a pad token.
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    rows = _packed([batch], seq_len=5, drop_remainder=False, pad_token=-1)
    assert rows[-1][0] == [6, 7, 8, -1, -1]
    assert rows[-1][1] == [3, 2]
    assert all(sum(s) == 5 for _, s in rows)


def test_one_document_wider_than_a_sequence_fills_every_row_it_covers() -> None:
    batch = pa.RecordBatch.from_pydict({"tokens": [list(range(10))]})
    rows = _packed([batch], seq_len=4)
    assert [s for _, s in rows] == [[4], [4]]


def test_null_rows_contribute_no_segment() -> None:
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2], None, [3, 4]]})
    rows = _packed([batch], seq_len=4)
    assert [t for t, _ in rows] == [[1, 2, 3, 4]]
    assert [s for _, s in rows] == [[2, 2]]


def test_omitting_the_column_leaves_the_schema_exactly_as_it_was() -> None:
    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    out = list(pack_sequences([batch], seq_len=4))
    assert out[0].schema.names == ["tokens"]


def test_the_boundaries_are_a_cu_seqlens_after_one_cumulative_sum() -> None:
    import itertools

    batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
    for _, lens in _packed([batch], seq_len=4):
        cu = [0, *itertools.accumulate(lens)]
        assert cu[0] == 0
        assert cu[-1] == 4
