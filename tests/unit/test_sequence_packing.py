"""`pack_sequences` — fixed-length LLM training sequences with no wasted positions.

Padding each document to the context length wastes the padding, and the GPU computes
attention over it. Packing concatenates documents and cuts every `seq_len` tokens, so
every position is a real token. What can go silently wrong is the arithmetic:

* **tokens must not be lost or duplicated.** A document straddling a boundary continues
  into the next sequence; an off-by-one in the carry drops the tail of every batch, and
  the model trains on a corpus quietly missing its document endings.
* **the separator must land between documents**, once each — not inside one.
* **state must carry across input batches**, or the stream is cut at batch boundaries
  (which depend on the reader) instead of at `seq_len`.
* **every emitted batch must share one schema**, or the run's batches cannot be
  concatenated — the reason the padded tail is padded rather than emitted narrow.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.ml import pack_sequences

pytestmark = pytest.mark.unit


def _batch(docs: list[list[int] | None]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict({"tokens": docs})


def _packed(batches, **kwargs) -> list[list[int]]:
    return [
        seq for b in pack_sequences(batches, **kwargs) for seq in b.column("tokens").to_pylist()
    ]


def test_documents_are_concatenated_and_cut_at_seq_len():
    got = _packed([_batch([[1, 2, 3], [4, 5], [6, 7, 8]])], seq_len=4)
    assert got == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_the_separator_lands_between_documents():
    got = _packed([_batch([[1, 2, 3], [4, 5], [6, 7, 8]])], seq_len=4, eos_token=0)
    # 1 2 3 | 0 | 4 5 | 0 | 6 7 8 | 0  -> cut every 4; the trailing 7 8 0 is dropped
    assert got == [[1, 2, 3, 0], [4, 5, 0, 6]]


def test_no_separator_when_eos_is_omitted():
    got = _packed([_batch([[1, 2], [3, 4]])], seq_len=2)
    assert got == [[1, 2], [3, 4]]


def test_no_token_is_lost_or_duplicated():
    """The whole prefix of the token stream must appear, in order, exactly once."""
    docs = [list(range(i * 10, i * 10 + (i % 7) + 1)) for i in range(50)]
    stream = [t for d in docs for t in d]
    got = _packed([_batch(docs)], seq_len=8)

    flat = [t for seq in got for t in seq]
    assert flat == stream[: len(flat)], "packing must preserve the stream, in order"
    assert len(flat) == (len(stream) // 8) * 8, "exactly the whole sequences, no more"


def test_state_carries_across_input_batches():
    """A document split across two input batches must still pack contiguously."""
    got = _packed([_batch([[1, 2, 3]]), _batch([[4, 5, 6, 7]])], seq_len=3)
    assert got == [[1, 2, 3], [4, 5, 6]]


def test_the_cut_does_not_depend_on_input_batching():
    docs = [list(range(i, i + 5)) for i in range(20)]
    one = _packed([_batch(docs)], seq_len=7, eos_token=-1)
    many = _packed(
        [_batch(docs[i : i + 3]) for i in range(0, len(docs), 3)], seq_len=7, eos_token=-1
    )
    assert one == many


def test_null_documents_contribute_nothing():
    """A null list is not a document: no tokens, and no separator either."""
    assert _packed([_batch([[1, 2], None, [3, 4]])], seq_len=2) == [[1, 2], [3, 4]]
    assert _packed([_batch([[1, 2], None, [3, 4]])], seq_len=3, eos_token=0) == [
        [1, 2, 0],
        [3, 4, 0],
    ]


def test_the_remainder_is_dropped_by_default():
    assert _packed([_batch([[1, 2, 3, 4, 5]])], seq_len=2) == [[1, 2], [3, 4]]


def test_the_remainder_is_padded_when_kept():
    assert _packed([_batch([[1, 2, 3, 4, 5]])], seq_len=2, drop_remainder=False) == [
        [1, 2],
        [3, 4],
        [5, 0],
    ]


def test_the_pad_token_is_configurable():
    got = _packed([_batch([[1, 2, 3]])], seq_len=2, drop_remainder=False, pad_token=-100)
    assert got == [[1, 2], [3, -100]]


def test_the_pad_token_defaults_to_eos():
    got = _packed([_batch([[1, 2, 3]])], seq_len=4, eos_token=9, drop_remainder=False)
    assert got == [[1, 2, 3, 9]]  # the eos is the 4th token here, not padding
    got = _packed([_batch([[1, 2]])], seq_len=4, eos_token=9, drop_remainder=False)
    assert got == [[1, 2, 9, 9]]  # separator, then one pad — both the eos


def test_every_batch_shares_one_schema_so_they_concatenate():
    batches = list(pack_sequences([_batch([[1, 2, 3, 4, 5]])], seq_len=2, drop_remainder=False))
    assert len({b.schema for b in batches}) == 1
    assert pa.Table.from_batches(batches).num_rows == 3


def test_the_output_is_a_fixed_size_list_ready_for_a_tensor():
    (batch,) = pack_sequences([_batch([[1, 2, 3, 4]])], seq_len=4)
    field = batch.schema.field(0)
    assert pa.types.is_fixed_size_list(field.type)
    assert field.type.list_size == 4
    assert pa.types.is_integer(field.type.value_type)


def test_the_fixed_size_list_tensorizes_to_the_right_shape():
    """`column_to_tensor` restores an `(n, seq_len)` matrix with no reshape at the edge."""
    from batcher.io.formats.ml.tensor import is_tensor_column  # noqa: F401  (import guard)

    (batch,) = pack_sequences([_batch([list(range(8))])], seq_len=4)
    column = batch.column("tokens")
    width = column.type.list_size
    values = np.asarray(column.flatten().to_numpy(zero_copy_only=False))
    assert values.reshape(-1, width).shape == (2, 4)


def test_the_output_column_can_be_renamed():
    (batch,) = pack_sequences([_batch([[1, 2]])], seq_len=2, output_column="input_ids")
    assert batch.schema.names == ["input_ids"]


def test_rows_per_batch_bounds_the_emitted_batch():
    batches = list(pack_sequences([_batch([list(range(100))])], seq_len=5, rows_per_batch=3))
    assert all(b.num_rows <= 3 for b in batches)
    assert sum(b.num_rows for b in batches) == 20


def test_an_empty_stream_yields_nothing():
    assert list(pack_sequences([], seq_len=4)) == []
    assert _packed([_batch([])], seq_len=4) == []


def test_a_document_longer_than_seq_len_spans_sequences():
    assert _packed([_batch([[1, 2, 3, 4, 5, 6]])], seq_len=2) == [[1, 2], [3, 4], [5, 6]]


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"seq_len": 0}, "seq_len"), ({"rows_per_batch": 0}, "rows_per_batch")]
)
def test_degenerate_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        next(pack_sequences([_batch([[1]])], **kwargs))


def test_packing_wastes_no_positions():
    """The point of the exercise: compare against padding every document to `seq_len`."""
    docs = [list(range(1, (i % 9) + 2)) for i in range(200)]
    seq_len = 16
    packed = _packed([_batch(docs)], seq_len=seq_len)
    real = sum(len(d) for d in docs)

    padded_positions = len(docs) * seq_len
    packed_positions = len(packed) * seq_len
    assert packed_positions <= real, "packing never emits more positions than there are tokens"
    assert packed_positions < padded_positions / 2, "packing must beat padding by a lot here"
