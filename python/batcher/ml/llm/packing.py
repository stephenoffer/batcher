"""Sequence packing — concatenate tokenized documents into fixed-length training sequences.

An LLM pretraining batch is `seq_len` tokens wide, and documents are not. Padding each
document to `seq_len` wastes the padding: on a corpus whose median document is a few
hundred tokens against a 4096-token context, most of every batch is padding, and most of
every GPU-hour computes attention over it. *Packing* instead lays the tokenized documents
end to end, separated by an end-of-sequence token, and cuts the stream every `seq_len`
tokens. Every position is then a real token.

Packing is inherently **sequential and stateful**: a document that does not fit is not
padded, it is carried into the next sequence. So this is a transform over a batch
*stream* (like `llm_generate`), not a `map_batches` — a parallel per-batch map would cut
the stream in a nondeterministic place. Within a batch the work is vectorized NumPy over
the Arrow list buffers: no Python touches a token.

The output column is a `FixedSizeList<Int64>[seq_len]`, which `ml.loader.column_to_tensor`
turns into a correctly-shaped `(rows, seq_len)` tensor with no reshape at the edge.

Packing has a cost that is invisible in the data and shows up in the model: a packed sequence
holds several unrelated documents, and plain causal attention lets every token attend across
those seams. The trainer needs to know where they are — FlashAttention's variable-length path
takes them as `cu_seqlens`, and a position-id reset needs the same list — so `boundaries_column`
emits the segment lengths inside each packed sequence. Without it the seams are unrecoverable
downstream, because the packed column is exactly as wide either way and nothing about it says
which token started a document.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = ["pack_sequences"]


def _tokens_with_eos(column: pa.Array, eos_token: int | None) -> tuple[np.ndarray, np.ndarray]:
    """One row's-worth of a list column flattened to `int64`, with `eos_token` after each
    document, and the resulting per-document token counts.

    Vectorized: the tokens are placed at their destination in one scatter, and the gaps
    the scatter leaves are exactly the per-document separator slots. The counts come back
    with the separator included, because that is how many tokens of the packed stream each
    document occupies — which is what a boundary list has to describe.
    """
    import numpy as np
    import pyarrow as pa

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    # An all-null (or empty, hence type-less) column carries no documents at all. Accept
    # every list flavour a tokenizer produces: `list` (the default), `large_list` (the
    # HuggingFace fast tokenizers' int32-offset output, which the old `is_list`-only check
    # silently dropped every token of), and `fixed_size_list` (padded/uniform outputs).
    empty = np.empty(0, dtype=np.int64)
    if len(column) == 0 or not _is_list_like(column.type):
        return empty, empty
    # A null list contributes no tokens, and no separator (it is not a document).
    if column.null_count:
        column = column.drop_null()

    values = column.flatten()
    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)
    lengths = _list_lengths(column)
    if eos_token is None or lengths.size == 0:
        return flat, lengths

    out = np.full(flat.size + lengths.size, eos_token, dtype=np.int64)
    # Document j's tokens shift right by j — one slot per preceding separator — so the
    # slot each document leaves behind is where its own separator lands.
    shift = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    out[np.arange(flat.size, dtype=np.int64) + shift] = flat
    return out, lengths + 1


def _is_list_like(dtype: pa.DataType) -> bool:
    """Whether `dtype` is a list column this packer can read tokens out of."""
    import pyarrow as pa

    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    )


def _list_lengths(column: pa.Array) -> np.ndarray:
    """Each row's token count. A `fixed_size_list` has no offsets — every row is `list_size`."""
    import numpy as np
    import pyarrow as pa

    if pa.types.is_fixed_size_list(column.type):
        return np.full(len(column), column.type.list_size, dtype=np.int64)
    offsets = np.asarray(column.offsets, dtype=np.int64)
    return np.diff(offsets)


def _emit(
    tokens: np.ndarray,
    seq_len: int,
    output_column: str,
    *,
    doc_ends: np.ndarray | None = None,
    boundaries_column: str | None = None,
) -> pa.RecordBatch:
    """`len(tokens) // seq_len` packed sequences as a `FixedSizeList<Int64>[seq_len]`.

    With `boundaries_column`, a second `List<Int32>` column carries the segment lengths inside
    each sequence, summing to `seq_len`. `doc_ends` are the offsets, relative to `tokens`, at
    which each document finishes.
    """
    import pyarrow as pa

    values = pa.array(tokens, type=pa.int64())
    packed = pa.FixedSizeListArray.from_arrays(values, seq_len)
    if boundaries_column is None:
        return pa.RecordBatch.from_arrays([packed], names=[output_column])
    lengths, offsets = _segments(tokens.size, seq_len, doc_ends)
    boundaries = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()), pa.array(lengths, type=pa.int32())
    )
    return pa.RecordBatch.from_arrays(
        [packed, boundaries], names=[output_column, boundaries_column]
    )


def _segments(total: int, seq_len: int, doc_ends: np.ndarray | None) -> tuple[np.ndarray, ...]:
    """Segment lengths per packed sequence, and the list offsets that group them.

    A document that straddles a sequence boundary contributes one segment on each side, which
    is what block-diagonal attention wants: within a sequence, the piece really is contiguous
    and the seam really is a seam. Every sequence therefore ends on a segment boundary, so the
    lengths of each row sum to `seq_len` and their cumulative sums are `cu_seqlens`.
    """
    import numpy as np

    seq_bounds = np.arange(seq_len, total + seq_len, seq_len, dtype=np.int64)
    ends = seq_bounds if doc_ends is None else np.union1d(doc_ends, seq_bounds)
    # A document end past the emitted region belongs to a later batch, and one at zero is not
    # an end at all.
    ends = ends[(ends > 0) & (ends <= total)]
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    lengths = ends - starts
    # Each segment belongs to the sequence its first token falls in.
    per_sequence = np.bincount(starts // seq_len, minlength=total // seq_len)
    return lengths, np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(per_sequence)])


def pack_sequences(
    batches: Iterable[pa.RecordBatch],
    *,
    token_column: str = "tokens",
    seq_len: int = 2048,
    eos_token: int | None = None,
    drop_remainder: bool = True,
    pad_token: int | None = None,
    output_column: str | None = None,
    boundaries_column: str | None = None,
    rows_per_batch: int = 64,
) -> Iterator[pa.RecordBatch]:
    """Pack a stream of tokenized documents into fixed-length training sequences.

    Documents are concatenated in stream order, separated by `eos_token`, and cut every
    `seq_len` tokens. A document straddling a boundary continues into the next sequence,
    so no position is padding. Order matters and state carries across batches: consume
    the result sequentially (it is a generator), and shuffle *before* packing, not after.

    Args:
        batches: An iterable of `pyarrow.RecordBatch` holding `token_column`.
        token_column: A `List<Int64>` column of token ids. Null rows contribute nothing.
        seq_len: Tokens per output sequence.
        eos_token: Written between documents. Omit to concatenate with no separator.
        drop_remainder: Drop the trailing partial sequence (the default — it is the one
            place padding would enter the run). When false the final sequence is padded
            to `seq_len` with `pad_token`, so every emitted batch keeps one schema.
        pad_token: Fills the final sequence when `drop_remainder` is false. Defaults to
            `eos_token`, or 0 when there is none.
        output_column: Name of the packed column; defaults to `token_column`.
        boundaries_column: Name of an additional `List<Int32>` column holding the document
            segment lengths inside each packed sequence, summing to `seq_len`. This is what a
            trainer needs to stop attention crossing the seams between the unrelated documents
            packing put together: cumulative-sum it for FlashAttention's `cu_seqlens`, or use
            it to restart position ids per document. Omit it and the seams are unrecoverable
            downstream, because the packed column is exactly as wide either way. A document
            spanning a sequence boundary contributes a segment on each side, and the final
            padded sequence's last segment covers its padding.
        rows_per_batch: Sequences per emitted `RecordBatch`.

    Yields:
        `RecordBatch`es with one `FixedSizeList<Int64>[seq_len]` column — the shape
        `ml.loader.column_to_tensor` turns into an `(n, seq_len)` tensor directly.

    Raises:
        ValueError: If `seq_len` or `rows_per_batch` is less than 1.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.ml import pack_sequences
            >>> batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
            >>> packed = list(pack_sequences([batch], seq_len=4))
            >>> packed[0].column("tokens").to_pylist()
            [[1, 2, 3, 4], [5, 6, 7, 8]]
    """
    import numpy as np

    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if rows_per_batch < 1:
        raise ValueError(f"rows_per_batch must be >= 1, got {rows_per_batch}")
    name = output_column or token_column
    chunk = seq_len * rows_per_batch

    carry = np.empty(0, dtype=np.int64)
    # Offsets into `carry` at which each pending document finishes — the seams a packed
    # sequence would otherwise lose. Tracked whether or not they are emitted, because keeping
    # two code paths through the carry is how the two stop agreeing.
    ends = np.empty(0, dtype=np.int64)

    def take(tokens: np.ndarray) -> pa.RecordBatch:
        return _emit(tokens, seq_len, name, doc_ends=ends, boundaries_column=boundaries_column)

    for batch in batches:
        tokens, doc_lengths = _tokens_with_eos(batch.column(token_column), eos_token)
        ends = np.concatenate([ends, carry.size + np.cumsum(doc_lengths)])
        carry = np.concatenate([carry, tokens])
        # Emit while a whole output batch is available; keep the tail for the next input.
        while carry.size >= chunk:
            yield take(carry[:chunk])
            carry, ends = carry[chunk:], ends[ends > chunk] - chunk
        whole = (carry.size // seq_len) * seq_len
        if whole:
            # A partial output batch is still whole sequences; emitting it now keeps the
            # carry bounded by `seq_len` rather than by the input batch size, which is what
            # makes the memory cost independent of how the caller chunked its input.
            yield take(carry[:whole])
            carry, ends = carry[whole:], ends[ends > whole] - whole

    # The loop above drains every whole sequence from `carry` after each input batch, so what
    # reaches here is strictly shorter than one sequence. Only the remainder is left to decide.
    if carry.size and not drop_remainder:
        # Pad rather than emit a narrower `FixedSizeList`: a final batch of a different
        # width has a different schema, and the run's batches could not be concatenated.
        fill = pad_token if pad_token is not None else (eos_token if eos_token is not None else 0)
        padded = np.full(seq_len, fill, dtype=np.int64)
        padded[: carry.size] = carry
        yield take(padded)
