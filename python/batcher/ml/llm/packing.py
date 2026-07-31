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
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = ["pack_sequences"]


def _tokens_with_eos(column: pa.Array, eos_token: int | None) -> np.ndarray:
    """One row's-worth of a list column flattened to `int64`, with `eos_token` after each
    document.

    Vectorized: the tokens are placed at their destination in one scatter, and the gaps
    the scatter leaves are exactly the per-document separator slots.
    """
    import numpy as np
    import pyarrow as pa

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    # An all-null (or empty, hence type-less) column carries no documents at all. Accept
    # every list flavour a tokenizer produces: `list` (the default), `large_list` (the
    # HuggingFace fast tokenizers' int32-offset output, which the old `is_list`-only check
    # silently dropped every token of), and `fixed_size_list` (padded/uniform outputs).
    if len(column) == 0 or not _is_list_like(column.type):
        return np.empty(0, dtype=np.int64)
    # A null list contributes no tokens, and no separator (it is not a document).
    if column.null_count:
        column = column.drop_null()

    values = column.flatten()
    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)
    if eos_token is None:
        return flat

    lengths = _list_lengths(column)
    if lengths.size == 0:
        return flat

    out = np.full(flat.size + lengths.size, eos_token, dtype=np.int64)
    # Document j's tokens shift right by j — one slot per preceding separator — so the
    # slot each document leaves behind is where its own separator lands.
    shift = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    out[np.arange(flat.size, dtype=np.int64) + shift] = flat
    return out


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


def _emit(tokens: np.ndarray, seq_len: int, output_column: str) -> pa.RecordBatch:
    """`len(tokens) // seq_len` packed sequences as a `FixedSizeList<Int64>[seq_len]`."""
    import pyarrow as pa

    values = pa.array(tokens, type=pa.int64())
    return pa.RecordBatch.from_arrays(
        [pa.FixedSizeListArray.from_arrays(values, seq_len)], names=[output_column]
    )


def pack_sequences(
    batches: Iterable[pa.RecordBatch],
    *,
    token_column: str = "tokens",
    seq_len: int = 2048,
    eos_token: int | None = None,
    drop_remainder: bool = True,
    pad_token: int | None = None,
    output_column: str | None = None,
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
    for batch in batches:
        carry = np.concatenate([carry, _tokens_with_eos(batch.column(token_column), eos_token)])
        # Emit while a whole output batch is available; keep the tail for the next input.
        while carry.size >= chunk:
            yield _emit(carry[:chunk], seq_len, name)
            carry = carry[chunk:]
        whole = (carry.size // seq_len) * seq_len
        if whole:
            # A partial output batch is still whole sequences; emitting it now keeps the
            # carry bounded by `seq_len` rather than by the input batch size, which is what
            # makes the memory cost independent of how the caller chunked its input.
            yield _emit(carry[:whole], seq_len, name)
            carry = carry[whole:]

    # The loop above drains every whole sequence from `carry` after each input batch, so what
    # reaches here is strictly shorter than one sequence. Only the remainder is left to decide.
    if carry.size and not drop_remainder:
        # Pad rather than emit a narrower `FixedSizeList`: a final batch of a different
        # width has a different schema, and the run's batches could not be concatenated.
        fill = pad_token if pad_token is not None else (eos_token if eos_token is not None else 0)
        padded = np.full(seq_len, fill, dtype=np.int64)
        padded[: carry.size] = carry
        yield _emit(padded, seq_len, name)
