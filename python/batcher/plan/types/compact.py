"""Compacting Arrow batches into one, without the row loss the obvious spelling causes.

`Table.combine_chunks().to_batches()[0]` reads as "the table as a single batch" and is not.
`to_batches` splits at Arrow's **32-bit offset limit**, so a column holding more than 2 GiB
of `string` or `binary` data comes back as several batches, and taking the first silently
drops every row after it. The result is a short answer with no error anywhere, on exactly
the payloads Batcher exists to carry: a protobuf blob column, a decoded image or audio
column, an embedding, a large text field.

Six call sites had reached that conclusion independently and written the fix out again, in
five different spellings, while three others still had the bug. That is what this module is
for. It is in `plan.types` because the callers span `io` (layer 2), `core` (3), `dist` (4)
and `api`/`ml` (5), and `plan` is the lowest layer all of them may import — the same
argument `footprint` and `layout` make for living here.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

__all__ = ["one_batch"]


def one_batch(data: pa.Table | pa.RecordBatch | Sequence[pa.RecordBatch]) -> pa.RecordBatch | None:
    """`data` as a single contiguous `RecordBatch`, or `None` when it holds no batches.

    Never silently drops rows. When the content genuinely cannot fit one batch — more than
    2 GiB in a 32-bit-offset column — `pyarrow` raises rather than returning a prefix, which
    is the whole difference from the spelling this replaces: a loud failure the caller can
    act on instead of a short result nothing reports.

    Zero-copy in the common case. A lone batch is returned as-is, and a table of one chunk
    per column costs a batch construction and no buffer copy.

    Args:
        data: A `Table`, a `RecordBatch`, or a sequence of batches. A sequence must share
            one schema, as every producer here does.

    Returns:
        The single batch, or `None` for an empty *sequence* — which carries no schema, so
        there is nothing to return. An empty *table* does know its schema, and yields a
        zero-row batch rather than `None`, because a caller handed `None` has to invent a
        schema to type whatever comes next.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types.compact import one_batch
            >>> first = pa.record_batch({"a": pa.array([1, 2])})
            >>> second = pa.record_batch({"a": pa.array([3])})
            >>> table = pa.Table.from_batches([first, second])
            >>> one_batch(table).num_rows
            3
            >>> one_batch([]) is None
            True
            >>> empty = pa.Table.from_pylist([], schema=pa.schema([("a", pa.int64())]))
            >>> one_batch(empty).schema.names
            ['a']
    """
    if isinstance(data, pa.RecordBatch):
        return data
    if isinstance(data, pa.Table):
        batches = data.to_batches()
        if not batches:
            # A table with no chunks still knows its schema, and a caller handed a `None`
            # here has to invent one to type whatever comes next. `to_batches` is what
            # loses it, so this is the one place that can give it back.
            return pa.RecordBatch.from_pylist([], schema=data.schema)
    else:
        batches = list(data)
        if not batches:
            return None
    if len(batches) == 1:
        return batches[0]
    # `concat_batches`, not `combine_chunks().to_batches()[0]`: it produces one batch or
    # raises, where the latter produces a prefix and says nothing.
    return pa.concat_batches(batches)
