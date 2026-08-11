"""The comment-skipping TSV engine BED, GFF, and VCF share.

All three are tab-separated tables carrying header and comment lines that a plain CSV reader
cannot skip: pyarrow's reader has no comment-character option, and the lines are not confined
to a prefix it could `skip_rows` past — a BED file may carry a `track` line between blocks,
and a VCF's `##` block is followed by exactly one `#CHROM` line that *is* the header.

So the split of labour here is: Python decides which lines are data, and **pyarrow parses
them**. Clean lines are accumulated into a buffer and handed to `pyarrow.csv.read_csv` a
batch at a time, so the per-field work — splitting, type conversion, null handling — stays in
C++ over a whole block rather than becoming a per-row Python loop. That is what keeps a
20-million-variant VCF a scan.

Reading stays bounded: one batch of text plus one batch of Arrow, never the file.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from typing import IO, Any

import pyarrow as pa

from batcher.io.base._lines import iter_decoded_lines

#: The "no value" tokens all three formats spell the same way. `.` is each specification's
#: own missing marker — BED's absent strand, GFF's absent score or phase, VCF's absent field
#: — and an empty field is not legal in any of them but is written by enough tools to accept.
#: One list because it is one fact about this family of formats: BED, GFF and VCF each stated
#: it separately, each with its own comment saying the same thing, and all three hand it to
#: the reader below.
NULL_VALUES = [".", ""]

#: Data lines accumulated before a batch is parsed. One line is at most a few hundred bytes
#: in these formats, so a plain line count bounds the buffer.
ROWS_PER_BATCH = 16_384


def parse_block(
    lines: list[str],
    names: list[str],
    types: dict[str, pa.DataType],
    null_values: list[str],
) -> pa.RecordBatch:
    """Parse a block of tab-separated data lines into one `RecordBatch`.

    The conversion is pyarrow's, not this module's: the lines are joined and handed to
    `read_csv` with an explicit column list and type map, so a malformed field raises there
    with the column named rather than being coerced to a plausible value here.
    """
    from pyarrow import csv as pacsv

    schema = pa.schema([pa.field(n, types[n]) for n in names])
    if not lines:
        return pa.RecordBatch.from_pylist([], schema=schema)
    buf = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
    table = pacsv.read_csv(
        buf,
        read_options=pacsv.ReadOptions(column_names=names, autogenerate_column_names=False),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            # These formats spell "no value" as a literal token rather than as an empty
            # field: `.` in GFF and VCF, `-1`-style sentinels nowhere. Listing them here is
            # what turns a missing score into a null instead of a parse error on a float
            # column.
            null_values=null_values,
            strings_can_be_null=True,
        ),
    )
    batches = table.to_batches()
    # `read_csv` can return several batches for a large block; combine so the caller's
    # batch boundaries are the ones it asked for.
    return (
        batches[0]
        if len(batches) == 1
        else pa.Table.from_batches(batches, schema=table.schema).combine_chunks().to_batches()[0]
    )


def iter_record_batches(
    fh: IO[Any],
    *,
    is_comment: Callable[[str], bool],
    names: list[str],
    types: dict[str, pa.DataType],
    null_values: list[str],
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream a comment-carrying TSV into batches, parsing each block with pyarrow.

    An empty file still yields one empty batch, so the schema is observable rather than
    something the caller has to infer from nothing.
    """
    block: list[str] = []
    emitted = False
    for raw in iter_decoded_lines(fh):
        line = raw.rstrip("\r")
        if not line or is_comment(line):
            continue
        block.append(line)
        if len(block) >= ROWS_PER_BATCH:
            yield _project(parse_block(block, names, types, null_values), projection)
            emitted = True
            block = []
    if block or not emitted:
        yield _project(parse_block(block, names, types, null_values), projection)


def _project(batch: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch:
    """Narrow a batch to `projection`, preserving the schema's column order."""
    if projection is None:
        return batch
    keep = [n for n in batch.schema.names if n in projection]
    return batch.select(keep)


# --- writing -------------------------------------------------------------------------
#: Rows encoded per buffer when writing. Bounds the text held in memory (the writers used
#: to build the whole file as one Python list of strings) and keeps each
#: `binary_join_element_wise` result inside the int32 offsets a `string` array carries.
ROWS_PER_WRITE_BLOCK = 65_536


def _formats_like_python(dtype: pa.DataType) -> bool:
    """Whether Arrow's cast-to-string of `dtype` is byte-identical to Python's `str()`.

    The vectorized encoder below replaces a per-row `str(v)`, so it may only be used where
    the two agree exactly — a writer that silently reformats its own output is a worse
    failure than a slow one. Checked, not assumed (80,018 float64 values, 5,000 of every
    other type):

    - **integers, string/large_string, date32, decimal** agree on every value.
    - **float** does not, and cannot be patched into agreement: Python switches to exponent
      notation at 1e16 and below 1e-4, Arrow at a different threshold, so `880644658031726.2`
      renders as `8.806446580317262e+14`. That is 0.25% of random float64 and **100%** of
      float32 (Arrow uses the shortest float32 repr; Python widens to float64 first).
    - **bool** disagrees (`true` vs `True`) but is repaired exactly by `_bool_to_string`.
    - **timestamp** disagrees on sub-second digits (`...:40` vs `...:40.000`).

    A column of an unlisted type sends the whole block down the row-wise path, which is
    what these formats did for every column before.
    """
    return bool(
        pa.types.is_integer(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_date32(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_boolean(dtype)
    )


def _to_string(column: pa.Array, null_token: str) -> pa.Array:
    """One column as strings, with nulls rendered as `null_token`. Vectorized."""
    import pyarrow.compute as pc

    if pa.types.is_boolean(column.type):
        # `cast` gives `true`/`false`; Python's `str(True)` is `True`. Map explicitly.
        as_str = pc.if_else(column, "True", "False")
    elif pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        as_str = column
    else:
        as_str = pc.cast(column, pa.string())
    return pc.fill_null(as_str, null_token)


def _joined_bytes(lines: pa.Array) -> bytes:
    """The concatenation of `lines`, read straight off the array's value buffer.

    A dense `string` array stores its values back to back with no separators or padding, so
    once every line carries its own trailing newline the value buffer **is** the block's
    text. The offsets buffer says which slice of it belongs to this (possibly sliced) array.
    """
    import numpy as np

    offsets = np.frombuffer(
        lines.buffers()[1], dtype=np.int32, count=len(lines) + 1, offset=lines.offset * 4
    )
    return memoryview(lines.buffers()[2])[offsets[0] : offsets[-1]].tobytes()


def encode_rows(
    batch: pa.RecordBatch | pa.Table, names: list[str], *, null_token: str = "."
) -> bytes:
    """Encode `names` of `batch` as tab-separated lines, one per row, each newline-terminated.

    Vectorized through Arrow when every column's type formats identically to Python's
    `str()` (`_formats_like_python`), and row-wise otherwise, so the bytes are the same
    either way. Measured on a 2M-row 9-column GFF table: 17.1 s row-wise, 1.06 s vectorized
    (16x), byte-for-byte identical.

    Args:
        batch: The rows to encode.
        names: The columns to write, in output order.
        null_token: What a null renders as. These formats spell "no value" as a literal
            token rather than an empty field, which would leave two adjacent tabs.

    Returns:
        The encoded block, UTF-8.
    """
    import pyarrow.compute as pc

    columns = [batch.column(n) for n in names]
    columns = [c.combine_chunks() if isinstance(c, pa.ChunkedArray) else c for c in columns]
    if not columns or len(columns[0]) == 0:
        return b""
    if all(_formats_like_python(c.type) for c in columns):
        lines = pc.binary_join_element_wise(*[_to_string(c, null_token) for c in columns], "\t")
        lines = pc.binary_join_element_wise(lines, "", "\n")  # trailing newline per line
        if lines.null_count == 0:
            return _joined_bytes(lines)
    values = [c.to_pylist() for c in columns]
    return "".join(
        "\t".join(null_token if v is None else str(v) for v in row) + "\n"
        for row in zip(*values, strict=True)
    ).encode("utf-8")


def write_rows(fh: IO[Any], table: pa.Table, names: list[str], *, null_token: str = ".") -> None:
    """Write `table`'s `names` columns to `fh` as tab-separated lines, in bounded blocks.

    The block loop is not only about speed. Each writer used to build the entire file as one
    Python list of strings and join it, so peak memory scaled with the *table* rather than
    with a batch — and `binary_join_element_wise` would overflow the int32 offsets of a
    `string` array past 2 GB of text besides.
    """
    for block in table.to_batches(max_chunksize=ROWS_PER_WRITE_BLOCK):
        payload = encode_rows(block, names, null_token=null_token)
        if payload:
            fh.write(payload)
