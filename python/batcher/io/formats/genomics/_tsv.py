"""The comment-skipping TSV engine BED, GFF, and VCF share.

All three are tab-separated tables carrying header and comment lines that a plain CSV reader
cannot skip: pyarrow's reader has no comment-character option, and the lines are not confined
to a prefix it could `skip_rows` past — a BED file may carry a `track` line between blocks,
and a VCF's `##` block is followed by exactly one `#CHROM` line that *is* the header.

So the split of labour is: `_blocks` turns a block of bytes into an Arrow array of lines, a
vectorized mask decides which of them are data, and **pyarrow parses the survivors** with one
`read_csv` call per block. No step touches a line in Python, which is what took BED from
26 MB/s to within reach of `pyarrow.csv` on the same bytes (see `_blocks`).

Reading stays bounded: one block of text plus one batch of Arrow, never the file.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from typing import IO, Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher.io.formats.genomics._blocks import join_lines, lines_of
from batcher.plan.types import one_batch

#: The "no value" tokens all three formats spell the same way. `.` is each specification's
#: own missing marker — BED's absent strand, GFF's absent score or phase, VCF's absent field
#: — and an empty field is not legal in any of them but is written by enough tools to accept.
#: One list because it is one fact about this family of formats: BED, GFF and VCF each stated
#: it separately, each with its own comment saying the same thing, and all three hand it to
#: the reader below.
NULL_VALUES = [".", ""]

#: A vectorized line classifier: one flag per line of a block.
LineMask = Callable[[pa.Array], pa.Array]


def hash_comment(lines: pa.Array) -> pa.Array:
    """True for the lines starting with `#` — the comment syntax all three formats share."""
    return pc.starts_with(lines, "#")


def first_index(mask: pa.Array) -> int:
    """The position of the first True in `mask`, or -1."""
    return pc.index(mask, True).as_py() if len(mask) else -1


def _read_tsv(
    data: bytes, names: list[str], types: dict[str, pa.DataType], null_values: list[str]
) -> pa.RecordBatch:
    """Parse `\n`-separated tab-separated data lines into one `RecordBatch`.

    The conversion is pyarrow's, not this module's: an explicit column list and type map,
    so a malformed field raises there with the column named rather than being coerced to a
    plausible value here.
    """
    from pyarrow import csv as pacsv

    table = pacsv.read_csv(
        io.BytesIO(data),
        read_options=pacsv.ReadOptions(column_names=names, autogenerate_column_names=False),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            # These formats spell "no value" as a literal token rather than as an empty
            # field: `.` in GFF and VCF. Listing them here is what turns a missing score
            # into a null instead of a parse error on a float column.
            null_values=null_values,
            strings_can_be_null=True,
        ),
    )
    # `read_csv` can return several batches for a large block; combine so one block is one
    # batch. Through `one_batch`, because the spelling this replaces dropped every row past
    # the 32-bit offset limit — reachable on a block of long annotation strings.
    return one_batch(table)


def _is_clean(data: bytes, markers: tuple[bytes, ...]) -> bool:
    """Whether a block provably holds no line to drop, judged on its bytes alone.

    A line is dropped when it is blank or starts with one of `markers`; each such line
    begins the block or follows a `\n`, so a handful of `find`s in C decide it without
    splitting. A clean block — nearly every block of a real file — goes to `read_csv` as it
    was read, which is what brings these formats within reach of a plain CSV parse.
    """
    if data.startswith((b"\n", *markers)):
        return False
    return not any(b"\n" + marker in data for marker in (b"\n", *markers))


def iter_record_batches(
    blocks: Iterator[bytes],
    *,
    markers: tuple[bytes, ...],
    is_comment: LineMask,
    names: list[str],
    types: dict[str, pa.DataType],
    null_values: list[str],
    projection: list[str] | None = None,
    end_of_data: LineMask | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream line-aligned byte blocks into batches, parsing each block's data with pyarrow.

    Blank lines and the lines `is_comment` flags are dropped; `markers` are the line
    prefixes that *might* be dropped (a superset of what `is_comment` and `end_of_data`
    flag), which is what lets a clean block skip the line split entirely. `end_of_data`,
    when given, flags a line at which the table ends — GFF3's `##FASTA` — and nothing from
    it onward is read. An input with no data still yields one empty batch, so the schema is
    observable rather than something the caller has to infer from nothing.
    """
    emitted = False
    for data in blocks:
        stop = -1
        if _is_clean(data, markers):
            batch = _read_tsv(data, names, types, null_values)
        else:
            lines = lines_of(data)
            stop = -1 if end_of_data is None else first_index(end_of_data(lines))
            if stop >= 0:
                lines = lines.slice(0, stop)
            blank = pc.equal(pc.binary_length(lines), 0)
            kept = lines.filter(pc.invert(pc.or_(blank, is_comment(lines))))
            batch = _read_tsv(join_lines(kept), names, types, null_values) if len(kept) else None
        if batch is not None and batch.num_rows:
            yield _project(batch, projection)
            emitted = True
        if stop >= 0:
            break
    if not emitted:
        schema = pa.schema([pa.field(n, types[n]) for n in names])
        yield _project(pa.RecordBatch.from_pylist([], schema=schema), projection)


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
    - **bool** disagrees (`true` vs `True`) but is repaired exactly in `_to_string`.
    - **timestamp** disagrees on sub-second digits (`...:40` vs `...:40.000`).

    A column of an unlisted type is rendered by Python's `str()` — that column only, so a
    GFF's float `score` no longer sends the other eight columns down the row-wise path.
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
    """One column as `string`, with nulls rendered as `null_token`, byte-identical to `str()`."""
    if not _formats_like_python(column.type):
        return pa.array(
            [null_token if v is None else str(v) for v in column.to_pylist()], pa.string()
        )
    if pa.types.is_boolean(column.type):
        # `cast` gives `true`/`false`; Python's `str(True)` is `True`. Map explicitly.
        as_str = pc.if_else(column, "True", "False")
    else:
        as_str = pc.cast(column, pa.string())
    return pc.fill_null(as_str, null_token)


def encode_rows(
    batch: pa.RecordBatch | pa.Table, names: list[str], *, null_token: str = "."
) -> bytes:
    """Encode `names` of `batch` as tab-separated lines, one per row, each newline-terminated.

    Each column is rendered to text through Arrow when its type formats identically to
    Python's `str()` (`_formats_like_python`) and by `str()` itself otherwise, then the
    columns are joined in one kernel, so the bytes are the same either way. Measured on a
    2M-row 9-column GFF table: 17.1 s row-wise, 1.06 s vectorized (16x), byte-for-byte
    identical.

    Args:
        batch: The rows to encode.
        names: The columns to write, in output order.
        null_token: What a null renders as. These formats spell "no value" as a literal
            token rather than an empty field, which would leave two adjacent tabs.

    Returns:
        The encoded block, UTF-8.
    """
    columns = [batch.column(n) for n in names]
    columns = [c.combine_chunks() if isinstance(c, pa.ChunkedArray) else c for c in columns]
    if not columns or len(columns[0]) == 0:
        return b""
    return join_lines(
        pc.binary_join_element_wise(*[_to_string(c, null_token) for c in columns], "\t")
    )


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
