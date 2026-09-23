"""Line blocks as Arrow arrays — the one line splitter every genomics reader shares.

Each of these formats used to read a line at a time in Python: a decode, an `rstrip` and a
list append of interpreted bytecode per line, followed for FASTA by a per-line `str` join.
That is per-row work in the read path, and it showed — BED read at 26 MB/s against 1,205
MB/s for the same bytes through `pyarrow.csv`. Here a whole block of bytes becomes one Arrow
string array, split by Arrow's CSV reader used purely as a line splitter, and the readers
make their decisions (comment, header, record boundary) with vectorized kernels over it.

Two things this does that `io.base._lines` does not, and both are why it is not reused:

* **Every line ending is a line ending.** `\\r\\n` and a bare `\\r` (classic Mac) are
  normalized to `\\n` before splitting. A CR-only FASTA used to read as zero records — the
  whole file was one line before the first `>` — and a line splitter that does not know
  about `\\r` cannot fix that downstream.
* **Invalid UTF-8 is an error, not a replacement character.** The shared splitter's Python
  fallback decodes with ``"replace"``, which suits free text; a genomics file with invalid
  bytes is usually a binary file under a text name (a BCF, a BGZF file missing its `.gz`),
  and a U+FFFD in a sequence is a wrong answer rather than a readable one.

Reading stays bounded: one block of bytes plus its lines, never the file.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "iter_blocks",
    "iter_line_arrays",
    "join_lines",
    "lines_of",
    "split_headers",
    "text_column",
]

#: Bytes pulled per read. Large enough that the per-block Python is noise against the
#: split, small enough that a schema read (which needs only the first block) stays cheap.
_BLOCK_BYTES = 4 << 20

# Arrow's CSV reader splits on a delimiter that never occurs in these formats, so each line
# is one field. A block that *does* contain it takes the exact Python path below.
_UNIT_SEPARATOR = b"\x1f"


def _split(data: bytes) -> pa.Array:
    """`data`'s lines (it is `\\n`-separated, with no `\\r`) as one string array."""
    if not data:
        return pa.array([], pa.string())
    if _UNIT_SEPARATOR not in data:
        import pyarrow.csv as pcsv

        try:
            table = pcsv.read_csv(
                pa.BufferReader(pa.py_buffer(data)),
                read_options=pcsv.ReadOptions(column_names=["line"], use_threads=True),
                parse_options=pcsv.ParseOptions(
                    delimiter=_UNIT_SEPARATOR.decode(),
                    quote_char=False,
                    escape_char=False,
                    newlines_in_values=False,
                    # A blank line is a line. FASTQ gives it meaning inside a record (an
                    # empty read), so the splitter must not decide it is noise.
                    ignore_empty_lines=False,
                ),
                convert_options=pcsv.ConvertOptions(
                    column_types={"line": pa.string()}, strings_can_be_null=False
                ),
            )
            column = table.column("line")
            return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
        except pa.ArrowInvalid:
            # A line longer than Arrow's parse block (an unwrapped chromosome), or bytes
            # that are not UTF-8. The strict decode below is exact for the first and raises
            # `UnicodeDecodeError` for the second, which is the error the caller should see.
            pass
    text = data.decode("utf-8")
    pieces = text.split("\n")
    if text.endswith("\n"):
        pieces.pop()  # a trailing newline closes the last line rather than opening one
    return pa.array(pieces, pa.string())


def iter_blocks(fh: IO[Any], first: bytes = b"") -> Iterator[bytes]:
    """Yield the handle's bytes in blocks that end on a line boundary, with `\n` endings only.

    `\r\n` and a bare `\r` are rewritten to `\n`, so every consumer splits on one byte.
    Each block but possibly the last ends with `\n`.

    Args:
        fh: An open binary handle.
        first: Bytes already read from `fh` (a magic-number peek), treated as its start.

    Yields:
        Non-empty blocks, in order, covering the input exactly once.
    """
    carry = first
    while True:
        block = fh.read(_BLOCK_BYTES)
        if not block:
            break
        data = carry + block if carry else block
        if b"\r" in data:
            # A `\r` ending the block may be the first half of a `\r\n` that the next read
            # completes, so it is held back rather than normalized on its own.
            held = b"\r" if data.endswith(b"\r") else b""
            if held:
                data = data[:-1]
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n") + held
        cut = data.rfind(b"\n")
        if cut < 0:
            carry = data
            continue
        carry = data[cut + 1 :]
        yield data[: cut + 1]
    if carry:
        yield carry.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def lines_of(data: bytes) -> pa.Array:
    """One block from `iter_blocks` as a string array of its lines, without terminators."""
    return _split(data)


def iter_line_arrays(fh: IO[Any], first: bytes = b"") -> Iterator[pa.Array]:
    """Yield the handle's lines, a block at a time, as Arrow string arrays.

    Every line of the input appears exactly once across the arrays, in order, without its
    terminator. `\n`, `\r\n` and a bare `\r` all end a line.

    Args:
        fh: An open binary handle.
        first: Bytes already read from `fh`, treated as its start.

    Yields:
        One string array per block.
    """
    for data in iter_blocks(fh, first):
        yield _split(data)


def join_lines(lines: pa.Array) -> bytes:
    """`lines` as one `\\n`-terminated byte block — the input a `read_csv` call takes.

    Reads the joined array's value buffer directly: once every line carries its own newline,
    a dense string array's values *are* the block's text, back to back.
    """
    import numpy as np

    if len(lines) == 0:
        return b""
    joined = pc.binary_join_element_wise(lines, "", "\n")
    width = np.int64 if pa.types.is_large_string(joined.type) else np.int32
    size = np.dtype(width).itemsize
    offsets = np.frombuffer(
        joined.buffers()[1], dtype=width, count=len(joined) + 1, offset=joined.offset * size
    )
    return memoryview(joined.buffers()[2])[offsets[0] : offsets[-1]].tobytes()


# The characters Python's `str.split()` treats as whitespace within ASCII, minus `\n`/`\r`
# (which cannot occur inside a line). Spelled out rather than taken from RE2's `\s` or
# Arrow's ASCII-whitespace kernels, which omit `\v` and the separators 0x1c-0x1f: the
# vectorized split must agree with `str.split` exactly.
_WS_CHARS = "\t\x0b\x0c\x1c\x1d\x1e\x1f "
_WS_RUN = "[\t\x0b\x0c\x1c-\x1f ]+"


def split_headers(headers: pa.Array) -> tuple[pa.Array, pa.Array]:
    """Split FASTA/FASTQ header texts into `(id, description)` on the first whitespace run.

    The NCBI convention: `chr1 Homo sapiens chromosome 1` is id `chr1`, description the rest,
    stripped; a header with no description has an empty one. Vectorized for ASCII headers;
    a block holding any non-ASCII header is split by Python's `str.split`, whose idea of
    whitespace (Unicode's) the regex does not reproduce.

    Args:
        headers: Header texts without their leading `>` / `@`.

    Returns:
        The id and description arrays, one entry per header.
    """
    if len(headers) == 0 or pc.all(pc.string_is_ascii(headers)).as_py():
        import numpy as np

        parts = pc.split_pattern_regex(
            pc.utf8_ltrim(headers, characters=_WS_CHARS), pattern=_WS_RUN, max_splits=1
        )
        # Every list holds the id and, when there was whitespace, the rest. Both are picked
        # out of the flat values with one `take` each; a row with no description points at
        # an appended empty string.
        offsets = parts.offsets.to_numpy()
        starts, lengths = offsets[:-1], np.diff(offsets)
        values = pa.concat_arrays([parts.values, pa.array([""], pa.string())])
        rest = values.take(pa.array(np.where(lengths == 2, starts + 1, len(values) - 1)))
        return values.take(pa.array(starts)), pc.utf8_trim(rest, characters=_WS_CHARS)
    ids: list[str] = []
    descs: list[str] = []
    for header in headers.to_pylist():
        found = header.split(maxsplit=1)
        ids.append(found[0] if found else "")
        descs.append(found[1].strip() if len(found) > 1 else "")
    return pa.array(ids, pa.string()), pa.array(descs, pa.string())


def text_column(block: pa.RecordBatch, name: str) -> pa.Array:
    """Column `name` of `block` as strings for a writer; null and a missing column are empty.

    A null id, sequence or quality has no spelling in these line formats -- a record with no
    name cannot be referred to, one with no sequence is not a record -- so each is written
    as empty rather than as the string "None", which would silently corrupt the file.
    """
    if name not in block.schema.names:
        return pa.array([""] * block.num_rows, pa.string())
    column = block.column(name)
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        column = pa.array(["" if v is None else str(v) for v in column.to_pylist()], pa.string())
    return pc.fill_null(column.cast(pa.string()), "")
