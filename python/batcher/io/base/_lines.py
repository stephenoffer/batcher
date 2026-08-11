"""Line-delimited decoding, shared by the text, log, and genomics sources.

Both read the same shape — a file of lines becoming one Arrow string column — and both used
to do it a line at a time in Python: a `decode`, a strip and a list append of interpreted
bytecode per row, on what is routinely the largest input in a pipeline. That is per-row work
in the read path, which the control plane is not supposed to do, and it showed: the text
source read **12.8 MB/s** and the log source **122 MB/s**, against **1,352 MB/s** for the
same bytes through Arrow's own line splitter.

So the fast path here is Arrow's CSV reader used purely as a line splitter — a delimiter
that cannot occur, no quoting, no escaping — which produces the string column with no Python
object per line at all. It is **guarded**, because Arrow splits on `\\n` (absorbing a
preceding `\\r`) and the two callers want subtly different things:

* the log source splits on `\\n` **only**, and keeps a trailing `\\r` in the value;
* the text source splits the way `str.splitlines()` does, which also breaks on `\\r`,
  `\\v`, `\\f`, the file/group/record separators, `\\x85`, `\\u2028` and `\\u2029`.

Neither matches Arrow on a block containing any of those bytes — so a block containing one
is decoded by the exact Python path instead. The check is a handful of substring scans in C
over a block that is about to be parsed anyway, and on ordinary Unix text none of them ever
hits. This keeps the semantics *identical* rather than nearly so, which matters more than
the speed: a silently different line count is a wrong answer, and a `\\r` appearing or
vanishing at the end of every row of a CRLF file would be exactly that.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

__all__ = ["iter_decoded_lines", "iter_line_blocks", "lines_of", "one_array"]


# Bytes pulled per read on the decoded path. Smaller than the Arrow path's block below
# because its callers are record-oriented rather than column-oriented: a genomics reader
# holds one record plus one batch, and a 16 MiB text block would dominate that footprint.
_DECODED_BLOCK_BYTES = 1 << 20  # 1 MiB


def iter_decoded_lines(fh: IO[Any], encoding: str = "utf-8") -> Iterator[str]:
    """Yield the handle's lines as `str`, decoding incrementally and holding no more than a block.

    The line-at-a-time counterpart to `iter_line_blocks`, for a caller whose records span
    lines and so has to look at them one by one — the FASTA/FASTQ readers, and the
    comment-skipping TSV engine behind BED, GFF and VCF. Those callers cannot use the
    Arrow-backed splitter above: they decide per line whether it is data, a comment or a
    header, which is a `str` decision.

    An incremental decoder rather than `block.decode()` because a multi-byte character can
    straddle a block boundary. Sequence data is ASCII, but a *description* is free text — a
    GFF attribute, a VCF description, an accented species name in a FASTA header.

    Args:
        fh: An open binary handle positioned at the start of the data.
        encoding: The text encoding to decode with.

    Yields:
        Each line of the input, in order, without its terminator.
    """
    import codecs

    decoder = codecs.getincrementaldecoder(encoding)()
    carry = ""
    while True:
        block = fh.read(_DECODED_BLOCK_BYTES)
        if not block:
            break
        carry += decoder.decode(block)
        lines = carry.split("\n")
        carry = lines.pop()
        yield from lines
    carry += decoder.decode(b"", True)
    if carry:
        yield carry


def one_array(blocks: list[pa.Array]) -> pa.Array:
    """`blocks` as a single string array — including the empty case `concat_arrays` refuses.

    An empty file produces no block at all, and both callers still emit one empty batch from
    it so the schema is observable, so "no blocks" is a real input here rather than a
    degenerate one.
    """
    if not blocks:
        return pa.array([], pa.string())
    return blocks[0] if len(blocks) == 1 else pa.concat_arrays(blocks)


# Bytes pulled per read. Also the granularity of the C-level split, so it wants to be large
# enough that the per-block Python is noise against the parse, and small enough that a
# multi-GB file never becomes resident.
_BLOCK_BYTES = 1 << 24  # 16 MiB

# The delimiter handed to Arrow's CSV parser. It never appears in text, and a block where it
# *does* appear is refused by the guard below rather than mis-split, so this is a performance
# choice and not a correctness assumption.
_UNIT_SEPARATOR = "\x1f"

# A block holding any of these is decoded in Python instead, because Arrow's line splitting
# and the caller's disagree about it. `\r` is in both lists: Arrow absorbs it into `\r\n`,
# the log source keeps it, and `str.splitlines` breaks on it alone.
_LOG_UNSAFE = (b"\r", b"\x1f")
_SPLITLINES_UNSAFE = (
    b"\r",
    b"\x1f",
    b"\x0b",  # vertical tab
    b"\x0c",  # form feed
    b"\x1c",  # file separator
    b"\x1d",  # group separator
    b"\x1e",  # record separator
    b"\xc2\x85",  # U+0085 NEL, in UTF-8
    b"\xe2\x80\xa8",  # U+2028 LINE SEPARATOR, in UTF-8
    b"\xe2\x80\xa9",  # U+2029 PARAGRAPH SEPARATOR, in UTF-8
)


def _arrow_lines(body: memoryview) -> list[pa.Array] | None:
    """`body`'s lines as Arrow string arrays, or None if Arrow declines to split it.

    `body` must end at a line boundary, and is a `memoryview` so the caller never copies the
    block to hand it here — `pa.py_buffer` wraps it in place. The reader is configured to do
    nothing but split: one column, no header, no quoting, no escaping, no type inference.

    Arrow's own block size is left alone so it splits the buffer across threads; pinning it
    to the whole buffer forced a single-threaded parse and measured 750 MB/s against the
    1,352 MB/s the same reader manages when it can parallelize. The chunks it returns are
    handed back as they are, in order, rather than combined — combining them is a full copy
    of the block, and the callers concatenate at batch boundaries anyway.
    """
    import pyarrow.csv as pcsv

    try:
        table = pcsv.read_csv(
            pa.BufferReader(pa.py_buffer(body)),
            read_options=pcsv.ReadOptions(column_names=["line"], use_threads=True),
            parse_options=pcsv.ParseOptions(
                delimiter=_UNIT_SEPARATOR,
                quote_char=False,
                escape_char=False,
                newlines_in_values=False,
                # A blank line is a row here, not noise. Arrow's CSV default drops it, which
                # would silently lose every empty line in a log — a wrong row count, and the
                # kind that no schema or type check can catch.
                ignore_empty_lines=False,
            ),
            convert_options=pcsv.ConvertOptions(
                column_types={"line": pa.string()}, strings_can_be_null=False
            ),
        )
    except Exception:
        return None  # a malformed block is the Python path's business, not a failure
    return table.column("line").chunks


def lines_of(data: bytes, *, splitlines: bool) -> pa.Array:
    """`data`'s lines as one Arrow string array, with the same semantics as the streaming path.

    For a caller that already holds the bytes — a newline-aligned byte range read by a
    distributed split — rather than a handle to stream. Same guard, same fallback, so a
    range read and a whole-file read cannot disagree about where the lines are.

    Args:
        data: The bytes to split. Whole lines only; a range read aligns them first.
        splitlines: True for `str.splitlines()` semantics, False to split on ``\\n`` only.

    Returns:
        One string array holding every line of `data`, in order.
    """
    unsafe = _SPLITLINES_UNSAFE if splitlines else _LOG_UNSAFE
    blocks = [
        block if isinstance(block, pa.Array) else pa.array(block, pa.string())
        for block in _split(data, len(data), unsafe, splitlines)
    ]
    return one_array(blocks)


def iter_line_blocks(fh: IO[Any], *, splitlines: bool) -> Iterator[pa.Array | list[str]]:
    """Yield each block's complete lines, as an Arrow array when Arrow can split it.

    A block is yielded as a `pa.Array` on the fast path and as a `list[str]` on the exact
    Python fallback, so a caller has to accept both — `pa.array()` over a list and an array
    already built cost the same to concatenate, and forcing one representation would mean
    converting the fast path's result back into Python objects, which is the entire cost this
    exists to avoid.

    Args:
        fh: An open binary (or text) handle positioned at the start of the data.
        splitlines: True for `str.splitlines()` semantics (the text source), False to split
            on ``\\n`` only and keep a trailing ``\\r`` in the value (the log source).

    Yields:
        Each block's complete lines, in file order. Every line of the input appears exactly
        once across the blocks.
    """
    unsafe = _SPLITLINES_UNSAFE if splitlines else _LOG_UNSAFE
    carry = b""
    while True:
        block = fh.read(_BLOCK_BYTES)
        if not block:
            break
        if isinstance(block, str):  # a text-mode handle
            block = block.encode("utf-8", "replace")
        # The partial line left by the previous block is at most one line long, so joining
        # it costs one copy of the block and nothing is copied when it is empty — which is
        # every block of a file whose lines do not straddle the boundary.
        data = block if not carry else carry + block
        cut = data.rfind(b"\n")
        if cut < 0:
            carry = data
            continue  # no complete line yet
        # The complete part is handed on as a *view*. Slicing it out as `bytes` copied the
        # whole block a second time, and a 16 MiB block copied twice per read is most of
        # what this loop costs — 0.60 s of a 1.66 s read, against 0.61 s for the parse it
        # exists to feed. Only the trailing partial line is copied, and it is one line long.
        carry = data[cut + 1 :]
        yield from _split(data, cut + 1, unsafe, splitlines)
    if carry:
        yield from _split(carry, len(carry), unsafe, splitlines)


def _split(data: bytes, end: int, unsafe: tuple[bytes, ...], splitlines: bool):
    """`data[:end]`'s lines, via Arrow when the guard passes and via Python when it does not.

    Takes `end` rather than a pre-sliced `bytes` so neither the guard nor the parser needs a
    copy: `bytes.find` is bounded by it, and `memoryview` slices to it for free.
    """
    if not any(data.find(marker, 0, end) >= 0 for marker in unsafe):
        chunks = _arrow_lines(memoryview(data)[:end])
        if chunks is not None:
            yield from chunks
            return
    text = bytes(data[:end]).decode("utf-8", "replace")
    if splitlines:
        yield text.splitlines()
        return
    # `\n`-only: a trailing newline closes the last line rather than opening an empty one.
    pieces = text.split("\n")
    yield pieces[:-1] if text.endswith("\n") else pieces
