"""FASTA format — the reference-sequence interchange format, as `{id, description, sequence}`.

A FASTA record is a `>` header line followed by the sequence, wrapped across as many lines
as the writer felt like. That wrapping is the whole reason this cannot be a `text` read plus
a group-by: the row boundary is a `>` at the start of a line, not a newline, so reassembling
records means a stateful scan. Doing that once here, in a streaming reader, is the difference
between a genome scan and a self-join.

The header is split on its first run of whitespace into `id` and `description`, which is the
NCBI convention every tool follows: `>chr1 Homo sapiens chromosome 1` is the sequence named
`chr1`, described as the rest. A header with no description yields an empty string rather
than null — the description is present and empty, which is a different fact from a header
this reader could not parse.

A line starting with `;` is a comment wherever it appears — the original Pearson format's
comment syntax, still written by some tools — and is never part of a sequence. Line endings
may be `\n`, `\r\n`, or a bare `\r`.

Reading is streaming and bounded: one block of lines plus the record that spans it, never
the file. That matters more here than for most formats, because a single FASTA file is
routinely a whole genome — a human chromosome is a quarter of a gigabyte in one record.
Records are assembled with Arrow kernels (a list view over the block's sequence lines, then
one `binary_join`), so no step touches a line in Python.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import FormatError
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.genomics._blocks import (
    iter_line_arrays,
    join_lines,
    split_headers,
    text_column,
)

__all__ = ["FastaSink", "FastaSource"]

#: Every FASTA read produces these three columns, in this order. The schema is fixed rather
#: than inferred because the format has exactly one shape — which is also why `_read_schema`
#: never touches the file.
FASTA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
    ]
)

#: Line width `write` wraps sequences at. 60 is the width the NCBI and UniProt reference
#: files use, so a round-tripped file is byte-comparable with the corpus it came from.
FASTA_LINE_WIDTH = 60

# Rows encoded per buffer when writing, so a wrapped block stays inside a `string` array's
# int32 offsets and the text held in memory scales with a block, not the table.
_ROWS_PER_WRITE_BLOCK = 4_096


class _Open:
    """The record a block ended inside: its header and the sequence text read so far."""

    __slots__ = ("header", "pieces")

    def __init__(self, header: str) -> None:
        self.header = header
        self.pieces: list[pa.Array] = []


@SOURCES.register("fasta")
class FastaSource(FileSource):
    """FASTA files as rows of `{id, description, sequence}`.

    One row per record, sequence lines re-joined. Splits are whole files: a record boundary
    is not byte-addressable without scanning, so a byte-range split could cut a record in
    half. A FASTA corpus is normally many files (one per assembly or per sample), which is
    where the parallelism comes from.
    """

    # A tuple, because a FASTA corpus mixes suffixes freely: `.fa`/`.fasta` for
    # nucleotides and `.faa`/`.fna`/`.ffn` for the amino-acid and nucleotide splits NCBI
    # publishes, each also gzipped. `expand` takes a tuple directly, so a directory is listed
    # once. The sink below keeps a single string — a writer has to choose one.
    suffix = tuple(s + gz for gz in ("", ".gz") for s in (".fasta", ".fa", ".faa", ".fna", ".ffn"))
    format_name = "fasta"

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed shape)
        return FASTA_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one file into one batch per block of the records that block completes."""
        pending: _Open | None = None
        for lines in iter_line_arrays(fh):
            batch, pending = _records(lines, pending, projection)
            if batch is not None:
                yield batch
        # The last record, and always a final batch — even an empty one, so an empty file
        # still reports the schema rather than yielding nothing for a caller to infer it from.
        headers = [] if pending is None else [pending.header]
        pieces = [] if pending is None else [_joined(pending.pieces)]
        yield _batch(pa.array(headers, pa.string()), pa.array(pieces, pa.string()), projection)


def _joined(pieces: list[pa.Array]) -> str:
    """The sequence text accumulated across blocks for one record."""
    if not pieces:
        return ""
    whole = pa.concat_arrays(pieces) if len(pieces) > 1 else pieces[0]
    return pc.binary_join(pa.ListArray.from_arrays([0, len(whole)], whole), "")[0].as_py()


def _records(
    lines: pa.Array, pending: _Open | None, projection: list[str] | None
) -> tuple[pa.RecordBatch | None, _Open | None]:
    """The records a block of lines completes, and the record it leaves open.

    A record's sequence lines are contiguous, so the lines between two headers are one run of
    a list array whose offsets are computed from the header positions; `binary_join` then
    concatenates every run in one kernel. The last record in a block may continue into the
    next, so it is carried rather than emitted.
    """
    # Blank lines and `;` comments belong to no sequence, wherever they appear.
    lines = lines.filter(
        pc.invert(pc.or_(pc.equal(pc.binary_length(lines), 0), pc.starts_with(lines, ";")))
    )
    is_header = pc.starts_with(lines, ">")
    at = np.flatnonzero(is_header.to_numpy(zero_copy_only=False))
    sequence = lines.filter(pc.invert(is_header))
    # Header i sits after i earlier headers, so its record's sequence lines start at
    # `at[i] - i` in `sequence`; the run before the first header continues `pending`.
    starts = at - np.arange(len(at))
    lead = int(starts[0]) if len(at) else len(sequence)
    if pending is not None and lead:
        pending.pieces.append(sequence.slice(0, lead))
    # Text before the first `>` of the file belongs to no record and is dropped.
    if not len(at):
        return None, pending
    headers = pc.utf8_slice_codeunits(lines.filter(is_header), 1)
    offsets = pa.array(np.append(starts, len(sequence)).astype(np.int32))
    runs = pc.binary_join(pa.ListArray.from_arrays(offsets, sequence), "")
    carry = _Open(headers[-1].as_py())
    carry.pieces.append(sequence.slice(int(starts[-1])))
    done_headers, done_runs = headers.slice(0, len(at) - 1), runs.slice(0, len(at) - 1)
    if pending is not None:
        done_headers = pa.concat_arrays([pa.array([pending.header], pa.string()), done_headers])
        done_runs = pa.concat_arrays([pa.array([_joined(pending.pieces)], pa.string()), done_runs])
    if not len(done_headers):
        return None, carry
    return _batch(done_headers, done_runs, projection), carry


def _batch(headers: pa.Array, sequences: pa.Array, projection: list[str] | None) -> pa.RecordBatch:
    """Assemble one batch from header texts and sequences, honoring a column projection."""
    ids, descs = split_headers(headers)
    columns = {"id": ids, "description": descs, "sequence": sequences}
    names = [n for n in FASTA_SCHEMA.names if projection is None or n in projection]
    return pa.RecordBatch.from_arrays(
        [columns[n] for n in names], schema=pa.schema([FASTA_SCHEMA.field(n) for n in names])
    )


@SINKS.register("fasta")
class FastaSink(FileSink):
    """Write `{id, description, sequence}` rows back out as FASTA.

    Sequences are wrapped at :data:`FASTA_LINE_WIDTH`, the width the NCBI and UniProt
    reference files use, so a file written here is comparable with the corpus it came from.
    """

    suffix = ".fasta"
    format_name = "fasta"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        missing = [n for n in ("id", "sequence") if n not in table.column_names]
        if missing:
            raise FormatError(
                f"fasta write: the table must have {missing} column(s); "
                f"got {table.column_names}. Rename or derive them before writing."
            )
        for block in table.to_batches(max_chunksize=_ROWS_PER_WRITE_BLOCK):
            fh.write(_encode(block))


def _encode(block: pa.RecordBatch) -> bytes:
    """One block of records as FASTA text, wrapped at `FASTA_LINE_WIDTH`. Vectorized.

    The wrap is one regex replace putting a newline after every full line's worth of
    characters, which leaves a trailing newline exactly when the length is a positive
    multiple of the width; that one is trimmed so every record closes with one newline. An
    empty sequence therefore still gets its blank line, or the next `>` would be read as this
    record's sequence on the way back in.
    """
    if block.num_rows == 0:
        return b""
    ids, descs, seqs = (text_column(block, n) for n in ("id", "description", "sequence"))
    for name, column in (("id", ids), ("description", descs), ("sequence", seqs)):
        if pc.any(pc.match_substring_regex(column, "[\r\n]")).as_py():
            raise FormatError(
                f"fasta write: a {name} contains a line break, which FASTA cannot represent — "
                "the file would read back as different records. Remove it before writing."
            )
    header = pc.if_else(
        pc.equal(pc.binary_length(descs), 0), ids, pc.binary_join_element_wise(ids, descs, " ")
    )
    wrapped = pc.replace_substring_regex(
        seqs, pattern=f"(.{{{FASTA_LINE_WIDTH}}})", replacement="\\1\n"
    )
    wrapped = pc.if_else(
        pc.ends_with(wrapped, "\n"), pc.utf8_slice_codeunits(wrapped, 0, -1), wrapped
    )
    return join_lines(
        pc.binary_join_element_wise(pc.binary_join_element_wise(">", header, ""), wrapped, "\n")
    )
