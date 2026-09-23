"""FASTQ format — sequencing reads as `{id, description, sequence, quality}`.

A FASTQ record is exactly four lines: `@header`, the sequence, a `+` separator, and the
per-base quality string. That fixed shape makes the reader simpler than FASTA's — no
line-wrapping to reassemble — and it makes one check load-bearing: the sequence and the
quality string must be the same length, because the quality string is one character per
base. A file where they disagree is corrupt, and every downstream quality filter would
silently read the wrong base's score.

The quality string is emitted as **text**, not as decoded integers. That is deliberate: the
ASCII offset (33 for Sanger and Illumina 1.8+, 64 for the older pipelines) is not recoverable
from the bytes, so decoding here would mean guessing. `.seq.phred_quality(offset=...)`,
`.seq.mean_quality(...)` and `.seq.expected_errors(...)` decode it in the data plane once the
caller has said which encoding the run used.

Reading is streaming and bounded: one block of reads, never the file. A FASTQ file is
routinely tens of gigabytes. The four-line records are cut out of a block of lines with
strided `take`s and checked with vectorized kernels; only a block containing a blank line —
legal between records, meaningful inside one — is walked line by line, because telling
those two apart is a state machine.
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

__all__ = ["FastqSink", "FastqSource"]

#: Every FASTQ read produces these four columns, in this order.
FASTQ_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("quality", pa.string(), nullable=False),
    ]
)

# Rows encoded per buffer when writing; see `_tsv.ROWS_PER_WRITE_BLOCK`.
_ROWS_PER_WRITE_BLOCK = 65_536


@SOURCES.register("fastq")
class FastqSource(FileSource):
    """FASTQ files as rows of `{id, description, sequence, quality}`.

    One row per read. Splits are whole files, which is where the parallelism comes from: a
    sequencing run is delivered as many files (per lane, per sample, per mate), and a
    byte-range split of one file could land mid-record with no way to tell — a `@` is also a
    legal quality character, so the four-line boundary is not recoverable from a random
    offset.
    """

    # Both conventional suffixes, plain and gzipped; see `FastaSource.suffix`.
    suffix = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
    format_name = "fastq"

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed shape)
        return FASTQ_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one file into batches of reads, one batch per block of lines."""
        # Lines of a record a block ended inside, carried to the next block.
        window = pa.array([], pa.string())
        done = 0  # complete records so far, for error messages
        emitted = False
        for block in iter_line_arrays(fh):
            lines = pa.concat_arrays([window, block]) if len(window) else block
            if pc.any(pc.equal(pc.binary_length(lines), 0)).as_py():
                quads, window = _quads_walked(lines)
            else:
                usable = len(lines) - len(lines) % 4
                quads, window = lines.slice(0, usable), lines.slice(usable)
            if len(quads):
                yield _batch(quads, done, projection)
                done += len(quads) // 4
                emitted = True
        if len(window):
            raise FormatError(
                f"fastq: the file ends mid-record — {len(window)} of 4 lines after "
                f"{done} complete record(s)."
            )
        if not emitted:
            # An empty file still reports the schema rather than yielding nothing.
            yield _batch(pa.array([], pa.string()), 0, projection)


def _quads_walked(lines: pa.Array) -> tuple[pa.Array, pa.Array]:
    """Split `lines` into whole four-line records and a trailing partial one, line by line.

    The exact rule for a blank line: between records it is tolerated (some writers emit
    one), inside a record it *is* an empty sequence or quality. Only a block that contains a
    blank line comes here.
    """
    kept: list[str] = []
    window: list[str] = []
    for line in lines.to_pylist():
        if not window and not line:
            continue
        window.append(line)
        if len(window) == 4:
            kept.extend(window)
            window = []
    return pa.array(kept, pa.string()), pa.array(window, pa.string())


def _batch(quads: pa.Array, done: int, projection: list[str] | None) -> pa.RecordBatch:
    """Validate and assemble whole four-line records into one batch."""
    n = len(quads) // 4
    stride = np.arange(n, dtype=np.int64) * 4
    header, seq, plus, qual = (quads.take(pa.array(stride + k)) for k in range(4))
    _check(header, seq, plus, qual, done)
    ids, descs = split_headers(pc.utf8_slice_codeunits(header, 1))
    columns = {"id": ids, "description": descs, "sequence": seq, "quality": qual}
    names = [c for c in FASTQ_SCHEMA.names if projection is None or c in projection]
    return pa.RecordBatch.from_arrays(
        [columns[c] for c in names], schema=pa.schema([FASTQ_SCHEMA.field(c) for c in names])
    )


def _check(header: pa.Array, seq: pa.Array, plus: pa.Array, qual: pa.Array, done: int) -> None:
    """Raise on the first malformed record, numbering it within the whole file."""
    bad_header = pc.invert(pc.starts_with(header, "@"))
    bad_plus = pc.invert(pc.starts_with(plus, "+"))
    # The one check that matters: the quality string is one character per base, so a
    # mismatch means every score downstream is attributed to the wrong base. Silently
    # truncating or padding would produce a plausible column.
    bad_length = pc.not_equal(pc.utf8_length(seq), pc.utf8_length(qual))
    bad = pc.or_(pc.or_(bad_header, bad_plus), bad_length)
    at = pc.index(bad, True).as_py()
    if at < 0:
        return
    record = done + at + 1
    if bad_header[at].as_py():
        raise FormatError(
            f"fastq: record {record} does not start with '@' (got {header[at].as_py()[:32]!r}). "
            "The file is not four-line FASTQ, or a record is truncated."
        )
    if bad_plus[at].as_py():
        raise FormatError(
            f"fastq: record {record} has no '+' separator on its third line "
            f"(got {plus[at].as_py()[:32]!r})."
        )
    raise FormatError(
        f"fastq: record {record} has {len(seq[at].as_py())} bases but "
        f"{len(qual[at].as_py())} quality characters; the file is corrupt or truncated."
    )


@SINKS.register("fastq")
class FastqSink(FileSink):
    """Write `{id, description, sequence, quality}` rows back out as four-line FASTQ."""

    suffix = ".fastq"
    format_name = "fastq"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        missing = [n for n in ("id", "sequence", "quality") if n not in table.column_names]
        if missing:
            raise FormatError(
                f"fastq write: the table must have {missing} column(s); "
                f"got {table.column_names}. Rename or derive them before writing."
            )
        start = 0
        for block in table.to_batches(max_chunksize=_ROWS_PER_WRITE_BLOCK):
            fh.write(_encode(block, start))
            start += block.num_rows


def _encode(block: pa.RecordBatch, start: int) -> bytes:
    """One block of reads as four-line FASTQ text. Vectorized."""
    if block.num_rows == 0:
        return b""
    ids, descs = text_column(block, "id"), text_column(block, "description")
    seqs, quals = text_column(block, "sequence"), text_column(block, "quality")
    mismatch = pc.not_equal(pc.utf8_length(seqs), pc.utf8_length(quals))
    at = pc.index(mismatch, True).as_py()
    if at >= 0:
        # Refused on the way out for the same reason it is refused on the way in: a file
        # whose two strings disagree is one every reader will misinterpret, and writing it
        # would push the corruption downstream instead of stopping here.
        raise FormatError(
            f"fastq write: row {start + at} has {len(seqs[at].as_py())} bases but "
            f"{len(quals[at].as_py())} quality characters; they must be equal."
        )
    header = pc.if_else(
        pc.equal(pc.binary_length(descs), 0), ids, pc.binary_join_element_wise(ids, descs, " ")
    )
    return join_lines(
        pc.binary_join_element_wise(
            pc.binary_join_element_wise("@", header, ""), seqs, "+", quals, "\n"
        )
    )
