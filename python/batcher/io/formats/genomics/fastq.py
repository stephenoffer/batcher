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

Reading is streaming and bounded: one batch of reads, never the file. A FASTQ file is
routinely tens of gigabytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base import FileSink, FileSource
from batcher.io.base._lines import iter_decoded_lines
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.genomics.fasta import _split_header

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

# Reads per emitted batch. Unlike FASTA, a FASTQ record is a sequencing read — bounded at a
# few hundred bases for short-read platforms — so a plain record count bounds memory.
_READS_PER_BATCH = 16_384


@SOURCES.register("fastq")
class FastqSource(FileSource):
    """FASTQ files as rows of `{id, description, sequence, quality}`.

    One row per read. Splits are whole files, which is where the parallelism comes from: a
    sequencing run is delivered as many files (per lane, per sample, per mate), and a
    byte-range split of one file could land mid-record with no way to tell — a `@` is also a
    legal quality character, so the four-line boundary is not recoverable from a random
    offset.
    """

    # Both conventional suffixes; see `FastaSource.suffix`.
    suffix = (".fastq", ".fq")
    format_name = "fastq"

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed shape)
        return FASTQ_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one file into batches of reads, four lines at a time."""
        ids: list[str] = []
        descs: list[str] = []
        seqs: list[str] = []
        quals: list[str] = []
        # The four-line window, held as a list so a truncated final record is detectable.
        window: list[str] = []
        record_no = 0

        for raw in iter_decoded_lines(fh):
            line = raw.rstrip("\r")
            # A blank line between records is tolerated (some writers emit one), but only
            # outside a record — inside, a blank line *is* an empty sequence or quality.
            if not window and not line:
                continue
            window.append(line)
            if len(window) < 4:
                continue
            record_no += 1
            header, seq, plus, qual = window
            window = []
            if not header.startswith("@"):
                raise FormatError(
                    f"fastq: record {record_no} does not start with '@' (got {header[:32]!r}). "
                    "The file is not four-line FASTQ, or a record is truncated."
                )
            if not plus.startswith("+"):
                raise FormatError(
                    f"fastq: record {record_no} has no '+' separator on its third line "
                    f"(got {plus[:32]!r})."
                )
            if len(seq) != len(qual):
                # The one check that matters: the quality string is one character per base,
                # so a mismatch means every score downstream is attributed to the wrong
                # base. Silently truncating or padding would produce a plausible column.
                raise FormatError(
                    f"fastq: record {record_no} has {len(seq)} bases but {len(qual)} "
                    "quality characters; the file is corrupt or truncated."
                )
            rec_id, desc = _split_header(header[1:])
            ids.append(rec_id)
            descs.append(desc)
            seqs.append(seq)
            quals.append(qual)
            if len(ids) >= _READS_PER_BATCH:
                yield _batch(ids, descs, seqs, quals, projection)
                ids, descs, seqs, quals = [], [], [], []

        if window:
            raise FormatError(
                f"fastq: the file ends mid-record — {len(window)} of 4 lines after "
                f"{record_no} complete record(s)."
            )
        # Always emit a final batch, even an empty one, so an empty file still reports the
        # schema rather than yielding nothing.
        yield _batch(ids, descs, seqs, quals, projection)


def _batch(
    ids: list[str],
    descs: list[str],
    seqs: list[str],
    quals: list[str],
    projection: list[str] | None,
) -> pa.RecordBatch:
    """Assemble one batch, honoring a column projection."""
    columns = {"id": ids, "description": descs, "sequence": seqs, "quality": quals}
    names = [n for n in FASTQ_SCHEMA.names if projection is None or n in projection]
    fields = [FASTQ_SCHEMA.field(n) for n in names]
    arrays = [pa.array(columns[n], type=pa.string()) for n in names]
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


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
        ids = table.column("id").to_pylist()
        seqs = table.column("sequence").to_pylist()
        quals = table.column("quality").to_pylist()
        descs = (
            table.column("description").to_pylist()
            if "description" in table.column_names
            else [None] * len(ids)
        )
        out: list[str] = []
        for i, (rec_id, desc, seq, qual) in enumerate(zip(ids, descs, seqs, quals, strict=True)):
            text, score = str(seq or ""), str(qual or "")
            if len(text) != len(score):
                # Refused on the way out for the same reason it is refused on the way in: a
                # file whose two strings disagree is one every reader will misinterpret, and
                # writing it would push the corruption downstream instead of stopping here.
                raise FormatError(
                    f"fastq write: row {i} has {len(text)} bases but {len(score)} quality "
                    "characters; they must be equal."
                )
            header = str(rec_id or "")
            if desc:
                header = f"{header} {desc}"
            out.append(f"@{header}\n{text}\n+\n{score}\n")
        fh.write("".join(out).encode("utf-8"))
