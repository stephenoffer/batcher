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

Reading is streaming and bounded: one record's sequence plus one batch, never the file. That
matters more here than for most formats, because a single FASTA file is routinely a whole
genome — a human chromosome is a quarter of a gigabyte in one record.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher.io.base import FileSink, FileSource
from batcher.io.base._lines import iter_decoded_lines
from batcher.io.formats.base import SINKS, SOURCES

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

# Records buffered before a batch is emitted. One genomic record can be enormous, so this is
# a *record* count and the memory it implies is data-dependent; the streaming loop below also
# flushes on accumulated bytes so one chromosome cannot be joined by 16,383 more before the
# batch is handed on.
_RECORDS_PER_BATCH = 16_384

# Accumulated sequence bytes that force a flush regardless of record count. 64 MiB keeps the
# reader's footprint bounded on a file of few, huge records (a reference genome) without
# fragmenting a file of many tiny ones (a protein database).
_FLUSH_BYTES = 64 << 20

#: Line width `write` wraps sequences at. 60 is the width the NCBI and UniProt reference
#: files use, so a round-tripped file is byte-comparable with the corpus it came from.
FASTA_LINE_WIDTH = 60


def _split_header(header: str) -> tuple[str, str]:
    """Split a `>` header into `(id, description)` on its first whitespace run."""
    parts = header.split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0], (parts[1].strip() if len(parts) > 1 else "")


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
    # publishes. `expand` takes a tuple directly, so a directory is listed once. The sink
    # below keeps a single string — a writer has to choose one.
    suffix = (".fasta", ".fa", ".faa", ".fna", ".ffn")
    format_name = "fasta"

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed shape)
        return FASTA_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one file into batches, holding one record plus one batch at a time."""
        ids: list[str] = []
        descs: list[str] = []
        seqs: list[str] = []
        chunks: list[str] = []
        header: str | None = None
        pending = 0

        def flush_record() -> None:
            nonlocal header, pending
            if header is None:
                return
            rec_id, desc = _split_header(header)
            seq = "".join(chunks)
            ids.append(rec_id)
            descs.append(desc)
            seqs.append(seq)
            chunks.clear()
            header = None
            pending += len(seq)

        for line in iter_decoded_lines(fh):
            line = line.rstrip("\r")
            if line.startswith(">"):
                flush_record()
                header = line[1:]
                if len(ids) >= _RECORDS_PER_BATCH or pending >= _FLUSH_BYTES:
                    yield _batch(ids, descs, seqs, projection)
                    ids, descs, seqs, pending = [], [], [], 0
            elif header is not None and line:
                chunks.append(line)
            # A line before the first `>` is not part of any record. FASTA has no comment
            # syntax in wide use (`;` was dropped decades ago), so anything there is either
            # a stray blank line or a malformed file; either way it belongs to no record and
            # is skipped rather than being attached to the first one.
        flush_record()
        # Always emit a final batch, even an empty one, so an empty file still reports the
        # schema rather than yielding nothing for a caller to infer it from.
        yield _batch(ids, descs, seqs, projection)


def _batch(
    ids: list[str], descs: list[str], seqs: list[str], projection: list[str] | None
) -> pa.RecordBatch:
    """Assemble one batch, honoring a column projection."""
    columns = {"id": ids, "description": descs, "sequence": seqs}
    names = [n for n in FASTA_SCHEMA.names if projection is None or n in projection]
    fields = [FASTA_SCHEMA.field(n) for n in names]
    arrays = [pa.array(columns[n], type=pa.string()) for n in names]
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


@SINKS.register("fasta")
class FastaSink(FileSink):
    """Write `{id, description, sequence}` rows back out as FASTA.

    Sequences are wrapped at :data:`FASTA_LINE_WIDTH`, the width the NCBI and UniProt
    reference files use, so a file written here is comparable with the corpus it came from.
    """

    suffix = ".fasta"
    format_name = "fasta"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        from batcher._internal.errors import FormatError

        missing = [n for n in ("id", "sequence") if n not in table.column_names]
        if missing:
            raise FormatError(
                f"fasta write: the table must have {missing} column(s); "
                f"got {table.column_names}. Rename or derive them before writing."
            )
        ids = table.column("id").to_pylist()
        seqs = table.column("sequence").to_pylist()
        descs = (
            table.column("description").to_pylist()
            if "description" in table.column_names
            else [None] * len(ids)
        )
        out: list[str] = []
        for rec_id, desc, seq in zip(ids, descs, seqs, strict=True):
            # A null id or sequence has no FASTA spelling — a record with no name cannot be
            # referred to, and one with no sequence is not a record — so they are written as
            # empty rather than as the string "None", which would silently corrupt the file.
            header = str(rec_id or "")
            if desc:
                header = f"{header} {desc}"
            out.append(f">{header}\n")
            text = str(seq or "")
            for i in range(0, len(text), FASTA_LINE_WIDTH):
                out.append(text[i : i + FASTA_LINE_WIDTH] + "\n")
            if not text:
                # An empty sequence still needs its blank line, or the next `>` would be
                # read as this record's sequence on the way back in.
                out.append("\n")
        fh.write("".join(out).encode("utf-8"))
