"""GFF3 and GTF format — genome annotations, nine tab-separated columns.

An annotation record says "this range of this sequence is a gene / exon / CDS", with the ninth
column carrying arbitrary key-value attributes. GFF3 and GTF differ *only* in how that ninth
column is encoded — `ID=gene1;Name=BRCA1` versus `gene_id "gene1"; gene_name "BRCA1"` — so
both read through one source and the attributes arrive as raw text.

That is deliberate rather than lazy. Parsing attributes here would mean either guessing the
dialect from the file (they are not reliably distinguishable, and a `.gff` extension is used
for both) or exploding a `Map` column whose keys differ per row and per feature type. The
honest shape is the text, plus the engine's own string vocabulary to pull a key out:

    col("attributes").str.regexp_extract(r'gene_id "([^"]+)"', 1)   # GTF
    col("attributes").str.regexp_extract(r'ID=([^;]+)', 1)          # GFF3

**Coordinates are 1-based and inclusive**, the opposite of BED. Both are read exactly as
written; converting silently would make a record disagree with its own file.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.genomics import _tsv
from batcher.io.formats.genomics._tsv import NULL_VALUES

__all__ = ["GFF_SCHEMA", "GffSink", "GffSource"]

#: The nine GFF/GTF columns. Names follow the GFF3 specification, which is what a reader
#: coming from `gffutils` or Bioconductor expects; GTF's own documentation uses the same
#: fields under slightly different names for the first three, and the positions are identical.
GFF_SCHEMA = pa.schema(
    [
        pa.field("seqid", pa.string()),
        pa.field("source", pa.string()),
        pa.field("type", pa.string()),
        pa.field("start", pa.int64()),
        pa.field("end", pa.int64()),
        # A score is a float and is very often absent, which the format spells `.`.
        pa.field("score", pa.float64()),
        pa.field("strand", pa.string()),
        # The codon phase, 0/1/2, absent on every feature that is not a CDS.
        pa.field("phase", pa.int32()),
        pa.field("attributes", pa.string()),
    ]
)

_NAMES = list(GFF_SCHEMA.names)
_TYPES = {f.name: f.type for f in GFF_SCHEMA}


def _is_comment(line: str) -> bool:
    # `#` covers both the `##` directives (`##gff-version 3`, `##sequence-region`) and plain
    # comments. The `##FASTA` directive ends the annotation section and is followed by
    # sequence data; those lines start with `>` or are bare sequence, and both are skipped
    # by the column-count check in `_parse`, so no separate state machine is needed.
    return line.startswith("#")


@SOURCES.register("gff")
class GffSource(FileSource):
    """GFF3 / GTF annotation files as nine-column rows.

    Splits are whole files. An annotation set is normally one file per genome build, and a
    byte-range split is not safe: a `#` can legally appear inside the attributes column, so a
    line boundary found from a random offset is not necessarily a record boundary.
    """

    # Every conventional spelling, including the GTF ones — the two dialects share this
    # reader because they share their nine columns.
    suffix = (".gff", ".gff3", ".gtf", ".gff.gz", ".gff3.gz", ".gtf.gz")
    format_name = "gff"

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed shape)
        return GFF_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        yield from _tsv.iter_record_batches(
            fh,
            is_comment=_is_comment,
            names=_NAMES,
            types=_TYPES,
            null_values=NULL_VALUES,
            projection=projection,
        )


@SINKS.register("gff")
class GffSink(FileSink):
    """Write annotation rows back out as GFF3, with the version directive on the first line."""

    suffix = ".gff3"
    format_name = "gff"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        missing = [n for n in _NAMES if n not in table.column_names]
        if missing:
            raise FormatError(
                f"gff write: the table must have {missing} column(s); got "
                f"{table.column_names}. GFF is positional and all nine are required."
            )
        # The version directive is required by the GFF3 specification and is what tells the
        # next reader which attribute dialect the file uses.
        fh.write(b"##gff-version 3\n")
        _tsv.write_rows(fh, table, _NAMES)
