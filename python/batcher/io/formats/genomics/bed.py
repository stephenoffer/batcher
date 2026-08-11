"""BED format — genomic intervals, the coordinate currency of every annotation track.

A BED file is a tab-separated interval list with between 3 and 12 columns, and the count is
the file's own choice: BED3 is `chrom/start/end`, BED6 adds `name/score/strand`, BED12 adds
the block structure that describes exons. There is no header, so the width is discovered from
the first data line and the standard names are applied in order.

**Coordinates are 0-based and half-open** — `chr1 0 100` is the first hundred bases — which is
the opposite convention to GFF, VCF, and every genome browser's display. That is the single
most common off-by-one in the field, and it is why `start` and `end` are read exactly as
written rather than being "helpfully" normalized: a silent +1 would make a BED interval
disagree with the file it came from and with every other tool that reads it.

Once read, an interval table joins against another with the engine's range join, which is
what makes "which variants fall in an exon" a relational query rather than a script.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base import FileSink, FileSource
from batcher.io.base._lines import iter_decoded_lines
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.genomics import _tsv
from batcher.io.formats.genomics._tsv import NULL_VALUES

__all__ = ["BED_COLUMNS", "BedSink", "BedSource"]

#: The twelve BED columns in their fixed order, with the Arrow type each carries. A file
#: declares its width by how many it writes; the names and order are the specification's, so
#: a BED6 file read here has the same first six columns as a BED12 file.
BED_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("chrom", pa.string()),
    ("start", pa.int64()),
    ("end", pa.int64()),
    ("name", pa.string()),
    ("score", pa.int64()),
    ("strand", pa.string()),
    ("thick_start", pa.int64()),
    ("thick_end", pa.int64()),
    ("item_rgb", pa.string()),
    ("block_count", pa.int64()),
    ("block_sizes", pa.string()),
    ("block_starts", pa.string()),
]

# Lines that carry display instructions for a genome browser rather than data. `track` and
# `browser` are part of the BED specification and appear *between* data blocks, not only at
# the top, which is why they are filtered per line rather than skipped as a prefix.
_COMMENT_PREFIXES = ("#", "track", "browser")


def _is_comment(line: str) -> bool:
    return line.startswith(_COMMENT_PREFIXES)


@SOURCES.register("bed")
class BedSource(FileSource):
    """BED interval files as rows, with the standard column names for the file's width.

    The width is read from the first data line: a BED3 file yields three columns, a BED12
    yields twelve. Reading a directory of mixed widths therefore produces files with
    different schemas — use `schema_mode="union"` to reconcile them, which is the general
    mechanism and not something this format should solve for itself.
    """

    # `.bed` plus the two compressed spellings a browser track is usually shipped as. The
    # base class decompresses by suffix, so they need no separate path here.
    suffix = (".bed", ".bed.gz", ".bedgraph")
    format_name = "bed"

    def _columns_for(self, width: int) -> list[str]:
        if not 3 <= width <= len(BED_COLUMNS):
            raise FormatError(
                f"bed: a record has {width} column(s); BED requires 3 to "
                f"{len(BED_COLUMNS)} (chrom, start, end, then the optional ones)."
            )
        return [name for name, _ in BED_COLUMNS[:width]]

    def _detect_width(self, fh: IO[Any]) -> int:
        """The column count of the first data line, which fixes the file's schema."""
        for raw in iter_decoded_lines(fh):
            line = raw.rstrip("\r")
            if line and not _is_comment(line):
                return len(line.split("\t"))
        # A file with no data lines still has a schema; BED3 is the minimum, and it is what
        # every wider file starts with, so a downstream union widens rather than conflicts.
        return 3

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        types = dict(BED_COLUMNS)
        return pa.schema([pa.field(n, types[n]) for n in self._columns_for(self._detect_width(fh))])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        # The width has to be known before the first block is parsed, and the handle is
        # already positioned at the start, so it is detected on a separate pass over the
        # first line rather than by peeking. `_open` hands back a seekable handle.
        width = self._detect_width(fh)
        fh.seek(0)
        names = self._columns_for(width)
        yield from _tsv.iter_record_batches(
            fh,
            is_comment=_is_comment,
            names=names,
            types=dict(BED_COLUMNS),
            null_values=NULL_VALUES,
            projection=projection,
        )


@SINKS.register("bed")
class BedSink(FileSink):
    """Write interval rows back out as BED, in the specification's column order.

    Only the leading run of standard columns present in the table is written: a table with
    `chrom/start/end/name` writes BED4, and one that skips `name` but has `strand` writes
    BED3, because BED is positional and a gap cannot be expressed. That truncation is
    reported rather than silent.
    """

    suffix = ".bed"
    format_name = "bed"

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        present = set(table.column_names)
        missing = [n for n, _ in BED_COLUMNS[:3] if n not in present]
        if missing:
            raise FormatError(
                f"bed write: the table must have {missing} column(s); got "
                f"{table.column_names}. BED is positional: chrom, start, end come first."
            )
        # The leading run only — BED has no way to say "column 4 is absent but column 6 is
        # present", so writing a gap would shift every later field into the wrong position.
        names: list[str] = []
        for name, _ in BED_COLUMNS:
            if name not in present:
                break
            names.append(name)
        # A null in an optional field becomes `.`, the specification's marker, rather than
        # an empty field — an empty field would leave two adjacent tabs, which some readers
        # treat as a column count change.
        _tsv.write_rows(fh, table, names)
