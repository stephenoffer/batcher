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
from itertools import chain
from typing import IO, Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import FormatError
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.genomics import _tsv
from batcher.io.formats.genomics._blocks import iter_blocks, lines_of
from batcher.io.formats.genomics._tsv import NULL_VALUES

__all__ = ["BEDGRAPH_COLUMNS", "BED_COLUMNS", "BedSink", "BedSource"]

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

#: bedGraph's four columns. It shares BED's first three and its coordinate convention, but
#: the fourth column is a measured *value* (coverage, signal), not a feature name — read as
#: BED4 it arrives as the string `"0.5"` in a column called `name`.
BEDGRAPH_COLUMNS: list[tuple[str, pa.DataType]] = [*BED_COLUMNS[:3], ("value", pa.float64())]

# The suffixes that declare a file bedGraph without a `track type=bedGraph` line.
_BEDGRAPH_SUFFIXES = (".bedgraph", ".bedgraph.gz", ".bdg", ".bdg.gz")

# `track` and `browser` lines carry display instructions for a genome browser rather than
# data, and appear *between* data blocks, not only at the top, so they are filtered per line.
# Matched as a whole word followed by a space (or alone): a prefix match dropped every record
# on a contig whose name merely starts with "track", such as `trackchr1`.
_DIRECTIVE = r"^(?:track|browser)(?: |$)"
_BEDGRAPH_TRACK = r"^track .*\btype=bedGraph\b"


def _is_comment(lines: pa.Array) -> pa.Array:
    comment = _tsv.hash_comment(lines)
    # The prefix test is a memcmp; the regex that confirms a whole word only runs on a block
    # that has a candidate, which on real files is the first block at most.
    candidate = pc.or_(pc.starts_with(lines, "track"), pc.starts_with(lines, "browser"))
    if not pc.any(candidate).as_py():
        return comment
    return pc.or_(comment, pc.match_substring_regex(lines, _DIRECTIVE))


class _Named:
    """A read handle that remembers the path it was opened from.

    The read hooks receive a handle, not a path, and a bedGraph is declared by its extension
    as often as by a `track` line — so the one fact the path carries rides along with it.
    """

    __slots__ = ("_fh", "bedgraph")

    def __init__(self, fh: Any, path: str) -> None:
        self._fh = fh
        self.bedgraph = path.lower().endswith(_BEDGRAPH_SUFFIXES)

    def read(self, size: int = -1) -> bytes:
        return self._fh.read(size)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> _Named:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@SOURCES.register("bed")
class BedSource(FileSource):
    """BED interval files as rows, with the standard column names for the file's width.

    The width is read from the first data line: a BED3 file yields three columns, a BED12
    yields twelve. Reading a directory of mixed widths therefore produces files with
    different schemas — use `schema_mode="union"` to reconcile them, which is the general
    mechanism and not something this format should solve for itself.

    A bedGraph — declared by a `track type=bedGraph` line or a `.bedgraph` / `.bdg` suffix —
    reads as `chrom/start/end/value` with a float `value`.
    """

    # `.bed` and bedGraph, plain or gzipped. The base class decompresses by suffix, so the
    # compressed spellings need no separate path here.
    suffix = (".bed", ".bed.gz", *_BEDGRAPH_SUFFIXES)
    format_name = "bed"

    def _open(self, path: str) -> Any:
        return _Named(super()._open(path), path)

    def _layout(self, fh: Any) -> tuple[list[tuple[str, pa.DataType]], Iterator[bytes]]:
        """The file's columns, and its blocks from the one holding the first data line on.

        One pass: the header region is scanned block by block until the first data line,
        whose block is handed back for the parse. The previous version read the first line
        and then `seek(0)`-ed to re-read, which a decompressing stream refuses — so every
        `.bed.gz` failed with "only valid on seekable files".
        """
        bedgraph = bool(getattr(fh, "bedgraph", False))
        blocks = iter_blocks(fh)
        for data in blocks:
            lines = lines_of(data)
            is_data = pc.invert(pc.or_(pc.equal(pc.binary_length(lines), 0), _is_comment(lines)))
            at = _tsv.first_index(is_data)
            head = lines if at < 0 else lines.slice(0, at)
            bedgraph = bedgraph or pc.any(pc.match_substring_regex(head, _BEDGRAPH_TRACK)).as_py()
            if at < 0:
                continue
            width = len(lines[at].as_py().split("\t"))
            return self._columns_for(width, bedgraph=bool(bedgraph)), chain([data], blocks)
        # A file with no data lines still has a schema; BED3 is the minimum, and it is what
        # every wider file starts with, so a downstream union widens rather than conflicts.
        return (BEDGRAPH_COLUMNS if bedgraph else BED_COLUMNS[:3]), iter(())

    def _columns_for(self, width: int, *, bedgraph: bool) -> list[tuple[str, pa.DataType]]:
        if bedgraph:
            if width != len(BEDGRAPH_COLUMNS):
                raise FormatError(
                    f"bed: a bedGraph record has {width} column(s); bedGraph has exactly 4 "
                    "(chrom, start, end, value)."
                )
            return BEDGRAPH_COLUMNS
        if not 3 <= width <= len(BED_COLUMNS):
            raise FormatError(
                f"bed: a record has {width} column(s); BED requires 3 to "
                f"{len(BED_COLUMNS)} (chrom, start, end, then the optional ones)."
            )
        return BED_COLUMNS[:width]

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        columns, _ = self._layout(fh)
        return pa.schema([pa.field(n, t) for n, t in columns])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        columns, blocks = self._layout(fh)
        yield from _tsv.iter_record_batches(
            blocks,
            markers=(b"#", b"track", b"browser"),
            is_comment=_is_comment,
            names=[n for n, _ in columns],
            types=dict(columns),
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
