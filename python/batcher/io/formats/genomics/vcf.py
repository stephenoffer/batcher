"""VCF format — variant calls, the output every variant caller agrees to write.

A VCF is a `##` metadata block, then exactly one `#CHROM` header line naming the columns, then
tab-separated records. The first eight columns are fixed; a file with genotypes adds `FORMAT`
and one column per sample, and those sample names are data — they come from the header line,
not from the specification — so the schema is read from the file rather than declared here.

`INFO` and the per-sample genotype columns arrive as **raw text**. Both are nested key-value
encodings whose keys are declared in the `##INFO` / `##FORMAT` metadata and differ per caller,
per pipeline, and per row. Exploding them into columns would mean either a schema that changes
between files or a `Map` whose values are all strings anyway; the honest shape is the text plus
the engine's string vocabulary:

    col("info").str.regexp_extract(r"AF=([0-9.]+)", 1).cast("float64")   # allele frequency
    col("info").str.contains("DB")                                        # a dbSNP membership flag

**Coordinates are 1-based**, like GFF and unlike BED. Read exactly as written.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base import FileSource
from batcher.io.base._lines import iter_decoded_lines
from batcher.io.formats.base import SOURCES
from batcher.io.formats.genomics import _tsv
from batcher.io.formats.genomics._tsv import NULL_VALUES

__all__ = ["VCF_FIXED_COLUMNS", "VcfSource"]

#: The eight fixed columns every VCF carries, lower-cased from the header's `#CHROM POS ID
#: REF ALT QUAL FILTER INFO`. Lower case because every other column in this engine is lower
#: case, and `chrom`/`pos` are what a reader coming from `pysam` or `cyvcf2` already types.
VCF_FIXED_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("chrom", pa.string()),
    ("pos", pa.int64()),
    # An unnamed variant is `.`, which is a genuine absence rather than the string ".".
    ("id", pa.string()),
    ("ref", pa.string()),
    # Multiple alternate alleles are comma-separated in one field; kept as written, because
    # splitting them would multiply rows and change what a record means.
    ("alt", pa.string()),
    ("qual", pa.float64()),
    ("filter", pa.string()),
    ("info", pa.string()),
]

_FIXED_NAMES = [n for n, _ in VCF_FIXED_COLUMNS]
_HEADER_PREFIX = "#CHROM"


def _is_comment(line: str) -> bool:
    return line.startswith("#")


def _sanitize(name: str) -> str:
    """Make a sample name usable as a column name without losing which sample it is."""
    return name.strip() or "sample"


@SOURCES.register("vcf")
class VcfSource(FileSource):
    """VCF variant files as rows, with one column per sample when the file carries genotypes.

    The schema is read from the `#CHROM` header line, because the sample names are data. A
    sites-only VCF (no genotypes) yields the eight fixed columns; a joint-called cohort yields
    those plus `format` and one string column per sample.

    Splits are whole files, which suits how VCFs are delivered — per chromosome, per cohort
    shard — and is required in any case: a `#` is legal inside an `INFO` field, so a record
    boundary is not recoverable from a byte offset.
    """

    suffix = (".vcf", ".vcf.gz", ".bcf")
    format_name = "vcf"

    def _header_columns(self, fh: IO[Any]) -> list[str]:
        """The column names from the `#CHROM` line: the eight fixed ones, then the samples."""
        for raw in iter_decoded_lines(fh):
            line = raw.rstrip("\r")
            if line.startswith(_HEADER_PREFIX):
                fields = line.lstrip("#").split("\t")
                if len(fields) < len(_FIXED_NAMES):
                    raise FormatError(
                        f"vcf: the #CHROM header names {len(fields)} column(s); a VCF has at "
                        f"least the {len(_FIXED_NAMES)} fixed ones."
                    )
                # Positions 0-7 are the specification's, whatever the header spells them; the
                # rest are `FORMAT` and the sample names, which are data.
                #
                # `FORMAT` is lower-cased with the other specification columns, but a *sample*
                # name is not: it identifies a person or a library and is matched against a
                # manifest elsewhere, so case-folding it would quietly merge `NA12878` with a
                # cohort that spells it differently.
                extra = [
                    "format" if i == 0 and f.strip().upper() == "FORMAT" else _sanitize(f)
                    for i, f in enumerate(fields[len(_FIXED_NAMES) :])
                ]
                return _FIXED_NAMES + extra
            if not line.startswith("#") and line:
                # Data before any header. The fixed columns are still knowable, so this reads
                # rather than failing — a sites-only fragment cut out of a larger file is a
                # real thing to be handed.
                break
        return list(_FIXED_NAMES)

    def _types_for(self, names: list[str]) -> dict[str, pa.DataType]:
        types = dict(VCF_FIXED_COLUMNS)
        # `format` and every sample column are the raw genotype text; see the module note.
        for name in names[len(_FIXED_NAMES) :]:
            types[name] = pa.string()
        return types

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        names = self._header_columns(fh)
        types = self._types_for(names)
        return pa.schema([pa.field(n, types[n]) for n in names])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        # The sample names come from the header, so they must be read before the first data
        # block is parsed. `_open` hands back a seekable handle, so this is one extra pass
        # over the metadata block rather than a buffered peek.
        names = self._header_columns(fh)
        fh.seek(0)
        yield from _tsv.iter_record_batches(
            fh,
            is_comment=_is_comment,
            names=names,
            types=self._types_for(names),
            null_values=NULL_VALUES,
            projection=projection,
        )
