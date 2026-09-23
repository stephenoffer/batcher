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

from collections import Counter
from collections.abc import Iterator
from itertools import chain
from typing import IO, Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import FormatError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.io.formats.genomics import _tsv
from batcher.io.formats.genomics._blocks import iter_blocks, lines_of
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

# The first bytes of a file that is not VCF text. BCF is VCF's binary encoding: raw it
# starts `BCF` plus a version byte, and as shipped it is BGZF — gzip — under a `.bcf` name
# that no suffix rule decompresses. Either way a text reader fails on the first byte that
# is not UTF-8, which named the codec rather than the problem.
_BCF_MAGIC = b"BCF"
_GZIP_MAGIC = b"\x1f\x8b"


def _sample_columns(fields: list[str]) -> list[str]:
    """`FORMAT` and the sample names from a `#CHROM` line, as distinct column names.

    `FORMAT` is lower-cased with the other specification columns, but a *sample* name is
    not: it identifies a person or a library and is matched against a manifest elsewhere, so
    case-folding it would quietly merge `NA12878` with a cohort that spells it differently.

    Two samples with one name are refused — the specification requires them to be unique,
    and renaming one would attach a genotype to a sample that does not exist. A sample whose
    name collides with a specification column (a sample called `pos`) is kept under
    `sample_<name>`, because the collision is this reader's lower-casing, not the file's.
    """
    extra = fields[len(_FIXED_NAMES) :]
    names: list[str] = []
    if extra and extra[0].strip().upper() == "FORMAT":
        names.append("format")
        extra = extra[1:]
    samples = [f.strip() or "sample" for f in extra]
    duplicates = sorted(name for name, count in Counter(samples).items() if count > 1)
    if duplicates:
        raise FormatError(
            f"vcf: the #CHROM header names sample(s) {duplicates} more than once; VCF sample "
            "names must be unique, and picking one column would attribute its genotypes to "
            "the wrong sample."
        )
    reserved = set(_FIXED_NAMES) | {"format"}
    for sample in samples:
        name = f"sample_{sample}" if sample in reserved else sample
        if name != sample and name in samples:
            raise FormatError(
                f"vcf: sample {sample!r} collides with a specification column, and its "
                f"renamed form {name!r} is itself a sample name in this file."
            )
        names.append(name)
    return names


def _refuse_binary(first: bytes) -> None:
    """Raise a `FormatError` naming BCF when the file's first bytes are not VCF text."""
    if first.startswith(_BCF_MAGIC) or first.startswith(_GZIP_MAGIC):
        raise FormatError(
            "vcf: this file is binary — BCF, or BGZF-compressed data under a name that does "
            "not end in .gz. BCF is not supported; convert it with `bcftools view -Ov` (or "
            "`-Oz` to a .vcf.gz). A bgzipped VCF reads as-is once it is named `.vcf.gz`."
        )


@SOURCES.register("vcf")
class VcfSource(FileSource):
    """VCF variant files as rows, with one column per sample when the file carries genotypes.

    The schema is read from the `#CHROM` header line, because the sample names are data. A
    sites-only VCF (no genotypes) yields the eight fixed columns; a joint-called cohort yields
    those plus `format` and one string column per sample.

    `.vcf` and bgzipped `.vcf.gz` are read. BCF, the binary encoding, is refused with an
    error that says so. There is no VCF writer: a VCF's `##` metadata block declares every
    `INFO` and `FORMAT` key, and a table does not carry it.

    Splits are whole files, which suits how VCFs are delivered — per chromosome, per cohort
    shard — and is required in any case: a `#` is legal inside an `INFO` field, so a record
    boundary is not recoverable from a byte offset.
    """

    suffix = (".vcf", ".vcf.gz")
    format_name = "vcf"

    def _layout(self, fh: IO[Any]) -> tuple[list[str], Iterator[bytes]]:
        """The column names, and the file's blocks from the one holding the first data line on.

        One pass over the header region. The previous version read the header and then
        `seek(0)`-ed to re-read the file, which a decompressing stream refuses — so every
        `.vcf.gz` failed with "only valid on seekable files".
        """
        first = fh.read(4)
        _refuse_binary(first)
        blocks = iter_blocks(fh, first)
        header: str | None = None
        for block in blocks:
            lines = lines_of(block)
            data = pc.invert(pc.or_(pc.equal(pc.binary_length(lines), 0), _tsv.hash_comment(lines)))
            at = _tsv.first_index(data)
            head = lines if at < 0 else lines.slice(0, at)
            if header is None:
                found = _tsv.first_index(pc.starts_with(head, _HEADER_PREFIX))
                header = None if found < 0 else head[found].as_py()
            if at >= 0:
                # Data before any header still has the eight fixed columns — a sites-only
                # fragment cut out of a larger file is a real thing to be handed.
                return self._names(header), chain([block], blocks)
        return self._names(header), iter(())

    @staticmethod
    def _names(header: str | None) -> list[str]:
        if header is None:
            return list(_FIXED_NAMES)
        fields = header.lstrip("#").split("\t")
        if len(fields) < len(_FIXED_NAMES):
            raise FormatError(
                f"vcf: the #CHROM header names {len(fields)} column(s); a VCF has at "
                f"least the {len(_FIXED_NAMES)} fixed ones."
            )
        # Positions 0-7 are the specification's, whatever the header spells them; the rest
        # are `FORMAT` and the sample names, which are data.
        return _FIXED_NAMES + _sample_columns(fields)

    def _types_for(self, names: list[str]) -> dict[str, pa.DataType]:
        types = dict(VCF_FIXED_COLUMNS)
        # `format` and every sample column are the raw genotype text; see the module note.
        for name in names[len(_FIXED_NAMES) :]:
            types[name] = pa.string()
        return types

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        names, _ = self._layout(fh)
        types = self._types_for(names)
        return pa.schema([pa.field(n, types[n]) for n in names])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        return list(self._iter_records(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._open(path) as fh:
            yield from self._iter_records(fh, projection)

    def _iter_records(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        names, blocks = self._layout(fh)
        yield from _tsv.iter_record_batches(
            blocks,
            markers=(b"#",),
            is_comment=_tsv.hash_comment,
            names=names,
            types=self._types_for(names),
            null_values=NULL_VALUES,
            projection=projection,
        )
