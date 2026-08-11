"""The BED, GFF/GTF, and VCF readers — intervals, annotations, and variants.

These three are comment-carrying TSVs, and each carries one thing a plain CSV read gets
wrong. BED has no header and a width the file chooses between 3 and 12. GFF spells every
absent value ``.``, which a naive read turns into the string "." on a float column or a parse
error. VCF's column list is partly *data*: the sample names come from its ``#CHROM`` line.

The coordinate conventions are pinned here too, deliberately. BED is 0-based half-open and
GFF/VCF are 1-based inclusive, and nothing in this engine normalizes between them — a silent
shift is the single most common off-by-one in genomics, and a test that asserts the numbers
come back exactly as written is what keeps a well-meaning "fix" out.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.formats.base import SINKS, SOURCES

pytestmark = pytest.mark.io

_BED6 = (
    "track name=demo\n"
    "#a comment between blocks\n"
    "chr1\t0\t100\tgeneA\t900\t+\n"
    "browser position chr1\n"
    "chr1\t150\t250\tgeneB\t500\t-\n"
)
_GFF = (
    "##gff-version 3\n"
    "##sequence-region chr1 1 1000\n"
    "chr1\tHAVANA\tgene\t11\t200\t.\t+\t.\tID=g1;Name=BRCA1\n"
    "chr1\tHAVANA\tCDS\t20\t100\t0.9\t+\t0\tID=c1;Parent=g1\n"
)
_VCF = (
    "##fileformat=VCFv4.2\n"
    "##INFO=<ID=AF,Number=A,Type=Float>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA12878\tNA12891\n"
    "chr1\t100\trs1\tA\tG\t50.5\tPASS\tAF=0.25;DB\tGT:DP\t0/1:30\t1/1:22\n"
    "chr1\t200\t.\tC\tT,A\t.\tq10\tAF=0.01\tGT:DP\t0/0:15\t0/1:19\n"
)


def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --- BED -----------------------------------------------------------------------------


def test_bed_names_the_columns_its_width_implies(tmp_path):
    out = bt.read.bed(_write(tmp_path, "x.bed", _BED6)).to_pydict()
    assert out == {
        "chrom": ["chr1", "chr1"],
        "start": [0, 150],
        "end": [100, 250],
        "name": ["geneA", "geneB"],
        "score": [900, 500],
        "strand": ["+", "-"],
    }


def test_bed3_is_three_columns_not_twelve_padded_with_nulls(tmp_path):
    """The width is the file's own statement, so a BED3 file has three columns."""
    ds = bt.read.bed(_write(tmp_path, "m.bed", "chr1\t0\t100\nchr2\t5\t9\n"))
    assert ds.schema.names == ["chrom", "start", "end"]


def test_bed_skips_track_and_browser_lines_wherever_they_appear(tmp_path):
    """They are display directives and the specification allows them *between* blocks."""
    assert bt.read.bed(_write(tmp_path, "t.bed", _BED6)).count() == 2


def test_bed_coordinates_are_read_exactly_as_written(tmp_path):
    """0-based, half-open. A helpful +1 here would silently disagree with every other tool."""
    out = bt.read.bed(_write(tmp_path, "c.bed", "chr1\t0\t100\n")).to_pydict()
    assert out["start"] == [0]
    assert out["end"] == [100]


def test_bed_round_trips(tmp_path):
    ds = bt.read.bed(_write(tmp_path, "x.bed", _BED6))
    out = str(tmp_path / "o.bed")
    ds.write.bed(out)
    assert bt.read.bed(out).to_pydict() == ds.to_pydict()


def test_bed_write_stops_at_the_first_absent_column(tmp_path):
    """BED is positional, so a gap would shift every later field into the wrong place."""
    ds = bt.from_pydict(
        {"chrom": ["chr1"], "start": [0], "end": [9], "strand": ["+"]}  # no `name`/`score`
    )
    out = str(tmp_path / "g.bed")
    ds.write.bed(out)
    assert Path(out).read_text() == "chr1\t0\t9\n"


def test_bed_write_requires_the_three_positional_columns(tmp_path):
    with pytest.raises(FormatError, match="chrom, start, end"):
        bt.from_pydict({"start": [0], "end": [1]}).write.bed(str(tmp_path / "b.bed"))


# --- GFF / GTF -----------------------------------------------------------------------


def test_gff_reads_nine_columns_with_dot_as_null(tmp_path):
    out = bt.read.gff(_write(tmp_path, "y.gff3", _GFF)).to_pydict()
    assert out["type"] == ["gene", "CDS"]
    assert out["start"] == [11, 20]
    # `.` is an absent score and an absent phase, not the string ".".
    assert out["score"] == [None, 0.9]
    assert out["phase"] == [None, 0]


def test_gff_keeps_attributes_as_text_so_either_dialect_reads(tmp_path):
    """GFF3 and GTF differ only here, and the dialect is not recoverable from the file."""
    gtf = 'chr1\tX\tgene\t1\t9\t.\t+\t.\tgene_id "g1"; gene_name "BRCA1";\n'
    ds = bt.read.gff(_write(tmp_path, "a.gtf", gtf))
    assert ds.schema.field("attributes").type == pa.string()
    # The engine's own string vocabulary pulls a key out of either encoding.
    got = ds.select(g=bt.col("attributes").str.regexp_extract(r'gene_id "([^"]+)"', 1))
    assert got.to_pydict()["g"] == ["g1"]


def test_gff_coordinates_are_one_based_and_read_as_written(tmp_path):
    """The opposite convention to BED, and equally untouched."""
    out = bt.read.gff(_write(tmp_path, "p.gff3", _GFF)).to_pydict()
    assert out["start"][0] == 11  # not 10, and not 12


def test_gff_round_trips(tmp_path):
    ds = bt.read.gff(_write(tmp_path, "y.gff3", _GFF))
    out = str(tmp_path / "o.gff3")
    ds.write.gff(out)
    assert bt.read.gff(out).to_pydict() == ds.to_pydict()
    assert Path(out).read_text().startswith("##gff-version 3\n")


def test_gff_write_requires_all_nine_columns(tmp_path):
    with pytest.raises(FormatError, match="positional"):
        bt.from_pydict({"seqid": ["c"]}).write.gff(str(tmp_path / "g.gff3"))


# --- VCF -----------------------------------------------------------------------------


def test_vcf_takes_its_sample_columns_from_the_header(tmp_path):
    """The column list is partly data: a cohort's sample names live in the `#CHROM` line."""
    ds = bt.read.vcf(_write(tmp_path, "z.vcf", _VCF))
    assert ds.schema.names == [
        "chrom", "pos", "id", "ref", "alt", "qual", "filter", "info",
        "format", "NA12878", "NA12891",
    ]  # fmt: skip


def test_vcf_lower_cases_the_specification_columns_but_not_a_sample_name(tmp_path):
    """A sample name identifies a library and is matched against a manifest — don't fold it."""
    ds = bt.read.vcf(_write(tmp_path, "z.vcf", _VCF))
    assert "format" in ds.schema.names
    assert "NA12878" in ds.schema.names


def test_vcf_reads_dot_as_null_and_keeps_a_multiallelic_alt_intact(tmp_path):
    out = bt.read.vcf(_write(tmp_path, "z.vcf", _VCF)).to_pydict()
    assert out["id"] == ["rs1", None]
    assert out["qual"] == [50.5, None]
    # Splitting `T,A` into two rows would change what the record means.
    assert out["alt"] == ["G", "T,A"]


def test_vcf_info_is_queryable_with_the_string_vocabulary(tmp_path):
    ds = bt.read.vcf(_write(tmp_path, "z.vcf", _VCF))
    af = ds.select(
        af=bt.col("info").str.regexp_extract(r"AF=([0-9.]+)", 1).cast("float64")
    ).to_pydict()["af"]
    assert af == [0.25, 0.01]


def test_a_sites_only_vcf_yields_the_eight_fixed_columns(tmp_path):
    text = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tT\t.\t.\t.\n"
    )
    ds = bt.read.vcf(_write(tmp_path, "s.vcf", text))
    assert ds.schema.names == [
        "chrom", "pos", "id", "ref", "alt", "qual", "filter", "info",
    ]  # fmt: skip
    assert ds.count() == 1


# --- Contracts every format owes -----------------------------------------------------


_CASES = [("bed", _BED6), ("gff", _GFF), ("vcf", _VCF)]


@pytest.mark.parametrize(("name", "text"), _CASES)
def test_an_empty_file_still_reports_a_schema(tmp_path, name, text):
    ds = bt.read(_write(tmp_path, f"e.{name}", ""), format=name)
    assert len(ds.schema.names) >= 3
    assert ds.count() == 0


@pytest.mark.parametrize(("name", "text"), _CASES)
def test_projection_narrows_the_batch(tmp_path, name, text):
    ds = bt.read(_write(tmp_path, f"p.{name}", text), format=name)
    first = ds.schema.names[0]
    assert list(ds.select(first).to_pydict()) == [first]


@pytest.mark.parametrize(("name", "text"), _CASES)
def test_iter_batches_streams_the_same_rows_collect_returns(tmp_path, name, text):
    ds = bt.read(_write(tmp_path, f"s.{name}", text), format=name)
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=ds.schema)
    assert streamed.to_pydict() == ds.to_pydict()


@pytest.mark.parametrize(("name", "text"), _CASES)
def test_splits_cover_the_source_exactly_once_and_survive_a_pickle(tmp_path, name, text):
    for i in range(3):
        _write(tmp_path, f"f{i}.{name}", text)
    source = SOURCES.get(name)(str(tmp_path))
    key = source.schema().names[0]
    whole = [v for b in source.read() for v in b.column(key).to_pylist()]
    from_splits: list[str] = []
    for split in source.splits():
        revived = pickle.loads(pickle.dumps(split))
        for batch in revived.read():
            from_splits += batch.column(key).to_pylist()
    assert sorted(from_splits) == sorted(whole)
    assert len(whole) == 6  # two records per file, three files


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".bed", "bed"),
        (".bedgraph", "bed"),
        (".gff", "gff"),
        (".gff3", "gff"),
        (".gtf", "gff"),
        (".vcf", "vcf"),
    ],
)
def test_the_extension_infers_the_format(suffix, expected):
    from batcher.io.detect import detect_format

    assert detect_format(f"sample{suffix}") == expected


def test_the_readers_are_registered_and_the_writers_where_they_exist():
    for name in ("bed", "gff", "vcf"):
        assert name in SOURCES.names()
    for name in ("bed", "gff"):
        assert name in SINKS.names()
    # VCF has no sink: a valid VCF needs the `##` metadata block declaring every INFO and
    # FORMAT key the records use, and that cannot be reconstructed from the columns alone.
    # Writing one without it would produce a file that parses and that no caller can
    # interpret, so the format is deliberately read-only.
    assert "vcf" not in SINKS.names()


def test_an_interval_join_is_the_engines_range_join(tmp_path):
    """The payoff for reading these into tables: overlap is a relational query.

    BED intervals against GFF features, matched on containment. This is what makes "which
    annotations cover my regions" a join rather than a script — and it is why the readers do
    not need interval logic of their own.
    """
    regions = bt.read.bed(_write(tmp_path, "r.bed", "chr1\t50\t120\tr1\t0\t+\n"))
    genes = bt.read.gff(_write(tmp_path, "g.gff3", _GFF))
    hits = (
        regions.join(genes, left_on="chrom", right_on="seqid", how="inner")
        .filter((bt.col("start_right") <= bt.col("end")) & (bt.col("end_right") >= bt.col("start")))
        .select("name", "type")
        .to_pydict()
    )
    # The r1 region (50-120) overlaps the gene (11-200) and the CDS (20-100).
    assert sorted(hits["type"]) == ["CDS", "gene"]


# --- Fault tolerance -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "good", "corrupt"),
    [
        # A VCF whose header promises eleven columns and whose record supplies four.
        (
            "vcf",
            _VCF,
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\tnot_a_position\tA\tG\n",
        ),
        # A GFF whose start column is not a number.
        ("gff", _GFF, "##gff-version 3\nchr1\tX\tgene\tNOT_A_NUMBER\t9\t.\t+\t.\tID=g\n"),
        # A BED whose second record has fewer columns than the first declared.
        ("bed", _BED6, "chr1\t0\t100\tg\t900\t+\nchr1\tbroken\n"),
    ],
)
def test_a_corrupt_file_is_skipped_rather_than_aborting_the_corpus(tmp_path, name, good, corrupt):
    """`on_error="skip"` drops the unreadable file and reads the rest.

    A corpus at scale always contains a few unreadable members — a truncated upload, an
    interrupted write — and aborting a 10,000-file scan for one of them is the wrong default.
    The policy is the shared one in `io/base/_tolerance.py`, and this checks that the
    genomics readers actually route through it rather than raising past it.
    """
    _write(tmp_path, f"good1.{name}", good)
    _write(tmp_path, f"bad.{name}", corrupt)
    _write(tmp_path, f"good2.{name}", good)

    with pytest.raises(Exception):  # noqa: B017 (the format decides which error it raises)
        bt.read(str(tmp_path), format=name).collect()

    kept = bt.read(str(tmp_path), format=name, on_error="skip").collect()
    assert kept.num_rows == 4, "both good files' records, and none of the bad file's"


def test_tolerance_is_per_file_not_per_record(tmp_path):
    """The documented granularity, pinned so a reader knows what `skip` costs.

    One bad record loses its **whole file**, not just that record. For FASTQ that is worth
    knowing before relying on it: a truncated final record is the commonest corruption there
    is, and a 50 GB run file is an expensive thing to drop for one of them. Convert to
    Parquet, or split the run, if that trade is wrong for you.
    """
    text = "chr1\t0\t10\tkeep_me\t0\t+\n" * 5 + "chr1\tbroken\n"
    _write(tmp_path, "partly_good.bed", text)
    assert bt.read.bed(str(tmp_path), on_error="skip").count() == 0, (
        "the five good records go with the file, not without it"
    )
