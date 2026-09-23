"""Edge cases of the genomics readers and writers, and proof the vectorized paths are exact.

Every regression here was reproduced against the previous implementation before it was
fixed: `.vcf.gz`/`.bed.gz` failing on a `seek(0)` a decompressing stream refuses, a GFF3
`##FASTA` section failing the whole file, `.bcf` advertised and then failing on the first
non-UTF-8 byte, duplicate VCF sample names and a sample called `pos` raising `KeyError`, a
BED contig called `trackchr1` silently dropped, bedGraph's value read as a string `name`, a
FASTA `;` comment glued into the sequence, and a CR-only FASTA reading as zero records.

The second half is the differential suite for the rewrite that moved these readers off a
per-line Python loop: each reader and writer is compared against a small pure-Python
reference that restates the old loop's semantics, over randomized inputs and with the block
size shrunk so records, CRLF pairs, and headers straddle block boundaries.
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.detect import detect_format
from batcher.io.formats.genomics import (
    BedSource,
    FastaSource,
    FastqSource,
    GffSource,
    VcfSource,
    _blocks,
)
from batcher.io.formats.genomics import fasta as fasta_mod
from batcher.io.formats.genomics import fastq as fastq_mod

pytestmark = pytest.mark.io

_FASTA = ">chr1 Homo sapiens\nACGT\nacgtNN\n\n>chr2\n>chr3  x\tY\nNNNN\n"
_FASTQ = "@r1 lane1\nACGT\n+\nIIII\n@r2\nAC\n+r2\n@I\n"
_GFF = (
    "##gff-version 3\n"
    "chr1\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1;Note=a#b\n"
    "chr1\tsrc\tCDS\t1\t99\t3.5\t+\t0\tParent=g1\n"
)
_VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1\tNA2\n"
    "1\t100\trs1\tA\tC,G\t50\tPASS\tDB;AF=0.1\tGT:DP\t0/1:3\t./.\n"
    "1\t200\t.\tT\t.\t.\t.\t.\tGT\t1|1\t.\n"
)
_BED = "browser position chr1:1-100\ntrack name=x\nchr1\t0\t10\tn1\t5\t+\nchr2\t5\t6\t.\t.\t.\n"

# name -> (reader, text, the suffix a plain file carries)
_FORMATS = {
    "fasta": (bt.read.fasta, _FASTA, ".fa"),
    "fastq": (bt.read.fastq, _FASTQ, ".fq"),
    "gff": (bt.read.gff, _GFF, ".gff3"),
    "vcf": (bt.read.vcf, _VCF, ".vcf"),
    "bed": (bt.read.bed, _BED, ".bed"),
}


def _write(tmp_path: Path, name: str, data: str | bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return str(path)


def _rows(ds) -> list[dict]:
    return ds.collect().to_pylist()


# --- compression ---------------------------------------------------------------------


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_every_format_reads_gzipped(tmp_path, fmt):
    """`.gz` reads the same rows as the plain file, through both read paths."""
    reader, text, suffix = _FORMATS[fmt]
    plain = _rows(reader(_write(tmp_path, f"a{suffix}", text)))
    packed = _write(tmp_path, f"a{suffix}.gz", gzip.compress(text.encode()))
    assert plain  # the fixture has rows, so the comparison below is not vacuous
    assert _rows(reader(packed)) == plain
    streamed = pa.Table.from_batches(list(reader(packed).iter_batches())).to_pylist()
    assert streamed == plain


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_a_directory_of_gzipped_files_is_listed(tmp_path, fmt):
    """The suffix list covers `.gz`, so a directory read finds compressed files."""
    reader, text, suffix = _FORMATS[fmt]
    (tmp_path / "d").mkdir()
    _write(tmp_path / "d", f"a{suffix}.gz", gzip.compress(text.encode()))
    _write(tmp_path / "d", f"b{suffix}", text)
    plain = _rows(reader(_write(tmp_path, f"one{suffix}", text)))
    assert len(_rows(reader(str(tmp_path / "d")))) == 2 * len(plain)


def test_a_bgzipped_vcf_reads_across_gzip_members(tmp_path):
    """Real `.vcf.gz` is BGZF: many concatenated gzip members, not one stream."""
    body = _VCF.encode()
    members = b"".join(gzip.compress(body[i : i + 40]) for i in range(0, len(body), 40))
    path = _write(tmp_path, "cohort.vcf.gz", members)
    assert _rows(bt.read.vcf(path)) == _rows(bt.read.vcf(_write(tmp_path, "c.vcf", _VCF)))


# --- line endings --------------------------------------------------------------------


@pytest.mark.parametrize("ending", ["\r\n", "\r"], ids=["crlf", "cr"])
@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_crlf_and_cr_only_read_like_lf(tmp_path, fmt, ending):
    reader, text, suffix = _FORMATS[fmt]
    expected = _rows(reader(_write(tmp_path, f"lf{suffix}", text)))
    got = _rows(reader(_write(tmp_path, f"x{suffix}", text.replace("\n", ending))))
    assert expected and got == expected


def test_a_cr_only_fasta_is_not_zero_records(tmp_path):
    rows = _rows(bt.read.fasta(_write(tmp_path, "mac.fa", ">a\rAC\rGT\r>b\rTT\r")))
    assert rows == [
        {"id": "a", "description": "", "sequence": "ACGT"},
        {"id": "b", "description": "", "sequence": "TT"},
    ]


# --- FASTA ---------------------------------------------------------------------------


def test_a_fasta_semicolon_line_is_a_comment_even_inside_a_record(tmp_path):
    rows = _rows(bt.read.fasta(_write(tmp_path, "c.fa", ";top\n>a\nAC\n;note\nGT\n")))
    assert rows == [{"id": "a", "description": "", "sequence": "ACGT"}]


def test_fasta_write_refuses_a_sequence_with_a_line_break(tmp_path):
    ds = bt.from_pydict({"id": ["a"], "sequence": ["AC\nGT"]})
    with pytest.raises(FormatError, match="line break"):
        ds.write.fasta(str(tmp_path / "out.fasta"))


# --- GFF -----------------------------------------------------------------------------


def test_gff_stops_at_the_fasta_directive(tmp_path):
    text = _GFF + "##FASTA\n>chr1\nACGTACGT\n"
    rows = _rows(bt.read.gff(_write(tmp_path, "a.gff3", text)))
    assert [r["type"] for r in rows] == ["gene", "CDS"]


def test_gff_stops_at_a_bare_fasta_header(tmp_path):
    """GFF3 forbids a seqid starting with `>`, so such a line can only start sequences."""
    rows = _rows(bt.read.gff(_write(tmp_path, "b.gff3", _GFF + ">chr1\nACGT\n")))
    assert [r["type"] for r in rows] == ["gene", "CDS"]


# --- VCF -----------------------------------------------------------------------------


def _vcf_with_samples(*samples: str) -> str:
    header = "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"])
    row = "\t".join(["1", "5", ".", "A", "T", ".", "PASS", ".", "GT"] + ["0/1"] * len(samples))
    return header + "\t" + "\t".join(samples) + "\n" + row + "\n"


def test_duplicate_vcf_sample_names_are_refused_by_name(tmp_path):
    path = _write(tmp_path, "dup.vcf", _vcf_with_samples("S", "T", "S"))
    with pytest.raises(FormatError, match=r"\['S'\] more than once"):
        bt.read.vcf(path).collect()


def test_a_sample_named_like_a_fixed_column_is_kept_under_a_prefix(tmp_path):
    ds = bt.read.vcf(_write(tmp_path, "pos.vcf", _vcf_with_samples("pos", "NA1")))
    assert ds.columns[-3:] == ["format", "sample_pos", "NA1"]
    assert _rows(ds)[0]["pos"] == 5 and _rows(ds)[0]["sample_pos"] == "0/1"


@pytest.mark.parametrize(
    "payload",
    [b"BCF\x02\x02rest", gzip.compress(b"BCF\x02\x02rest")],
    ids=["raw-bcf", "bgzf-bcf"],
)
def test_bcf_is_refused_with_an_error_that_names_it(tmp_path, payload):
    with pytest.raises(FormatError, match="BCF is not supported"):
        bt.read.vcf(_write(tmp_path, "x.bcf", payload)).collect()


def test_bcf_is_not_advertised():
    assert ".bcf" not in VcfSource.suffix
    assert detect_format("calls.vcf.gz") == "vcf"  # the control: detection does see VCF
    with pytest.raises(FormatError, match=r"Unknown file extension '\.bcf'"):
        detect_format("calls.bcf")


# --- BED / bedGraph ------------------------------------------------------------------


def test_a_contig_whose_name_starts_with_track_is_data(tmp_path):
    text = "track name=x\ntrackchr\t0\t1\nbrowserX\t2\t3\n"
    rows = _rows(bt.read.bed(_write(tmp_path, "t.bed", text)))
    assert [r["chrom"] for r in rows] == ["trackchr", "browserX"]


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("a.bed", "track type=bedGraph name=cov\nchr1\t0\t10\t0.5\nchr1\t10\t20\t2\n"),
        ("a.bedgraph", "chr1\t0\t10\t0.5\nchr1\t10\t20\t2\n"),
        ("a.bedgraph.gz", gzip.compress(b"chr1\t0\t10\t0.5\nchr1\t10\t20\t2\n")),
    ],
    ids=["track-line", "extension", "gzipped-extension"],
)
def test_bedgraph_reads_a_float_value(tmp_path, name, text):
    ds = bt.read.bed(_write(tmp_path, name, text))
    assert ds.schema.field("value").type == pa.float64()
    assert [r["value"] for r in _rows(ds)] == [0.5, 2.0]


def test_a_plain_bed4_still_has_a_string_name(tmp_path):
    ds = bt.read.bed(_write(tmp_path, "n.bed", "chr1\t0\t10\t0.5\n"))
    assert ds.columns == ["chrom", "start", "end", "name"]
    assert _rows(ds)[0]["name"] == "0.5"


# --- differential: the vectorized paths against the old per-line semantics -----------


def _ref_split(header: str) -> tuple[str, str]:
    parts = header.split(maxsplit=1)
    return (parts[0] if parts else ""), (parts[1].strip() if len(parts) > 1 else "")


def _ref_fasta(text: str) -> list[dict]:
    """The pre-rewrite FASTA loop, plus the two fixes (`;` comments, CR-only endings)."""
    out, header, chunks = [], None, []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        elif header is not None and line and not line.startswith(";"):
            chunks.append(line)
    if header is not None:
        out.append((header, "".join(chunks)))
    return [
        {"id": _ref_split(h)[0], "description": _ref_split(h)[1], "sequence": s} for h, s in out
    ]


def _ref_fastq(text: str) -> list[dict]:
    out, window = [], []
    for line in text.replace("\r\n", "\n").split("\n")[:-1]:
        if not window and not line:
            continue
        window.append(line)
        if len(window) == 4:
            rid, desc = _ref_split(window[0][1:])
            out.append(
                {"id": rid, "description": desc, "sequence": window[1], "quality": window[3]}
            )
            window = []
    return out


_ALPHABET = "ACGTN acgt\t:=é"


def _random_fasta(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(0, 40)):
        kind = rng.random()
        if kind < 0.3:
            lines.append(
                ">" + "".join(rng.choice(_ALPHABET + " ") for _ in range(rng.randint(0, 12)))
            )
        elif kind < 0.35:
            lines.append(";" + "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 5))))
        elif kind < 0.4:
            lines.append("")
        else:
            lines.append("".join(rng.choice("ACGTNacgt") for _ in range(rng.randint(1, 30))))
    ending = rng.choice(["\n", "\r\n", "\r"])
    return ending.join(lines) + rng.choice(["", ending])


def _random_fastq(rng: random.Random) -> str:
    records = []
    for i in range(rng.randint(0, 30)):
        seq = "".join(rng.choice("ACGTN") for _ in range(rng.randint(0, 20)))
        qual = "".join(rng.choice("!@#I+5") for _ in seq)
        header = f"@r{i}" + rng.choice(["", " desc x", "\tlane:1 ", "  é u"])
        blank = "\n" if rng.random() < 0.1 else ""
        records.append(f"{header}\n{seq}\n+{rng.choice(['', 'r'])}\n{qual}\n{blank}")
    text = "".join(records)
    return text.replace("\n", "\r\n") if rng.random() < 0.3 else text


@pytest.fixture(params=[5, 64, 1 << 22], ids=["block5", "block64", "block4m"])
def block_bytes(request, monkeypatch):
    """Shrink the read block so records, CRLF pairs and headers straddle block boundaries."""
    monkeypatch.setattr(_blocks, "_BLOCK_BYTES", request.param)
    return request.param


def test_fasta_matches_the_per_line_reference(tmp_path, block_bytes):
    rng = random.Random(1234 + block_bytes)
    for i in range(60):
        text = _random_fasta(rng)
        path = _write(tmp_path, f"f{i}.fa", text)
        got = pa.Table.from_batches(FastaSource(path).read()).to_pylist()
        assert got == _ref_fasta(text), repr(text)


def test_fastq_matches_the_per_line_reference(tmp_path, block_bytes):
    rng = random.Random(99 + block_bytes)
    for i in range(60):
        text = _random_fastq(rng)
        path = _write(tmp_path, f"q{i}.fq", text)
        got = pa.Table.from_batches(FastqSource(path).read()).to_pylist()
        assert got == _ref_fastq(text), repr(text)


def _ref_tsv(text: str, is_comment) -> list[list[str]]:
    lines = text.replace("\r\n", "\n").split("\n")
    return [line.split("\t") for line in lines if line and not is_comment(line)]


@pytest.mark.parametrize("fmt", ["bed", "gff", "vcf"])
def test_tsv_formats_match_the_per_line_reference(tmp_path, block_bytes, fmt):
    """Comment and directive lines are sprinkled between records so both the clean-block
    fast path and the line-filtering path run, at every block size."""
    rng = random.Random(7 + block_bytes)
    source = {"bed": BedSource, "gff": GffSource, "vcf": VcfSource}[fmt]
    for i in range(30):
        body = []
        for j in range(rng.randint(1, 40)):
            if rng.random() < 0.15:
                body.append(rng.choice(["# note", "", "track name=t" if fmt == "bed" else "#x"]))
            if fmt == "bed":
                body.append(f"chr{j % 3}\t{j}\t{j + 5}\tn{j}")
            elif fmt == "gff":
                body.append(
                    f"c{j}\ts\tgene\t{j + 1}\t{j + 9}\t{rng.choice(['.', '1.5'])}\t+\t.\tID=g{j}"
                )
            else:
                body.append(f"{j % 2}\t{j}\t.\tA\tC\t.\tPASS\tAF=0.{j}")
        head = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        text = (head if fmt == "vcf" else "") + "\n".join(body) + "\n"
        text = text.replace("\n", "\r\n") if rng.random() < 0.3 else text
        path = _write(tmp_path, f"t{i}.{fmt}", text)
        got = pa.Table.from_batches(source(path).read())
        comment = (
            (lambda s: s.startswith(("#", "track ")))
            if fmt == "bed"
            else (lambda s: s.startswith("#"))
        )
        expected = _ref_tsv(text, comment)
        as_text = [["." if v is None else str(v) for v in row.values()] for row in got.to_pylist()]
        assert as_text == expected


def _ref_fasta_bytes(rows: list[dict]) -> bytes:
    out = []
    for row in rows:
        header = str(row["id"] or "")
        if row.get("description"):
            header = f"{header} {row['description']}"
        out.append(f">{header}\n")
        text = str(row["sequence"] or "")
        out.extend(text[i : i + 60] + "\n" for i in range(0, len(text), 60))
        if not text:
            out.append("\n")
    return "".join(out).encode()


def _ref_fastq_bytes(rows: list[dict]) -> bytes:
    out = []
    for row in rows:
        header = str(row["id"] or "")
        if row.get("description"):
            header = f"{header} {row['description']}"
        out.append(f"@{header}\n{row['sequence'] or ''}\n+\n{row['quality'] or ''}\n")
    return "".join(out).encode()


def test_the_vectorized_writers_match_the_per_row_reference():
    rng = random.Random(5)
    lengths = [0, 1, 59, 60, 61, 119, 120, 121, 300]
    rows = []
    for i in range(400):
        seq = "".join(rng.choice("ACGTé") for _ in range(rng.choice(lengths)))
        rows.append(
            {
                "id": rng.choice([f"s{i}", None, ""]),
                "description": rng.choice([None, "", "a desc", "x\ty"]),
                "sequence": rng.choice([seq, None]) if i % 7 else seq,
                "quality": None,
            }
        )
    fasta_rows = [{k: r[k] for k in ("id", "description", "sequence")} for r in rows]
    batch = pa.RecordBatch.from_pylist(fasta_rows)
    assert fasta_mod._encode(batch) == _ref_fasta_bytes(fasta_rows)
    fastq_rows = [
        {**r, "sequence": r["sequence"] or "", "quality": "I" * len(r["sequence"] or "")}
        for r in rows
    ]
    assert fastq_mod._encode(pa.RecordBatch.from_pylist(fastq_rows), 0) == _ref_fastq_bytes(
        fastq_rows
    )


def test_the_tsv_writer_renders_floats_exactly_as_python_does(tmp_path):
    """The score column is float64, which Arrow's cast formats differently from `str()`."""
    scores = [None, 0.1, 880644658031726.2, 1e-05, 3.0, 1e16]
    table = {
        "seqid": ["c"] * 6,
        "source": ["s"] * 6,
        "type": ["gene"] * 6,
        "start": list(range(1, 7)),
        "end": list(range(10, 16)),
        "score": scores,
        "strand": ["+", None, "-", ".", "+", "+"],
        "phase": [0, None, 1, 2, None, 0],
        "attributes": ["ID=a"] * 6,
    }
    out = tmp_path / "o.gff3"
    bt.from_pydict(table).write.gff(str(out))
    body = out.read_text().splitlines()[1:]
    assert [line.split("\t")[5] for line in body] == ["." if s is None else str(s) for s in scores]
