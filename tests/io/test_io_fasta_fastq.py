"""The FASTA and FASTQ readers and writers.

These two formats are where a genomics pipeline starts, and both have a failure mode that a
naive reader hits silently rather than loudly. FASTA wraps one record's sequence across
arbitrarily many lines, so a line-oriented read produces fragments that look like records.
FASTQ's quality string is one character per base, so a file where the two lengths disagree
produces a column where every score is attributed to the wrong base.

Both are pinned here, alongside the round-trip, split-coverage, and detection contracts every
format owes.
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

# One record wrapped across two lines and one that is not, plus a header with a description
# and one without — the four shapes a real FASTA file mixes.
_FASTA = ">chr1 Homo sapiens chromosome 1\nATGGCC\nTAA\n>chr2\nGGCC\n"
_FASTQ = "@r1 lane1\nACGT\n+\nIIII\n@r2\nGGCC\n+\n!5I?\n"


def _write(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --- FASTA ---------------------------------------------------------------------------


def test_fasta_rejoins_a_wrapped_sequence(tmp_path):
    """A record's sequence spans lines, so the row boundary is `>`, not the newline."""
    out = bt.read.fasta(_write(tmp_path, "a.fasta", _FASTA)).to_pydict()
    assert out == {
        "id": ["chr1", "chr2"],
        "description": ["Homo sapiens chromosome 1", ""],
        "sequence": ["ATGGCCTAA", "GGCC"],
    }


def test_fasta_splits_the_header_on_its_first_whitespace(tmp_path):
    """The NCBI convention: the name, then free text. A missing description is empty, not null.

    Null would say "this reader could not parse a description", which is a different fact
    from "the header carried none" — and only the second is true here.
    """
    text = ">id_only\nA\n>id_and_desc  a  long   description \nC\n"
    out = bt.read.fasta(_write(tmp_path, "h.fasta", text)).to_pydict()
    assert out["id"] == ["id_only", "id_and_desc"]
    assert out["description"] == ["", "a  long   description"]


def test_fasta_reads_every_conventional_suffix_from_one_directory(tmp_path):
    """A corpus mixes `.fa`, `.fasta`, `.faa`, `.fna` and `.ffn` freely."""
    for i, ext in enumerate((".fasta", ".fa", ".faa", ".fna", ".ffn")):
        _write(tmp_path, f"s{i}{ext}", f">rec{i}\nACGT\n")
    ids = bt.read.fasta(str(tmp_path)).to_pydict()["id"]
    assert sorted(ids) == [f"rec{i}" for i in range(5)]


def test_fasta_round_trips(tmp_path):
    """write → read → the same table, wrapped at the reference width on the way out."""
    ds = bt.read.fasta(_write(tmp_path, "a.fasta", _FASTA))
    out = str(tmp_path / "out.fasta")
    ds.write.fasta(out)
    assert bt.read.fasta(out).to_pydict() == ds.to_pydict()
    # 60 characters is the NCBI/UniProt width, so a written file is comparable with the
    # corpus it came from rather than being one long line.
    body = Path(out).read_text()
    assert all(len(line) <= 60 for line in body.splitlines() if not line.startswith(">"))


def test_fasta_wraps_a_long_sequence_at_sixty(tmp_path):
    ds = bt.from_pydict({"id": ["long"], "sequence": ["A" * 150]})
    out = str(tmp_path / "w.fasta")
    ds.write.fasta(out)
    lines = [ln for ln in Path(out).read_text().splitlines() if not ln.startswith(">")]
    assert [len(ln) for ln in lines] == [60, 60, 30]
    assert bt.read.fasta(out).to_pydict()["sequence"] == ["A" * 150]


def test_fasta_survives_an_empty_sequence_round_trip(tmp_path):
    """A record with no bases must not swallow the next record's header on the way back."""
    ds = bt.from_pydict({"id": ["empty", "next"], "sequence": ["", "ACGT"]})
    out = str(tmp_path / "e.fasta")
    ds.write.fasta(out)
    back = bt.read.fasta(out).to_pydict()
    assert back["id"] == ["empty", "next"]
    assert back["sequence"] == ["", "ACGT"]


def test_fasta_write_names_the_columns_it_needs(tmp_path):
    with pytest.raises(FormatError, match="id"):
        bt.from_pydict({"seq": ["ACGT"]}).write.fasta(str(tmp_path / "x.fasta"))


# --- FASTQ ---------------------------------------------------------------------------


def test_fastq_reads_four_line_records(tmp_path):
    out = bt.read.fastq(_write(tmp_path, "r.fastq", _FASTQ)).to_pydict()
    assert out == {
        "id": ["r1", "r2"],
        "description": ["lane1", ""],
        "sequence": ["ACGT", "GGCC"],
        "quality": ["IIII", "!5I?"],
    }


def test_fastq_keeps_quality_as_text_so_the_offset_stays_the_callers_choice(tmp_path):
    """The bytes cannot say whether the run encoded Q+33 or Q+64, so the reader must not guess.

    Decoding here would bake in a guess that shifts every score by 31 when it is wrong. The
    column arrives as text and `.seq` decodes it once the caller has stated the encoding.
    """
    ds = bt.read.fastq(_write(tmp_path, "r.fastq", _FASTQ))
    assert ds.schema.field("quality").type == pa.string()
    sanger = ds.select(m=bt.col("quality").seq.mean_quality()).to_pydict()["m"]
    illumina = ds.select(m=bt.col("quality").seq.mean_quality(offset=64)).to_pydict()["m"]
    assert sanger[0] == 40.0
    assert illumina[0] == 9.0


def test_fastq_refuses_a_record_whose_quality_length_disagrees(tmp_path):
    """The check that matters: one character per base, or every score is misattributed."""
    with pytest.raises(FormatError, match="4 bases but 2 quality characters"):
        bt.read.fastq(_write(tmp_path, "bad.fastq", "@r1\nACGT\n+\nII\n")).to_pydict()


def test_fastq_refuses_a_truncated_final_record(tmp_path):
    with pytest.raises(FormatError, match="ends mid-record"):
        bt.read.fastq(_write(tmp_path, "t.fastq", "@r1\nACGT\n+\n")).to_pydict()


def test_fastq_refuses_a_file_that_is_not_four_line_fastq(tmp_path):
    with pytest.raises(FormatError, match="does not start with"):
        bt.read.fastq(_write(tmp_path, "n.fastq", "r1\nACGT\n+\nIIII\n")).to_pydict()
    with pytest.raises(FormatError, match="no '\\+' separator"):
        bt.read.fastq(_write(tmp_path, "p.fastq", "@r1\nACGT\n-\nIIII\n")).to_pydict()


def test_fastq_round_trips(tmp_path):
    ds = bt.read.fastq(_write(tmp_path, "r.fastq", _FASTQ))
    out = str(tmp_path / "out.fastq")
    ds.write.fastq(out)
    assert bt.read.fastq(out).to_pydict() == ds.to_pydict()


def test_fastq_write_refuses_a_length_mismatch_on_the_way_out_too(tmp_path):
    """Refused symmetrically: writing it would push the corruption to the next reader."""
    ds = bt.from_pydict({"id": ["r"], "sequence": ["ACGT"], "quality": ["II"]})
    with pytest.raises(FormatError, match="4 bases but 2 quality"):
        ds.write.fastq(str(tmp_path / "x.fastq"))


# --- Contracts every format owes -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("fasta", ["id", "description", "sequence"]),
        ("fastq", ["id", "description", "sequence", "quality"]),
    ],
)
def test_an_empty_file_still_reports_the_schema(tmp_path, name, columns):
    """No rows, but the columns — so a caller can build a plan against an empty partition."""
    ds = bt.read(_write(tmp_path, f"e.{name}", ""), format=name)
    assert ds.schema.names == columns
    assert ds.to_pydict() == {name: [] for name in columns}


@pytest.mark.parametrize(("name", "text"), [("fasta", _FASTA), ("fastq", _FASTQ)])
def test_projection_narrows_the_batch(tmp_path, name, text):
    out = bt.read(_write(tmp_path, f"p.{name}", text), format=name).select("id").to_pydict()
    assert list(out) == ["id"]


@pytest.mark.parametrize(("name", "text"), [("fasta", _FASTA), ("fastq", _FASTQ)])
def test_splits_cover_the_source_exactly_once_and_survive_a_pickle(tmp_path, name, text):
    """The distributed path in miniature: a split ships to a worker and reads its own slice."""
    for i in range(3):
        _write(tmp_path, f"f{i}.{name}", text)
    source = SOURCES.get(name)(str(tmp_path))
    whole = [row for batch in source.read() for row in batch.column("id").to_pylist()]
    from_splits: list[str] = []
    for split in source.splits():
        revived = pickle.loads(pickle.dumps(split))
        for batch in revived.read():
            from_splits += batch.column("id").to_pylist()
    assert sorted(from_splits) == sorted(whole)
    assert len(whole) == 6  # two records per file, three files


@pytest.mark.parametrize(("name", "text"), [("fasta", _FASTA), ("fastq", _FASTQ)])
def test_iter_batches_streams_the_same_rows_collect_returns(tmp_path, name, text):
    ds = bt.read(_write(tmp_path, f"s.{name}", text), format=name)
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=ds.schema)
    assert streamed.to_pydict() == ds.to_pydict()


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".fasta", "fasta"),
        (".fa", "fasta"),
        (".faa", "fasta"),
        (".fna", "fasta"),
        (".ffn", "fasta"),
        (".fastq", "fastq"),
        (".fq", "fastq"),
    ],
)
def test_the_extension_infers_the_format(suffix, expected):
    """`bt.read("reads.fq")` must not need a `format=` for a suffix the field standardized."""
    from batcher.io.detect import detect_format

    assert detect_format(f"sample{suffix}") == expected


def test_both_formats_are_registered_under_their_conventional_names():
    for name in ("fasta", "fastq"):
        assert name in SOURCES.names()
        assert name in SINKS.names()
