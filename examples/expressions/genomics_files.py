"""FASTA and FASTQ files as tables: read, measure, filter, write.

The two formats a genomics pipeline starts from, and the reason each needs a real reader
rather than a text read plus string work. Reading streams, so a reference genome held in a
handful of enormous records never materializes whole.

    python examples/expressions/genomics_files.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    workdir = Path(tempfile.mkdtemp())

    # --- FASTA ----------------------------------------------------------------------
    #
    # A FASTA writer wraps a sequence across as many lines as it likes, so the row
    # boundary is the `>` header rather than the newline. A line-oriented text read would
    # turn this two-line record into two rows that both look like sequence.
    genome = workdir / "contigs.fasta"
    genome.write_text(
        ">chr1 Homo sapiens chromosome 1\n"
        "ATGGCCTAAGGCCATTAGATG\n"
        "GCCTAAGGCCATTAGCCCGGG\n"
        ">chr2 plasmid\n"
        "GGGCCCGGGCCCAAATTTGGG\n"
        ">chr3 low-complexity\n"
        "AAAAAAAAAAAAAAAAAAAA\n"
    )

    contigs = bt.read.fasta(str(genome))
    print(contigs.to_pydict())

    # Two records, two rows — and chr1's two lines are one sequence again.
    result = contigs.to_pydict()
    assert result["id"] == ["chr1", "chr2", "chr3"]
    assert len(result["sequence"][0]) == 42  # both lines, rejoined
    # The header splits on its first whitespace: name, then free text.
    assert result["description"][0] == "Homo sapiens chromosome 1"
    assert result["description"][2] == "low-complexity"

    # Measuring is a projection, so a whole-genome scan is one pass and no Python.
    measured = contigs.with_columns(
        length=col("sequence").str.len(),
        gc=col("sequence").seq.gc_content(),
        longest_run=col("sequence").seq.max_homopolymer(),
    )
    print(measured.select("id", "length", "gc", "longest_run").to_pydict())

    stats = measured.to_pydict()
    assert stats["length"] == [42, 21, 20]
    assert stats["longest_run"][2] == 20  # the poly-A contig

    # Which is a filter, not a script: drop the low-complexity contig.
    usable = measured.filter(col("longest_run") < 10).to_pydict()
    assert usable["id"] == ["chr1", "chr2"]

    # --- FASTQ ----------------------------------------------------------------------
    #
    # A FASTQ record is exactly four lines, and the quality string is one character per
    # base. The reader checks that, because a file where the two disagree produces a
    # column in which every score belongs to the wrong base.
    run = workdir / "reads.fastq"
    run.write_text(
        "@read1 lane1\nACGTTGCAAGG\n+\nIIIIIIIIIII\n"
        "@read2 lane1\nACGTTGCAAGG\n+\nIIIIIIIII#!\n"
        "@read3 lane2\nTTTTTTTTTTT\n+\nIIIIIIIIIII\n"
    )

    reads = bt.read.fastq(str(run))
    print(reads.to_pydict())

    # The quality column arrives as *text*, not as decoded scores, because the ASCII
    # offset is not recoverable from the bytes: Sanger and Illumina 1.8+ encode Q+33 and
    # the older pipelines Q+64, and the ranges overlap. `.seq` decodes it once you say
    # which the run used.
    scored = reads.with_columns(
        mean_q=col("quality").seq.mean_quality(),
        errors=col("quality").seq.expected_errors(),
    )
    print(scored.select("id", "mean_q", "errors").to_pydict())

    q = scored.to_pydict()
    assert q["mean_q"][0] == 40.0
    # read2 ends in a Q2 and a Q0 base. Its mean is still high, but it is expected to
    # contain more than one wrong base — which is why the filter is on expected errors.
    assert q["mean_q"][1] > 32
    assert q["errors"][1] > 1.0

    clean = reads.filter(col("quality").seq.expected_errors() < 1.0)
    assert clean.to_pydict()["id"] == ["read1", "read3"]

    # --- Writing back out -----------------------------------------------------------
    #
    # Both sinks round-trip, so a filtered run is still a FASTQ file the rest of a
    # toolchain can read.
    filtered = workdir / "clean.fastq"
    clean.write.fastq(str(filtered))
    assert bt.read.fastq(str(filtered)).to_pydict() == clean.to_pydict()

    # A derived column can be written back as FASTA — here, the proteins the contigs encode.
    proteins = usable_proteins(contigs)
    out = workdir / "proteins.faa"
    proteins.write.fasta(str(out))
    back = bt.read.fasta(str(out)).to_pydict()
    print(back)
    assert back["id"] == ["chr1", "chr2", "chr3"]
    assert back["sequence"][0].startswith("MA")


def usable_proteins(contigs):
    """Translate each contig's first reading frame, keeping the record identity."""
    return contigs.select(
        id=bt.col("id"),
        description=bt.col("description"),
        sequence=bt.col("sequence").seq.translate(to_stop=True),
    )


if __name__ == "__main__":
    main()
