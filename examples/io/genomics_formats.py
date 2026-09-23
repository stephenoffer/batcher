"""Reading and writing the genomics formats: FASTA, FASTQ, BED, GFF3, and VCF.

Each format reads as an ordinary table and reports what its file says, so BED coordinates stay
0-based half-open while GFF and VCF stay 1-based. The script builds a small file of each kind,
gzips some of them the way they are usually shipped, reads them back, and writes the four
formats that have a writer.

    python examples/io/genomics_formats.py
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # FASTA: a record's sequence is wrapped across lines and re-joined on read; a `;`
        # line is a comment even inside a record. Gzipped, it reads the same way.
        fasta = ">chr1 Homo sapiens\nACGTACGT\n;annotation\nNNAC\n>chrM\nGATTACA\n"
        (root / "ref.fa.gz").write_bytes(gzip.compress(fasta.encode()))
        ref = bt.read.fasta(str(root / "ref.fa.gz"))
        print(ref.to_pydict())
        assert ref.to_pydict() == {
            "id": ["chr1", "chrM"],
            "description": ["Homo sapiens", ""],
            "sequence": ["ACGTACGTNNAC", "GATTACA"],
        }

        # FASTQ: four lines per read; the quality string stays text, so the Phred offset is
        # the caller's to name when decoding it.
        fastq = "@r1 lane:1\nACGT\n+\nIIII\n@r2 lane:2\nGGCA\n+\n!!II\n"
        (root / "reads.fq").write_text(fastq)
        reads = bt.read.fastq(str(root / "reads.fq"))
        kept = reads.filter(col("sequence").str.starts_with("A")).select("id", "quality")
        print(kept.to_pydict())
        assert kept.to_pydict() == {"id": ["r1"], "quality": ["IIII"]}

        # VCF, bgzipped: the samples come from the #CHROM header, and INFO stays text for the
        # string vocabulary to query.
        vcf = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA12878\tNA12891\n"
            "chr1\t100\trs1\tA\tG\t50\tPASS\tAF=0.25;DB\tGT\t0/1\t1/1\n"
            "chr1\t250\t.\tC\tT,A\t.\tq10\tAF=0.01\tGT\t0/0\t0/1\n"
        )
        (root / "calls.vcf.gz").write_bytes(gzip.compress(vcf.encode()))
        calls = bt.read.vcf(str(root / "calls.vcf.gz"))
        assert calls.columns[-2:] == ["NA12878", "NA12891"]
        common = calls.filter(col("info").str.contains("DB")).select("pos", "NA12878")
        print(common.to_pydict())
        assert common.to_pydict() == {"pos": [100], "NA12878": ["0/1"]}

        # GFF3 with an appended ##FASTA section: the table ends at the directive.
        gff = (
            "##gff-version 3\n"
            "chr1\tsrc\tgene\t90\t300\t.\t+\t.\tID=g1\n"
            "chr1\tsrc\tCDS\t90\t200\t.\t+\t0\tParent=g1\n"
            "##FASTA\n>chr1\nACGT\n"
        )
        (root / "genes.gff3").write_text(gff)
        genes = bt.read.gff(str(root / "genes.gff3"))
        assert genes.to_pydict()["type"] == ["gene", "CDS"]

        # BED and bedGraph: a `track type=bedGraph` line makes the fourth column a float.
        (root / "peaks.bed").write_text("track name=peaks\nchr1\t95\t120\tp1\nchr1\t400\t410\tp2\n")
        (root / "cov.bed").write_text("track type=bedGraph\nchr1\t0\t100\t2.5\n")
        peaks = bt.read.bed(str(root / "peaks.bed"))
        coverage = bt.read.bed(str(root / "cov.bed"))
        assert coverage.to_pydict()["value"] == [2.5]

        # Which peaks overlap a gene? BED is 0-based half-open and GFF 1-based inclusive,
        # so the GFF start moves down by one before the interval comparison.
        overlaps = (
            peaks.join(genes.filter(col("type") == "gene"), left_on="chrom", right_on="seqid")
            .filter((col("start") < col("end_right")) & (col("start_right") - 1 < col("end")))
            .select("name")
        )
        print(overlaps.to_pydict())
        assert overlaps.to_pydict() == {"name": ["p1"]}

        # Writing: every format but VCF has a writer, and each round-trips.
        ref.write.fasta(str(root / "out.fasta"))
        reads.write.fastq(str(root / "out.fastq"))
        peaks.write.bed(str(root / "out.bed"))
        genes.write.gff(str(root / "out.gff3"))
        assert bt.read.fasta(str(root / "out.fasta")).to_pydict() == ref.to_pydict()
        assert bt.read.fastq(str(root / "out.fastq")).to_pydict() == reads.to_pydict()
        assert bt.read.bed(str(root / "out.bed")).to_pydict() == peaks.to_pydict()
        assert bt.read.gff(str(root / "out.gff3")).to_pydict() == genes.to_pydict()
        print("all four writers round-trip")


if __name__ == "__main__":
    main()
