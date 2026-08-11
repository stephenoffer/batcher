"""Intervals, annotations, and variants: BED, GFF, and VCF as joinable tables.

The payoff for reading these formats into tables rather than iterating them: "which variants
fall in a coding exon" is a join and a filter, not a script. That means it optimizes,
streams, and distributes like any other query.

The coordinate conventions differ between the formats and nothing normalizes them, so this
script shows the conversion done explicitly — which is the only way it should ever happen.

    python examples/expressions/genomics_intervals.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    workdir = Path(tempfile.mkdtemp())

    # --- Three files, three shapes ---------------------------------------------------
    #
    # BED: intervals, no header, width chosen by the file, browser directives interleaved.
    regions = workdir / "regions.bed"
    regions.write_text(
        "track name=targets\n"
        "chr1\t0\t100\ttarget_A\t900\t+\n"
        "chr1\t150\t400\ttarget_B\t500\t-\n"
        "chr2\t0\t50\ttarget_C\t100\t+\n"
    )

    # GFF3: annotations, nine columns, `.` for every absent value.
    genes = workdir / "genes.gff3"
    genes.write_text(
        "##gff-version 3\n"
        "##sequence-region chr1 1 1000\n"
        "chr1\tHAVANA\tgene\t11\t200\t.\t+\t.\tID=g1;Name=BRCA1\n"
        "chr1\tHAVANA\tCDS\t20\t100\t0.9\t+\t0\tID=c1;Parent=g1\n"
        "chr1\tHAVANA\tgene\t300\t900\t.\t-\t.\tID=g2;Name=TP53\n"
    )

    # VCF: variants, with the sample columns named by the file's own header line.
    variants = workdir / "cohort.vcf"
    variants.write_text(
        "##fileformat=VCFv4.2\n"
        "##INFO=<ID=AF,Number=A,Type=Float>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA12878\n"
        "chr1\t50\trs1\tA\tG\t50.5\tPASS\tAF=0.25;DB\tGT:DP\t0/1:30\n"
        "chr1\t250\t.\tC\tT\t20.0\tq10\tAF=0.01\tGT:DP\t0/0:15\n"
        "chr1\t350\trs3\tG\tA\t99.0\tPASS\tAF=0.40\tGT:DP\t1/1:40\n"
    )

    beds = bt.read.bed(str(regions))
    gff = bt.read.gff(str(genes))
    vcf = bt.read.vcf(str(variants))

    print(beds.to_pydict())
    print(gff.select("type", "start", "end", "attributes").to_pydict())
    print(vcf.select("chrom", "pos", "id", "filter", "info").to_pydict())

    # The sample column is named from the file, not from the specification.
    assert "NA12878" in vcf.schema.names
    # `.` is a genuine absence, not the string ".".
    assert vcf.to_pydict()["id"] == ["rs1", None, "rs3"]
    assert gff.to_pydict()["score"] == [None, 0.9, None]

    # --- Coordinates: convert explicitly, never silently -----------------------------
    #
    # BED is 0-based half-open; GFF and VCF are 1-based inclusive. Both readers report
    # what their file says, so a comparison between them needs the conversion written
    # down. A BED interval [start, end) covers 1-based positions start+1 .. end.
    beds_1based = beds.with_columns(
        start_1=col("start") + 1,
        end_1=col("end"),
    )
    print(beds_1based.select("name", "start", "start_1", "end_1").to_pydict())

    converted = beds_1based.to_pydict()
    assert converted["start"] == [0, 150, 0]
    assert converted["start_1"] == [1, 151, 1]

    # --- Which variants fall inside a target region? ---------------------------------
    #
    # A join on the contig, then an overlap predicate. This is the operation a genomics
    # pipeline is built around, and it is an ordinary relational query.
    in_target = (
        vcf.join(beds_1based, left_on="chrom", right_on="chrom", how="inner")
        .filter((col("pos") >= col("start_1")) & (col("pos") <= col("end_1")))
        .select(variant=col("id"), region=col("name"), pos=col("pos"))
        .sort("pos")
    )
    result = in_target.to_pydict()
    print(result)

    # rs1 at 50 is inside target_A (1-100); the variant at 250 and rs3 at 350 are inside
    # target_B (151-400).
    assert result["region"] == ["target_A", "target_B", "target_B"]
    assert result["pos"] == [50, 250, 350]

    # --- Which variants fall in a coding exon? ---------------------------------------
    #
    # The same shape against the annotation table, restricted to CDS features, and pulling
    # the gene name out of the attributes column with the ordinary string vocabulary.
    coding = gff.filter(col("type") == "CDS").with_columns(
        parent=col("attributes").str.regexp_extract(r"Parent=([^;]+)", 1)
    )
    in_cds = (
        vcf.join(coding, left_on="chrom", right_on="seqid", how="inner")
        .filter((col("pos") >= col("start")) & (col("pos") <= col("end")))
        .select(variant=col("id"), parent=col("parent"))
    )
    print(in_cds.to_pydict())
    assert in_cds.to_pydict()["variant"] == ["rs1"]  # only rs1 (pos 50) is in the CDS 20-100

    # --- Passing, common variants per contig -----------------------------------------
    #
    # INFO is text, so the allele frequency is an extract-and-cast — and then it is just a
    # number, so a group-by works on it like any other column.
    summary = (
        vcf.filter(col("filter") == "PASS")
        .with_columns(af=col("info").str.regexp_extract(r"AF=([0-9.]+)", 1).cast("float64"))
        .group_by("chrom")
        .agg(n=bt.count(), mean_af=col("af").mean())
        .to_pydict()
    )
    print(summary)
    assert summary["n"] == [2]
    assert abs(summary["mean_af"][0] - 0.325) < 1e-9

    # --- Writing intervals back out --------------------------------------------------
    out = workdir / "hits.bed"
    hits = in_target.select(
        chrom=bt.lit("chr1"), start=col("pos") - 1, end=col("pos"), name=col("region")
    )
    hits.write.bed(str(out))
    print(bt.read.bed(str(out)).to_pydict())
    assert bt.read.bed(str(out)).count() == 3


if __name__ == "__main__":
    main()
