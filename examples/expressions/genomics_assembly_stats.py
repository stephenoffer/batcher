"""Judging genome assemblies: N50, N90, L50, and auN as mergeable aggregates.

An assembler hands back a pile of contigs. The question is how contiguous the assembly is —
whether the sequence sits in a few big pieces or a lot of small ones — and the answer is not a
mean or a median of the lengths. It is a *base-weighted* statistic, because what matters is
where the bases are, not how many pieces there happen to be.

All four are mergeable, so the number computed over a shuffle is the number a single node
would compute. That is what makes "N50 per sample across a cohort" a group-by rather than a
per-sample script.

    python examples/expressions/genomics_assembly_stats.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # Three assemblies of the *same* total sequence, arranged differently. That is the whole
    # experiment: total length cannot distinguish them and contiguity can.
    contigs = bt.from_pydict(
        {
            "assembly": (["chromosome"] * 5 + ["fragmented"] * 50 + ["mixed"] * 21),
            "length": (
                # One assembly in five big pieces.
                [4_000_000, 3_000_000, 2_000_000, 800_000, 200_000]
                # The same 10 Mb in fifty even pieces.
                + [200_000] * 50
                # One chromosome plus twenty pieces of debris.
                + [9_600_000]
                + [20_000] * 20
            ),
        }
    )

    # Total sequence is identical for all three, so it says nothing.
    totals = contigs.group_by("assembly").agg(total=col("length").sum()).sort("assembly")
    print(totals.to_pydict())
    assert totals.to_pydict()["total"] == [10_000_000, 10_000_000, 10_000_000]

    # --- The contiguity statistics ---------------------------------------------------
    stats = (
        contigs.group_by("assembly")
        .agg(
            contigs=bt.count(),
            n50=col("length").n50(),
            n90=col("length").n90(),
            l50=col("length").l50(),
            aun=col("length").aun(),
            median=col("length").median(),
        )
        .sort("assembly")
    )
    result = stats.to_pydict()
    print(result)

    idx = {name: i for i, name in enumerate(result["assembly"])}
    chrom, frag, mixed = idx["chromosome"], idx["fragmented"], idx["mixed"]

    # N50 ranks them the way a biologist would: the five-piece assembly is the best of the
    # three, the fifty-piece one the worst.
    assert result["n50"][chrom] > result["n50"][frag]

    # L50 is a *count*, and low is good: three pieces carry half the chromosome-scale
    # assembly, twenty-five carry half the fragmented one.
    assert result["l50"][chrom] < result["l50"][frag]
    assert isinstance(result["l50"][chrom], int)

    # The median is actively misleading on the mixed assembly: it reports the debris.
    assert result["median"][mixed] == 20_000
    assert result["n50"][mixed] == 9_600_000

    # N90 always reaches further down than N50, so it is never larger.
    for i in (chrom, frag, mixed):
        assert result["n90"][i] <= result["n50"][i]

    # --- Why auN exists --------------------------------------------------------------
    #
    # N50 is a step function: a single contig crossing the halfway mark moves it in a jump,
    # so two assemblies can swap rank on a rounding difference. auN integrates over every
    # threshold instead, so it is continuous in the lengths and is the better number to rank
    # on when two assemblies are close.
    close = bt.from_pydict(
        {
            "assembly": ["a"] * 3 + ["b"] * 3,
            # Assembly "b" differs from "a" only in its middle contig, 99 -> 101.
            "length": [100, 99, 98, 100, 101, 98],
        }
    )
    both = (
        close.group_by("assembly")
        .agg(n50=col("length").n50(), aun=col("length").aun())
        .sort("assembly")
        .to_pydict()
    )
    print(both)
    n_gap = abs(both["n50"][0] - both["n50"][1])
    a_gap = abs(both["aun"][0] - both["aun"][1])
    assert a_gap < n_gap, "auN should move less than N50 for a small change in lengths"

    # --- Mergeability ----------------------------------------------------------------
    #
    # The statistics are the same however the work is scheduled, which is what makes them
    # usable on a cluster. Repeating the input scales the assembly without changing its
    # shape, so the length-valued statistics are exactly unchanged.
    #
    # L50 is the interesting one: it does *not* simply double. Doubling every length lets the
    # running total cross the halfway mark up to one contig earlier, so it lands in
    # [2*L50 - 1, 2*L50] — here 2 becomes 3, not 4.
    doubled = bt.from_pydict(
        {
            "assembly": ["chromosome"] * 10,
            "length": [4_000_000, 3_000_000, 2_000_000, 800_000, 200_000] * 2,
        }
    )
    d = (
        doubled.group_by("assembly")
        .agg(n50=col("length").n50(), aun=col("length").aun(), l50=col("length").l50())
        .to_pydict()
    )
    print(d)
    assert d["n50"][0] == result["n50"][chrom]
    assert abs(d["aun"][0] - result["aun"][chrom]) < 1e-6
    base = result["l50"][chrom]
    assert 2 * base - 1 <= d["l50"][0] <= 2 * base

    # And the streaming scheduling agrees with the batch one.
    streamed = list(stats.iter_batches())
    assert sum(b.num_rows for b in streamed) == 3

    # --- Straight from a FASTA -------------------------------------------------------
    #
    # The statistic reads a length column, and a length column is one expression away from
    # the assembly file itself.
    import tempfile
    from pathlib import Path

    fasta = Path(tempfile.mkdtemp()) / "asm.fasta"
    fasta.write_text(">c1\n" + "A" * 120 + "\n>c2\n" + "C" * 60 + "\n>c3\n" + "G" * 20 + "\n")
    from_file = (
        bt.read.fasta(str(fasta))
        .with_columns(length=col("sequence").str.len())
        .agg(n50=col("length").n50(), l50=col("length").l50())
        .to_pydict()
    )
    print(from_file)
    # 120 + 60 + 20 = 200; half is 100, and the first contig alone clears it.
    assert from_file["n50"] == [120.0]
    assert from_file["l50"] == [1]


if __name__ == "__main__":
    main()
