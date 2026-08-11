"""Sequencing reads as a table: quality filtering, k-mer sketching, and primer design.

The three things done to a run of reads before anything else — drop the bad ones, sketch
the rest so they can be compared, and check the oligos that produced them. Each is a
column expression here, so a run of hundreds of millions of reads stays a scan.

    python examples/expressions/genomics_reads.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # A FASTQ record is a sequence and an equally long quality string. `I` is Q40, `5` is
    # Q20, `#` is Q2, `!` is Q0 — one ASCII character per base.
    reads = bt.from_pydict(
        {
            "id": ["read1", "read2", "read3", "read4"],
            "seq": ["ACGTTGCAAGG", "ACGTTGCAAGG", "TTTTTTTTTTT", "CCTTGCAACGT"],
            "qual": ["IIIIIIIIIII", "IIIIIIIII#!", "IIIIIIIIIII", "IIIIIIIIIII"],
        }
    )

    # --- Quality --------------------------------------------------------------------
    #
    # The offset is stated rather than sniffed: Sanger and Illumina 1.8+ encode Q+33, the
    # older Illumina pipelines Q+64, and the two ranges overlap, so a wrong guess shifts
    # every score by 31.
    scored = reads.with_columns(
        mean_q=col("qual").seq.mean_quality(),
        errors=col("qual").seq.expected_errors(),
        per_base=col("qual").seq.phred_quality(),
    )
    result = scored.to_pydict()
    print(result)

    assert result["mean_q"][0] == 40.0
    assert result["per_base"][0][0] == 40
    # read2 ends in a Q2 and a Q0 base. Its mean is still high...
    assert result["mean_q"][1] > 32
    # ...but it is expected to contain more than one wrong base, which the mean cannot
    # see. This is why `expected_errors` is the filter to threshold on: it is additive,
    # so one terrible base outweighs sixty good ones.
    assert result["errors"][1] > 1.0
    assert result["errors"][0] < 0.01

    # The standard high-accuracy filter, in the engine.
    clean = reads.filter(col("qual").seq.expected_errors() < 1.0).to_pydict()
    assert clean["id"] == ["read1", "read3", "read4"]

    # --- Sketching ------------------------------------------------------------------
    #
    # Canonical k-mers fold each window with its reverse complement, so a read and the
    # same fragment sequenced from the other strand produce the same table. read4 is
    # read1's reverse complement, and their k-mer sets agree because of it.
    sketched = reads.with_columns(
        kmers=col("seq").seq.canonical_kmers(5),
        sketch=col("seq").seq.minimizers(5, 4),
    )
    result = sketched.to_pydict()
    print(result)

    assert sorted(result["kmers"][0]) == sorted(result["kmers"][3])
    # A minimizer sketch is a subset of the k-mers — a few per window rather than all.
    assert set(result["sketch"][0]) <= set(result["kmers"][0])
    assert len(result["sketch"][0]) < len(result["kmers"][0])

    # Overlap detection: two reads sharing a substring of `window + k - 1` bases are
    # guaranteed to share a minimizer, so a list intersection cannot miss a real overlap.
    pairs = bt.from_pydict(
        {
            "a": ["TTTTTACGTTGCAAGGGGGGG"],
            "b": ["CCCCCACGTTGCAAGGAAAAA"],
        }
    )
    shared = pairs.select(
        n=col("a").seq.minimizers(5, 4).list.intersect(col("b").seq.minimizers(5, 4)).list.len()
    ).to_pydict()
    print(shared)
    assert shared["n"][0] > 0

    # A k-mer frequency table is an explode plus a group-by — no bespoke counter.
    freq = (
        reads.select(k=col("seq").seq.canonical_kmers(3))
        .explode("k")
        .group_by("k")
        .agg(n=bt.count())
        .sort("n", descending=True)
        .to_pydict()
    )
    print(freq)
    # read3 is a poly-T run, so AAA (the canonical form of TTT) is the commonest 3-mer.
    assert freq["k"][0] == "AAA"

    # --- Primer design --------------------------------------------------------------
    #
    # Screening candidate oligos is a filter over a computed column. `melting_temp` uses
    # the SantaLucia nearest-neighbour model, so two oligos with the same base
    # composition but different stacking get different answers — which a GC-percentage
    # formula cannot do.
    candidates = bt.from_pydict(
        {
            "primer": [
                "GTAAAACGACGGCCAGTGAA",  # M13 forward, a real primer
                "ATATATATATATATATATAT",  # too AT-rich to anneal
                "GCGCGCGCGCGCGCGCGCGC",  # too GC-rich, and prone to self-binding
                "ACGTNACGTACGTACGTACG",  # an ambiguity code: no defined answer
            ]
        }
    )
    designed = candidates.with_columns(
        tm=col("primer").seq.melting_temp(),
        gc=col("primer").seq.gc_content(),
        mass=col("primer").seq.molecular_weight("dna"),
        run=col("primer").seq.max_homopolymer(),
    )
    result = designed.to_pydict()
    print(result)

    assert 50 < result["tm"][0] < 70
    assert result["tm"][1] < result["tm"][0] < result["tm"][2]
    # An ambiguity code has no defined stacking energy, so the row is null rather than a
    # specific temperature the data does not support.
    assert result["tm"][3] is None

    usable = candidates.filter(
        (col("primer").seq.melting_temp() >= 55)
        & (col("primer").seq.melting_temp() <= 65)
        & (col("primer").seq.max_homopolymer() <= 4)
    ).to_pydict()
    assert usable["primer"] == ["GTAAAACGACGGCCAGTGAA"]


if __name__ == "__main__":
    main()
