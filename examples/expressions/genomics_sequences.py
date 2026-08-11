"""Nucleotide sequences as a column: complementing, measuring, translating, and searching.

A genomics pipeline is a string pipeline with a different alphabet, and the ``.seq``
accessor is that alphabet's vocabulary. Every operation here runs per-base in Rust over
whole columns, so a scan over a reference genome never materializes a sequence in Python.

    python examples/expressions/genomics_sequences.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    contigs = bt.from_pydict(
        {
            "name": ["chr1_frag", "chr2_frag", "masked", "gapped"],
            # Lowercase is how a reference genome marks soft-masked repeats, and one
            # fragment carries an assembly gap as a run of N.
            "dna": ["ATGGCCTAA", "GGCCATTAG", "atggcctaa", "ATGNNNTAA"],
        }
    )

    # --- Strand ---------------------------------------------------------------------
    #
    # Reverse-complementing is the most-used operation in genomics: a read maps to either
    # strand, so anything comparing two sequences starts here. Case is preserved, so the
    # soft mask survives the transform.
    strands = contigs.with_columns(
        rc=col("dna").seq.reverse_complement(),
        comp=col("dna").seq.complement(),
        rna=col("dna").seq.transcribe(),
        # And back again, for a transcript column that has to line up with a genome.
        round_trip=col("dna").seq.transcribe().seq.back_transcribe(),
    )
    result = strands.to_pydict()
    print(result)

    assert result["rc"][0] == "TTAGGCCAT"
    assert result["comp"][0] == "TACCGGATT"
    assert result["rna"][0] == "AUGGCCUAA"
    # transcribe and back_transcribe are exact inverses, case included.
    assert result["round_trip"] == result["dna"]
    # The mask survives: the soft-masked fragment reverse-complements to lower case.
    assert result["rc"][2] == "ttaggccat"
    # Reverse-complementing twice is the identity.
    twice = contigs.select(x=col("dna").seq.reverse_complement().seq.reverse_complement())
    assert twice.to_pydict()["x"] == result["dna"]

    # --- Composition ----------------------------------------------------------------
    #
    # `gc_content` excludes ambiguous bases from the denominator. That is the difference
    # between "no data" and "a real signal": counting the Ns would put the gapped fragment
    # at the AT-rich extreme of every histogram built on it.
    counts = col("dna").seq.base_counts()
    composition = contigs.with_columns(
        gc=col("dna").seq.gc_content(),
        skew=col("dna").seq.gc_skew(),
        n_bases=counts.struct.field("n"),
        a_bases=counts.struct.field("a"),
        longest_run=col("dna").seq.max_homopolymer(),
    )
    result = composition.to_pydict()
    print(result)

    # "ATGGCCTAA": 4 of 9 bases are G or C.
    assert abs(result["gc"][0] - 4 / 9) < 1e-12
    # "ATGNNNTAA" has six *known* bases (ATG-TAA) of which one is a G, so it reports 1/6.
    # Counting the three Ns as non-GC would have said 1/9 — a fifth lower, and an assembly
    # gap would read as a genuinely AT-rich stretch.
    assert abs(result["gc"][3] - 1 / 6) < 1e-12
    assert result["n_bases"][3] == 3
    # Case folds for the measures, so the masked fragment measures like its unmasked copy.
    assert result["gc"][2] == result["gc"][0]
    assert result["longest_run"][0] == 2  # the GG, the CC, and the trailing AA
    assert result["longest_run"][3] == 3  # the NNN run

    # --- Coding ---------------------------------------------------------------------
    #
    # Translation reads three bases at a time, which is exactly why it cannot be spelled
    # with substring arithmetic without a per-row Python loop.
    proteins = contigs.with_columns(
        protein=col("dna").seq.translate(),
        orf=col("dna").seq.translate(to_stop=True),
        # The three forward frames; combine with reverse_complement for the other three.
        frame1=col("dna").seq.translate(frame=1),
    )
    result = proteins.to_pydict()
    print(result)

    assert result["protein"][0] == "MA*"  # ATG GCC TAA -> Met Ala Stop
    assert result["orf"][0] == "MA"  # the protein this ORF actually encodes
    assert result["frame1"][0] == "WP"  # TGG CCT -> Trp Pro
    # An ambiguous codon is X, never a guessed residue.
    assert result["protein"][3] == "MX*"

    # --- Motifs ---------------------------------------------------------------------
    #
    # A motif is written in the IUPAC degenerate alphabet, and matching is defined on
    # *sets* of bases rather than on text — so an N in the reference matches every pattern
    # base, which a character-class regex over the literal text does not do.
    sites = bt.from_pydict({"dna": ["GGAATTCC", "GGATTTCC", "AAAA", "NNNNNN"]})
    found = sites.with_columns(
        ecori=col("dna").seq.count_motif("GAATTC"),
        degenerate=col("dna").seq.count_motif("GGWWTT"),
        positions=col("dna").seq.find_motif("AA"),
    )
    result = found.to_pydict()
    print(result)

    assert result["ecori"] == [1, 0, 0, 1]  # the run of Ns is consistent with the site
    assert result["degenerate"] == [1, 1, 0, 1]  # W is A-or-T
    # Matches overlap, and positions are 1-based like every genome coordinate.
    assert result["positions"][2] == [1, 2, 3]

    # --- The point ------------------------------------------------------------------
    #
    # These compose with the ordinary relational verbs, so "which fragments are GC-rich
    # and contain a restriction site" is a filter, not a script.
    usable = contigs.filter(
        (col("dna").seq.gc_content() > 0.4) & (col("dna").seq.is_valid("dna"))
    ).to_pydict()
    assert usable["name"] == ["chr1_frag", "chr2_frag", "masked"]


if __name__ == "__main__":
    main()
