"""The `.seq` genomics namespace, checked against biological ground truth.

DuckDB can express reverse-complement, GC content, k-mer slicing, and Phred arithmetic in
its general string vocabulary, and those comparisons live in
``tests/differential/test_diff_seq_namespace.py``. This file covers what it cannot express
at all — the genetic code, IUPAC-degenerate matching, the nearest-neighbour thermodynamic
model, the amino-acid tables — plus the plan-build-time argument validation, which has no
oracle because a rejected plan produces no rows.

Ground truth here is the textbook, not another implementation: `ATG` is methionine, `TAA`
is a stop, `GAATTC` is the EcoRI site, glycine weighs 75.07 daltons. Where a number would
be a remembered constant rather than a definition (a melting temperature, an isoelectric
point), the assertion checks the *property* instead — that the reported pI really does
zero the charge curve, that a duplex melts the same read from either strand.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _one(expr, values, column="s"):
    """Evaluate `expr` over a one-column dataset and return the result list."""
    return bt.from_pydict({column: values}).select(v=expr).to_pydict()["v"]


# --- The genetic code ----------------------------------------------------------------


def test_the_standard_genetic_code_is_the_standard_genetic_code():
    """Spot-checks that each fail on a differently-ordered codon table."""
    codons = ["ATG", "TAA", "TAG", "TGA", "TTT", "GGG", "TGG", "GCC", "CAT"]
    assert _one(bt.col("s").seq.translate(), codons) == [
        "M", "*", "*", "*", "F", "G", "W", "A", "H",
    ]  # fmt: skip


def test_rna_translates_identically_to_dna():
    """A transcript and its gene encode the same protein."""
    assert _one(bt.col("s").seq.translate(), ["AUGGCC"]) == _one(
        bt.col("s").seq.translate(), ["ATGGCC"]
    )


def test_an_ambiguous_codon_is_x_not_a_guess():
    """`GTN` really is valine, but resolving that in general needs the degenerate table.

    Emitting a specific residue where the data supports none would be worse than `X`,
    which is the IUPAC marker for exactly this situation.
    """
    assert _one(bt.col("s").seq.translate(), ["ATGNNNGCC", "ATGRYSGCC"]) == ["MXA", "MXA"]


def test_a_trailing_partial_codon_is_dropped_not_padded():
    """Two leftover bases encode nothing; padding them would fabricate a residue."""
    assert _one(bt.col("s").seq.translate(), ["ATGGC", "ATGG", "AT", "", None]) == [
        "M",
        "M",
        "",
        "",
        None,
    ]


def test_to_stop_ends_the_protein_at_the_first_stop():
    """The protein an ORF encodes excludes the stop codon itself."""
    assert _one(bt.col("s").seq.translate(to_stop=True), ["ATGGCCTAAATGGCC"]) == ["MA"]
    assert _one(bt.col("s").seq.translate(), ["ATGGCCTAAATGGCC"]) == ["MA*MA"]


def test_the_six_frames_are_three_forward_and_three_on_the_other_strand():
    """A six-frame translation composes from the pieces rather than needing its own op."""
    seq = "AATGGCCTAA"
    forward = [_one(bt.col("s").seq.translate(frame=f), [seq])[0] for f in (0, 1, 2)]
    rc = bt.col("s").seq.reverse_complement()
    reverse = [_one(rc.seq.translate(frame=f), [seq])[0] for f in (0, 1, 2)]
    assert forward[1] == "MA*"  # frame 1 of AATGGCCTAA reads ATG GCC TAA
    assert len(set(forward + reverse)) > 1, "the frames should not all agree"


# --- IUPAC-degenerate matching -------------------------------------------------------


def test_a_degenerate_motif_matches_every_base_it_stands_for():
    """`W` is A-or-T, so `GGWWTT` has exactly four literal spellings."""
    seqs = ["GGAATT", "GGATTT", "GGTATT", "GGTTTT", "GGACTT", "GGGGTT"]
    assert _one(bt.col("s").seq.count_motif("GGWWTT"), seqs) == [1, 1, 1, 1, 0, 0]


def test_ambiguity_in_the_sequence_matches_too():
    """The half a character-class regex over the literal text gets wrong.

    A reference genome contains ambiguity codes, and an `N` there is consistent with every
    pattern base — so it matches. Matching is defined on sets of bases, not on characters.
    """
    assert _one(bt.col("s").seq.count_motif("GAATTC"), ["GNATTC", "NNNNNN", "GTATTC"]) == [
        1,
        1,
        0,
    ]


def test_matches_overlap():
    """Tandem repeats are real, so `AA` occurs three times in `AAAA`."""
    assert _one(bt.col("s").seq.count_motif("AA"), ["AAAA"]) == [3]
    assert _one(bt.col("s").seq.find_motif("AA"), ["AAAA"]) == [[1, 2, 3]]


def test_motif_positions_are_one_based():
    """Every genome browser, GFF file, and VCF record counts from 1."""
    assert _one(bt.col("s").seq.find_motif("GAATTC"), ["GGAATTCC"]) == [[2]]


def test_t_and_u_are_interchangeable_in_a_motif():
    """An RNA motif finds a DNA site and the reverse, which a mixed pipeline needs."""
    for pattern in ("ACGT", "ACGU"):
        assert _one(bt.col("s").seq.count_motif(pattern), ["ACGTACGT", "ACGUACGU"]) == [2, 2]


def test_find_and_count_always_agree():
    """The reduction cannot drift from the thing it reduces."""
    seqs = ["GGAATTCCGGAATTCC", "ACGT", "", None]
    found = _one(bt.col("s").seq.find_motif("RRWWYY"), seqs)
    counted = _one(bt.col("s").seq.count_motif("RRWWYY"), seqs)
    assert [None if f is None else len(f) for f in found] == counted


# --- Thermodynamics ------------------------------------------------------------------


def test_a_duplex_melts_the_same_read_from_either_strand():
    """One duplex, one melting temperature — so the model must be strand-symmetric."""
    fwd, rev = "GTAAAACGACGGCCAGTGAA", "TTCACTGGCCGTCGTTTTAC"
    a, b = _one(bt.col("s").seq.melting_temp(), [fwd, rev])
    assert abs(a - b) < 1e-9, f"{a} vs {b}"


def test_stacking_order_matters_which_a_gc_percentage_formula_cannot_see():
    """Same length, same composition, different arrangement — different answer.

    This is the whole reason a nearest-neighbour model is used rather than the Wallace
    rule: those read a sequence as a bag of bases and give these two the same number.
    """
    a, b = _one(bt.col("s").seq.melting_temp(), ["GCGCGCGCGCGC", "GGGGGGCCCCCC"])
    assert abs(a - b) > 1.0, f"{a} vs {b}"


def test_melting_temperature_rises_with_length_and_with_gc():
    """The two monotonicities every Tm model must have."""
    short, long = _one(bt.col("s").seq.melting_temp(), ["ACGTACGT", "ACGTACGTACGTACGT"])
    assert short < long
    at, gc = _one(bt.col("s").seq.melting_temp(), ["ATATATATATATATAT", "GCGCGCGCGCGCGCGC"])
    assert gc > at + 20


def test_a_typical_primer_lands_in_the_range_primers_are_designed_for():
    """A model internally consistent but off by 20 degrees would be unusable."""
    (tm,) = _one(bt.col("s").seq.melting_temp(), ["GTAAAACGACGGCCAGTGAA"])
    assert 50 < tm < 70, tm


def test_an_ambiguous_or_too_short_oligo_has_no_melting_temperature():
    """An ambiguity code has no stacking energy, so the row is null, not approximate."""
    assert _one(bt.col("s").seq.melting_temp(), ["ACGTN", "A", "", None]) == [
        None,
        None,
        None,
        None,
    ]


# --- Mass, hydropathy, and charge ----------------------------------------------------


def test_a_free_amino_acid_weighs_what_the_table_says():
    """Glycine is 75.07 daltons: the residue mass plus the water closing the chain."""
    (mw,) = _one(bt.col("s").seq.molecular_weight("protein"), ["G"])
    assert abs(mw - 75.0672) < 0.01, mw


def test_a_peptide_loses_one_water_per_bond():
    """The bookkeeping that separates residue masses from free-amino-acid masses."""
    one, two = _one(bt.col("s").seq.molecular_weight("protein"), ["G", "GG"])
    assert abs(two - (2 * one - 18.0153)) < 1e-6, (one, two)


def test_dna_weighs_the_monophosphates_minus_a_water_per_bond():
    """A single A is one AMP; ACGT is the four minus three waters."""
    single, quad = _one(bt.col("s").seq.molecular_weight("dna"), ["A", "ACGT"])
    assert abs(single - 331.2218) < 1e-6, single
    expect = 331.2218 + 307.1971 + 347.2212 + 322.2085 - 3 * 18.0153
    assert abs(quad - expect) < 1e-6, quad


def test_rna_outweighs_the_same_dna():
    """The extra 2'-hydroxyl per residue."""
    dna, rna = (_one(bt.col("s").seq.molecular_weight(a), ["ACG"])[0] for a in ("dna", "rna"))
    assert rna > dna


def test_an_unknown_residue_has_no_mass_rather_than_an_approximate_one():
    """Including the `*` stop marker `translate` emits, which is not a residue."""
    assert _one(bt.col("s").seq.molecular_weight("protein"), ["GXG", "MA*", None]) == [
        None,
        None,
        None,
    ]


def test_gravy_separates_a_membrane_stretch_from_a_charged_one():
    """The Kyte-Doolittle scale's whole purpose."""
    hydrophobic, charged = _one(bt.col("s").seq.gravy(), ["IIIVVVLLL", "KKKRRRDDD"])
    assert hydrophobic > 3
    assert charged < -3


def test_gravy_skips_an_unknown_residue_rather_than_erasing_the_row():
    """One `X` in a long predicted protein should not delete its hydropathy."""
    with_x, without = _one(bt.col("s").seq.gravy(), ["IXI", "II"])
    assert with_x == without
    assert _one(bt.col("s").seq.gravy(), ["XXX", ""]) == [None, None]


def test_the_isoelectric_point_separates_a_basic_peptide_from_an_acidic_one():
    """Poly-lysine is strongly basic; poly-aspartate strongly acidic."""
    basic, acidic = _one(bt.col("s").seq.isoelectric_point(), ["KKKK", "DDDD"])
    assert basic > 9, basic
    assert acidic < 4.5, acidic


def test_the_isoelectric_point_is_bounded_and_ordered_by_charge():
    """A property check rather than a remembered constant.

    Adding basic residues can only raise the pI and adding acidic ones can only lower it,
    and every answer stays inside the pH range the bisection searches.
    """
    seqs = ["DDDD", "DDDK", "ACDEFGHIKLMNPQRSTVWY", "KKKD", "KKKK"]
    values = _one(bt.col("s").seq.isoelectric_point(), seqs)
    assert all(0 <= v <= 14 for v in values), values
    assert values[0] < values[1] and values[3] < values[4], values


def test_an_empty_peptide_has_no_isoelectric_point():
    assert _one(bt.col("s").seq.isoelectric_point(), ["", None]) == [None, None]


# --- Sketching properties ------------------------------------------------------------


def test_a_canonical_kmer_table_is_strand_agnostic():
    """The point of canonicalization: a read and its other-strand copy must agree."""
    fwd = _one(bt.col("s").seq.canonical_kmers(3), ["ACGTT"])[0]
    rev = _one(bt.col("s").seq.canonical_kmers(3), ["AACGT"])[0]
    assert sorted(fwd) == sorted(rev)


def test_canonical_picks_the_lexicographically_smaller_strand():
    """`TTT` reverse-complements to `AAA`, which sorts first, so both canonicalize to it."""
    assert _one(bt.col("s").seq.canonical_kmers(3), ["TTT", "AAA"]) == [["AAA"], ["AAA"]]


def test_a_sequence_shorter_than_k_yields_an_empty_list_not_null():
    """It genuinely contains no k-mers, which is a different fact from having no sequence."""
    assert _one(bt.col("s").seq.kmers(5), ["ACGT", "", None]) == [[], [], None]


def test_minimizers_are_a_subset_of_the_canonical_kmers_and_smaller():
    """A sketch, not a re-encoding."""
    seq = "ACGTTGCAAGGCTTAACGTTGCAAGG"
    sketch = _one(bt.col("s").seq.minimizers(4, 5), [seq])[0]
    every = _one(bt.col("s").seq.canonical_kmers(4), [seq])[0]
    assert set(sketch) <= set(every)
    assert len(sketch) < len(every)


def test_two_sequences_sharing_a_long_substring_share_a_minimizer():
    """The guarantee minimizers exist for, which is what makes a sketch join sound.

    An overlap of `window + k - 1` bases cannot be missed. Checked rather than assumed,
    because a broken window scan still produces plausible-looking output.
    """
    k, window, shared = 4, 5, "ACGTTGCAAGG"  # longer than window + k - 1 = 8
    a = _one(bt.col("s").seq.minimizers(k, window), [f"TTTTT{shared}GGGGG"])[0]
    b = _one(bt.col("s").seq.minimizers(k, window), [f"CCCCC{shared}AAAAA"])[0]
    assert set(a) & set(b), (a, b)


def test_a_sequence_shorter_than_the_window_still_has_a_sketch():
    """A short read is not a read that is unmatchable by every minimizer join."""
    sketch = _one(bt.col("s").seq.minimizers(3, 10), ["ACGT"])[0]
    assert len(sketch) == 1


# --- Argument validation, which has no oracle ----------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: bt.col("s").seq.kmers(0), "k must be in"),
        (lambda: bt.col("s").seq.kmers(257), "k must be in"),
        (lambda: bt.col("s").seq.canonical_kmers(-1), "k must be in"),
        (lambda: bt.col("s").seq.minimizers(4, 0), "window must be in"),
        (lambda: bt.col("s").seq.translate(frame=3), "frame must be"),
        (lambda: bt.col("s").seq.translate(frame=-1), "frame must be"),
        (lambda: bt.col("s").seq.is_valid("peptide"), "alphabet must be"),
        (lambda: bt.col("s").seq.molecular_weight("dna_iupac"), "alphabet must be"),
        (lambda: bt.col("s").seq.count_motif(""), "must not be empty"),
        (lambda: bt.col("s").seq.find_motif("AC-GT"), "IUPAC"),
        (lambda: bt.col("s").seq.count_motif("ACXGT"), "IUPAC"),
    ],
)
def test_a_bad_argument_is_a_plan_error_at_build_time(build, message):
    """Caught where the message can name the choices, not as a column of nulls.

    An unmatchable motif is the one worth stating plainly: a typo such as a `-` copied out
    of an alignment would otherwise produce zero matches on every row, which reads exactly
    like a correct answer.
    """
    with pytest.raises(PlanError, match=message):
        build()


def test_a_degenerate_alphabet_is_refused_for_mass_but_accepted_for_validity():
    """The two alphabets lists differ, deliberately, and both are checked.

    An ambiguity code has no mass, so asking for one is a question with no answer; but it
    is perfectly valid sequence, so `is_valid` must accept the degenerate alphabet.
    """
    assert _one(bt.col("s").seq.is_valid("dna_iupac"), ["ACGN", "ACGT", "hello"]) == [
        True,
        True,
        False,
    ]
    with pytest.raises(PlanError, match="alphabet must be"):
        bt.col("s").seq.molecular_weight("rna_iupac")


# --- Null and empty handling, uniformly -----------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        bt.col("s").seq.reverse_complement(),
        bt.col("s").seq.complement(),
        bt.col("s").seq.transcribe(),
        bt.col("s").seq.gc_content(),
        bt.col("s").seq.gc_skew(),
        bt.col("s").seq.base_counts(),
        bt.col("s").seq.translate(),
        bt.col("s").seq.kmers(3),
        bt.col("s").seq.canonical_kmers(3),
        bt.col("s").seq.minimizers(3, 4),
        bt.col("s").seq.melting_temp(),
        bt.col("s").seq.molecular_weight("dna"),
        bt.col("s").seq.gravy(),
        bt.col("s").seq.isoelectric_point(),
        bt.col("s").seq.phred_quality(),
        bt.col("s").seq.mean_quality(),
        bt.col("s").seq.expected_errors(),
        bt.col("s").seq.find_motif("ACGT"),
        bt.col("s").seq.count_motif("ACGT"),
        bt.col("s").seq.max_homopolymer(),
        bt.col("s").seq.is_valid("dna"),
    ],
)
def test_every_op_answers_null_for_a_null_row(expr):
    """One rule, checked across the whole family rather than per function.

    A kernel that dropped the null mask would produce a plausible value for a row that has
    no sequence, and only a filter downstream would ever notice.
    """
    assert _one(expr, [None, None]) == [None, None]


def test_an_all_null_column_does_not_fail_on_its_arrow_type():
    """A column of nothing but nulls arrives typed `Null`, not `Utf8`.

    That shape is routine — an upstream filter that matched nothing, an outer join with no
    partner — and it used to be a type error about a column the caller never typed.
    """
    ds = bt.from_pydict({"s": [None, None, None]})
    out = ds.select(
        rc=bt.col("s").seq.reverse_complement(),
        gc=bt.col("s").seq.gc_content(),
        k=bt.col("s").seq.kmers(3),
    ).to_pydict()
    assert out == {"rc": [None] * 3, "gc": [None] * 3, "k": [None] * 3}
