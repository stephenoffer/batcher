"""Differential tests for the `.seq` genomics namespace, against DuckDB as the oracle.

DuckDB has no genomics functions, so the oracle here is DuckDB's *general* string vocabulary
spelling out the same definition independently: ``reverse(translate(...))`` for a reverse
complement, ``list_transform(range(...), i -> substring(...))`` for k-mers, arithmetic on
``ascii()`` for Phred decoding. That is a genuinely independent implementation — a different
engine, a different algorithm, written from the definition rather than from the kernel — which
is the whole value of a differential test. Where a definition has no DuckDB spelling at all
(codon translation, IUPAC-degenerate matching, the thermodynamic model), the check lives in
``tests/unit/test_seq_namespace.py`` against biological ground truth instead.

The inputs deliberately mix case, ambiguity codes, the empty string, and NULL, because those
are the four shapes that separate a correct kernel from one that only looks correct on
uppercase ACGT.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

# Mixed case, an ambiguity code, an empty sequence, and a null — every shape a real FASTA
# column contains. `row` keeps the comparison anchored, since `assert_same` is
# order-independent.
_SEQS = ["ATGGCC", "atggcc", "GGCCAT", "ACGTN", "", None]

# Uppercase and unambiguous, for the oracles whose DuckDB spelling is case-sensitive or
# undefined on `N` (GC content, the base tallies, the non-degenerate motif count).
_CLEAN = ["ATGGCC", "GGCCAT", "AAAA", "GCGC", "", None]


def _tbl(seqs: list[str | None]) -> pa.Table:
    return pa.table({"s": pa.array(seqs, pa.string()), "row": list(range(len(seqs)))})


def test_reverse_complement_matches_a_translate_and_reverse(duck):
    """`reverse_complement` == DuckDB `reverse(translate(s, 'ACGTacgt', 'TGCAtgca'))`."""
    t = _tbl(_SEQS)
    out = (
        bt.from_arrow(t).select(v=bt.col("s").seq.reverse_complement(), row=bt.col("row")).collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT reverse(translate(s, 'ACGTacgtRYKMBDHVrykmbdhv', "
            "'TGCAtgcaYRMKVHDByrmkvhdb')) AS v, row FROM t"
        ),
    )


def test_complement_matches_a_translate(duck):
    """`complement` == DuckDB `translate` over the same IUPAC pairing, case preserved."""
    t = _tbl(_SEQS)
    out = bt.from_arrow(t).select(v=bt.col("s").seq.complement(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT translate(s, 'ACGTacgtRYKMBDHVrykmbdhv', "
            "'TGCAtgcaYRMKVHDByrmkvhdb') AS v, row FROM t"
        ),
    )


def test_transcribe_matches_a_t_to_u_translate(duck):
    """`transcribe` == DuckDB `translate(s, 'Tt', 'Uu')`."""
    t = _tbl(_SEQS)
    out = bt.from_arrow(t).select(v=bt.col("s").seq.transcribe(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT translate(s, 'Tt', 'Uu') AS v, row FROM t"))


def test_back_transcribe_matches_a_u_to_t_translate(duck):
    """`back_transcribe` == DuckDB `translate(s, 'Uu', 'Tt')`."""
    t = _tbl(["AUGGCC", "auggcc", "ACGT", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.back_transcribe(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT translate(s, 'Uu', 'Tt') AS v, row FROM t"))


def test_gc_content_matches_a_delete_and_measure(duck):
    """`gc_content` == the fraction of bases left standing after deleting the A/T ones.

    Restricted to unambiguous uppercase input, because that is where the two definitions
    coincide: the engine excludes `N` from the denominator and DuckDB's `len` cannot.
    """
    t = _tbl(_CLEAN)
    out = bt.from_arrow(t).select(v=bt.col("s").seq.gc_content(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(replace(replace(s, 'A', ''), 'T', '')) * 1.0 "
            "/ nullif(len(s), 0) AS v, row FROM t"
        ),
    )


def test_gc_skew_matches_the_ratio_written_out(duck):
    """`gc_skew` == `(G - C) / (G + C)` counted by delete-and-measure."""
    t = _tbl(_CLEAN)
    out = bt.from_arrow(t).select(v=bt.col("s").seq.gc_skew(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT (g - c) * 1.0 / nullif(g + c, 0) AS v, row FROM ("
            "  SELECT len(s) - len(replace(s, 'G', '')) AS g,"
            "         len(s) - len(replace(s, 'C', '')) AS c, row FROM t)"
        ),
    )


def test_base_counts_match_a_delete_and_measure_per_base(duck):
    """Each field of `base_counts` == DuckDB's delete-and-measure for that base."""
    t = _tbl(_CLEAN)
    counts = bt.col("s").seq.base_counts()
    out = (
        bt.from_arrow(t)
        .select(
            a=counts.struct.field("a"),
            c=counts.struct.field("c"),
            g=counts.struct.field("g"),
            row=bt.col("row"),
        )
        .collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(s) - len(replace(s, 'A', '')) AS a,"
            "       len(s) - len(replace(s, 'C', '')) AS c,"
            "       len(s) - len(replace(s, 'G', '')) AS g, row FROM t"
        ),
    )


def test_kmers_match_a_sliding_substring(duck):
    """`kmers(3)` == `list_transform(range(1, len-1), i -> substring(s, i, 3))`.

    The empty-list case matters and is covered by the short and empty rows: DuckDB's `range`
    goes empty exactly where the engine's window loop does.
    """
    t = _tbl(["ACGTA", "AC", "GGGG", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.kmers(3), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_transform(range(1, len(s) - 1), i -> substring(s, i, 3)) AS v, row FROM t"
        ),
    )


def test_kmers_upper_case_their_input(duck):
    """`kmers` folds case, so the oracle upper-cases before slicing."""
    t = _tbl(["acgta", "AcGtA"])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.kmers(3), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_transform(range(1, len(s) - 1), "
            "i -> upper(substring(s, i, 3))) AS v, row FROM t"
        ),
    )


def test_count_motif_matches_a_delete_and_measure(duck):
    """`count_motif` of a literal, non-self-overlapping motif == delete-and-measure.

    The motif `GAATTC` (the EcoRI site) cannot overlap itself, so the two definitions of
    "how many occurrences" agree; the overlapping case has no DuckDB spelling and is pinned in
    the unit tests instead.
    """
    t = _tbl(["GGAATTCC", "GAATTCGAATTC", "ACGT", "", None])
    out = (
        bt.from_arrow(t)
        .select(v=bt.col("s").seq.count_motif("GAATTC"), row=bt.col("row"))
        .collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql("SELECT (len(s) - len(replace(s, 'GAATTC', ''))) / 6 AS v, row FROM t"),
    )


def test_find_motif_matches_a_position_scan(duck):
    """`find_motif` == every 1-based offset where the substring equals the motif."""
    t = _tbl(["GGAATTCC", "GAATTCGAATTC", "ACGT", "", None])
    out = (
        bt.from_arrow(t).select(v=bt.col("s").seq.find_motif("GAATTC"), row=bt.col("row")).collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_filter(range(1, len(s) + 1), "
            "i -> substring(s, i, 6) = 'GAATTC') AS v, row FROM t"
        ),
    )


def test_phred_quality_matches_ascii_arithmetic(duck):
    """`phred_quality` == `ascii(char) - 33` per position."""
    t = _tbl(["!5I", "IIII", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.phred_quality(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_transform(range(1, len(s) + 1), "
            "i -> ascii(substring(s, i, 1)) - 33) AS v, row FROM t"
        ),
    )


def test_mean_quality_matches_the_average_of_those_scores(duck):
    """`mean_quality` == `list_avg` of the same decoded scores."""
    t = _tbl(["!5I", "IIII", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.mean_quality(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_avg(list_transform(range(1, len(s) + 1), "
            "i -> ascii(substring(s, i, 1)) - 33)) AS v, row FROM t"
        ),
    )


def test_mean_quality_honours_a_non_default_offset(duck):
    """The offset shifts every score, which is why it is stated rather than sniffed."""
    t = _tbl(["IIII", "abcd", None])
    out = (
        bt.from_arrow(t)
        .select(v=bt.col("s").seq.mean_quality(offset=64), row=bt.col("row"))
        .collect()
    )
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT list_avg(list_transform(range(1, len(s) + 1), "
            "i -> ascii(substring(s, i, 1)) - 64)) AS v, row FROM t"
        ),
    )


def test_expected_errors_matches_the_summed_error_probabilities(duck):
    """`expected_errors` == `sum(10 ** (-Q/10))`, the `fastq_maxee` quantity."""
    t = _tbl(["!5I", "IIII", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.expected_errors(), row=bt.col("row")).collect()
    duck.register("t", t)
    # The `coalesce` turns an *empty* string's empty list into 0.0, which is what the engine
    # answers — but it would do the same to a NULL row, where the engine answers NULL. The
    # outer CASE keeps the two apart; without it this oracle asserts the opposite of the
    # null-propagation rule every other op here follows.
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN s IS NULL THEN NULL ELSE "
            "coalesce(list_sum(list_transform(range(1, len(s) + 1), "
            "i -> pow(10, -(ascii(substring(s, i, 1)) - 33) / 10.0))), 0.0) END AS v, row FROM t"
        ),
    )


def test_max_homopolymer_matches_a_run_length_scan(duck):
    """`max_homopolymer` == the longest run, found by comparing each base to its predecessor."""
    t = _tbl(["ACGT", "AAAATG", "AaaaTG", "GGCCCCA", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.max_homopolymer(), row=bt.col("row")).collect()
    duck.register("t", t)
    # For each start position, how far the run of identical (case-folded) bases extends; the
    # answer is the largest such run. Written as a nested list comprehension rather than a
    # window function so it stays a per-row expression, matching the engine's shape.
    # As above: `coalesce` gives the *empty* string 0, which is right, and would give a NULL
    # row 0 too, which is not.
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN s IS NULL THEN NULL ELSE "
            "coalesce(list_max(list_transform(range(1, len(s) + 1), i -> "
            "  len(list_filter(range(i, len(s) + 1), j -> "
            "    upper(substring(s, j, 1)) = upper(substring(s, i, 1)) AND "
            "    len(list_filter(range(i, j + 1), m -> "
            "      upper(substring(s, m, 1)) != upper(substring(s, i, 1)))) = 0)))), 0) "
            "END AS v, row FROM t"
        ),
    )


def test_is_valid_matches_a_character_class_check(duck):
    """`is_valid('dna')` == "nothing survives deleting every ACGT character"."""
    t = _tbl(["ACGT", "acgt", "ACGN", "hello", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").seq.is_valid("dna"), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql("SELECT len(regexp_replace(s, '[ACGTacgt]', '', 'g')) = 0 AS v, row FROM t"),
    )


def test_seq_ops_agree_across_the_execution_schedulings(duck):
    """`collect()`, `collect(spill=True)` and `iter_batches()` produce the same rows.

    `.seq` is a scalar projection, so no scheduling can change it — which is exactly why
    this is worth pinning: a kernel that read per-batch state instead of per-row would pass
    every single-batch test and split its answers at a morsel boundary. The row count is
    pushed past one morsel so the streaming path really does hand over more than one batch.
    """
    t = _tbl(_SEQS * 4000)
    ds = bt.from_arrow(t).select(
        rc=bt.col("s").seq.reverse_complement(),
        gc=bt.col("s").seq.gc_content(),
        k=bt.col("s").seq.kmers(3),
        row=bt.col("row"),
    )
    single = ds.collect()
    assert_tables_equal(ds.collect(spill=True), single)
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=single.schema)
    assert_tables_equal(streamed, single)

    duck.register("t", t)
    assert_same(
        single.select(["rc", "gc", "row"]),
        duck.sql(
            "SELECT reverse(translate(s, 'ACGTacgtRYKMBDHVrykmbdhv', "
            "'TGCAtgcaYRMKVHDByrmkvhdb')) AS rc, "
            "len(replace(replace(replace(upper(s), 'A', ''), 'T', ''), 'N', '')) * 1.0 "
            "/ nullif(len(replace(upper(s), 'N', '')), 0) AS gc, row FROM t"
        ),
    )
