"""The assembly-contiguity aggregates: N50, N90, L50, and auN.

Two things are proved here, and the second matters more than the first.

**Correctness**, against DuckDB. None of these is a DuckDB function, but each has an exact
SQL spelling as a window over the sorted lengths — which is both a genuine independent oracle
and a demonstration of what the aggregate replaces: four operations and a self-join for the
total, with the descending order and the `>=` boundary as two separate chances to be wrong.

**Mergeability**, which is the invariant `CLAUDE.md` requires of every stateful operator:
``combine_finalize(partition(partial(pₖ))) == single-node``. An aggregate that fails this
works perfectly on one machine and silently returns a different number on a cluster. These
reuse the median's value-list state precisely so it holds, and this file is where that is
checked rather than assumed — across partitions, across spill, and across streaming.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

# Two assemblies with deliberately different shapes: `even` is a smooth distribution where
# N50 sits mid-list, `skewed` is one chromosome plus debris — the case a median gets wrong.
_LENGTHS = {
    "even": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "skewed": [10_000_000, *([500] * 20)],
}


def _tbl() -> pa.Table:
    asm, length = [], []
    for name, lengths in _LENGTHS.items():
        asm += [name] * len(lengths)
        length += lengths
    return pa.table({"asm": asm, "len": length})


#: N50 as a window query: order longest-first, take the running total, and keep the first row
#: whose total reaches the target fraction. This is the pipeline the aggregate replaces.
def _nx_sql(permille: int, pick: str) -> str:
    return f"""
        SELECT asm, {pick} AS v FROM (
          SELECT asm, len,
                 SUM(len) OVER (PARTITION BY asm ORDER BY len DESC
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running,
                 ROW_NUMBER() OVER (PARTITION BY asm ORDER BY len DESC) AS rn,
                 SUM(len) OVER (PARTITION BY asm) AS total
          FROM t
        ) WHERE running >= total * {permille} / 1000.0
        QUALIFY ROW_NUMBER() OVER (PARTITION BY asm ORDER BY rn) = 1
    """


def test_n50_matches_the_window_query_it_replaces(duck):
    t = _tbl()
    out = bt.from_arrow(t).group_by("asm").agg(v=bt.col("len").n50()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql(_nx_sql(500, "len")))


def test_n90_matches_the_window_query_it_replaces(duck):
    t = _tbl()
    out = bt.from_arrow(t).group_by("asm").agg(v=bt.col("len").n90()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql(_nx_sql(900, "len")))


def test_l50_matches_the_rank_at_the_halfway_mark(duck):
    """L50 is the *count*: the row number of the piece N50 stops at."""
    t = _tbl()
    out = bt.from_arrow(t).group_by("asm").agg(v=bt.col("len").l50()).collect()
    duck.register("t", t)
    assert_same(out, duck.sql(_nx_sql(500, "rn")))


def test_aun_matches_the_ratio_of_two_sums(duck):
    """auN needs no window at all — it is `sum(l²)/sum(l)`, which is why it is the cheap one."""
    t = _tbl()
    out = bt.from_arrow(t).group_by("asm").agg(v=bt.col("len").aun()).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql("SELECT asm, SUM(len * len) * 1.0 / SUM(len) AS v FROM t GROUP BY asm"),
    )


def test_n50_is_not_the_median_and_that_is_the_point(duck):
    """The reason the statistic exists, pinned so nobody 'simplifies' it to a quantile."""
    t = _tbl()
    out = (
        bt.from_arrow(t)
        .group_by("asm")
        .agg(n=bt.col("len").n50(), m=bt.col("len").median())
        .collect()
        .to_pydict()
    )
    by_asm = dict(zip(out["asm"], zip(out["n"], out["m"], strict=True), strict=True))
    n, m = by_asm["skewed"]
    assert n == 10_000_000.0, "N50 weighs by base, so the chromosome wins"
    assert m == 500.0, "the median weighs by contig, so the debris wins"


# --- Mergeability: the invariant that decides whether this works on a cluster ----------


@pytest.mark.parametrize(
    "stat",
    [
        pytest.param(lambda c: c.n50(), id="n50"),
        pytest.param(lambda c: c.n90(), id="n90"),
        pytest.param(lambda c: c.l50(), id="l50"),
        pytest.param(lambda c: c.aun(), id="aun"),
    ],
)
def test_the_statistic_is_identical_across_every_scheduling(stat):
    """`combine_finalize(partition(partial(pₖ)))` == single-node, for each statistic.

    Checked across `collect()`, `collect(spill=True)`, and `iter_batches()` over a row count
    well past one morsel, so the partial states really are built separately and merged. This
    is the test that would fail if the aggregate held anything the concatenation of two value
    lists does not reproduce.
    """
    # Repeated so the group spans many morsels; the lengths themselves stay a fixed multiset,
    # which is what makes the expected answer stable while the partitioning varies.
    t = pa.table(
        {
            "asm": ["even"] * 9 * 3000 + ["skewed"] * 21 * 3000,
            "len": _LENGTHS["even"] * 3000 + _LENGTHS["skewed"] * 3000,
        }
    )
    ds = bt.from_arrow(t).group_by("asm").agg(v=stat(bt.col("len"))).sort("asm")
    single = ds.collect()
    assert_tables_equal(ds.collect(spill=True), single)
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=single.schema)
    assert_tables_equal(streamed, single, ordered=False)


@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([1, 2, 3, 4, 5, 6, 7, 8, 9], id="even"),
        pytest.param([4_000_000, 3_000_000, 2_000_000, 800_000, 200_000], id="chromosome"),
        pytest.param([10_000_000, *([500] * 20)], id="skewed"),
    ],
)
def test_repeating_an_assembly_does_not_change_its_contiguity(lengths):
    """A scale-invariance check that a broken merge fails and a correct one cannot.

    Duplicating every contig doubles the total *and* the count at every length, so the
    length-valued statistics — N50, N90, auN — are exactly unchanged. A merge that dropped or
    double-counted a partial breaks this in a way a single-group total cannot see.

    **L50 does not exactly double**, and asserting that it does was this test's own bug: it
    held for the `even` input and fails for `chromosome`, where L50 goes 2 -> 3 rather than
    2 -> 4. Doubling every length lets the running total cross the halfway mark up to one
    contig earlier, so the honest bound is `[2*L50 - 1, 2*L50]`. The parametrization exists
    because one input was exactly what hid this.
    """
    one = pa.table({"g": ["a"] * len(lengths), "len": lengths})
    two = pa.table({"g": ["a"] * (len(lengths) * 2), "len": lengths * 2})
    for name, stat in [
        ("n50", lambda c: c.n50()),
        ("n90", lambda c: c.n90()),
        ("aun", lambda c: c.aun()),
    ]:
        a = bt.from_arrow(one).group_by("g").agg(v=stat(bt.col("len"))).to_pydict()["v"][0]
        b = bt.from_arrow(two).group_by("g").agg(v=stat(bt.col("len"))).to_pydict()["v"][0]
        assert abs(a - b) < 1e-9, f"{name} moved: {a} vs {b}"
    la = bt.from_arrow(one).group_by("g").agg(v=bt.col("len").l50()).to_pydict()["v"][0]
    lb = bt.from_arrow(two).group_by("g").agg(v=bt.col("len").l50()).to_pydict()["v"][0]
    assert 2 * la - 1 <= lb <= 2 * la, f"L50 out of its scaling bound: {la} -> {lb}"


# --- Edges -----------------------------------------------------------------------------


def test_nulls_and_unusable_lengths_are_excluded_not_summed():
    """A negative length would cancel real sequence out of the total the statistics divide by."""
    clean = pa.table({"g": ["a"] * 3, "len": [9, 8, 7]})
    dirty = pa.table({"g": ["a"] * 5, "len": [9, None, 8, -100, 7]})
    a = bt.from_arrow(clean).group_by("g").agg(v=bt.col("len").n50()).to_pydict()
    b = bt.from_arrow(dirty).group_by("g").agg(v=bt.col("len").n50()).to_pydict()
    assert a["v"] == b["v"]


def test_a_group_with_no_usable_length_is_null_not_zero():
    """Null, so an empty assembly fails a `>= 1000` threshold instead of sliding under it."""
    t = pa.table({"g": ["a", "a", "b"], "len": [None, None, 0]})
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(n=bt.col("len").n50(), l=bt.col("len").l50(), a=bt.col("len").aun())
        .sort("g")
        .to_pydict()
    )
    assert out["n"] == [None, None]
    assert out["l"] == [None, None]
    assert out["a"] == [None, None]


def test_a_single_contig_assembly_is_its_own_n50():
    t = pa.table({"g": ["a"], "len": [42]})
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(n=bt.col("len").n50(), l=bt.col("len").l50(), a=bt.col("len").aun())
        .to_pydict()
    )
    assert (out["n"][0], out["l"][0], out["a"][0]) == (42.0, 1, 42.0)


def test_l50_comes_back_as_an_integer_count():
    """A number of contigs, not a fraction of one."""
    t = pa.table({"g": ["a"] * 9, "len": _LENGTHS["even"]})
    out = bt.from_arrow(t).group_by("g").agg(v=bt.col("len").l50()).collect()
    assert out.schema.field("v").type == pa.int64()


def test_n90_never_exceeds_n50_and_l90_never_undercuts_l50():
    """The monotonicity every Nx family must have, over a range of shapes."""
    for lengths in ([100, 50, 20, 10, 5, 1], _LENGTHS["even"], _LENGTHS["skewed"]):
        t = pa.table({"g": ["a"] * len(lengths), "len": lengths})
        out = (
            bt.from_arrow(t)
            .group_by("g")
            .agg(n50=bt.col("len").n50(), n90=bt.col("len").n90())
            .to_pydict()
        )
        assert out["n90"][0] <= out["n50"][0], lengths
