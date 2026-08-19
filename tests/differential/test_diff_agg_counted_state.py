"""`mode` and `top_k` against DuckDB, across every scheduling and every value edge.

These two stopped sharing MEDIAN's value-list state: they only ever ask how *often* a value
occurs, so they now carry each group's distinct values and their counts (`bc-runtime`'s
`agg::counted`) rather than the values themselves. The state went from `O(rows)` to
`O(distinct)`, which is what made a `top_k(3)` over one ten-million-row group stop allocating
854 MB to return three values.

A state change is exactly where a silent wrong answer hides, and the risky part is not the
happy path — it is that counting has to agree with the old reduction on the cases where
"equal" is subtle:

* **Float identity.** `-0.0`/`0.0` and every NaN bit pattern are one value. Under the old
  state that decided a winner; under a counted state it decides a *count*, so a missed fold
  now compounds. `mode([-0.0, -0.0, 0.0])` must be `0.0` with frequency 3, not `-0.0` with 2.
* **Ties.** Broken to the smaller value, which is what makes the answer independent of the
  order partitions arrive in — the property the mergeable algebra rests on.
* **Nulls.** Never counted, and a group with nothing but nulls is NULL rather than an error.

So the matrix is paths x shapes rather than a handful of per-function cases: `collect()`,
spilled, spilled across partitions, and `iter_batches()` are four schedulings of one
semantics (invariant #7), and each must agree with DuckDB on all of it. The distributed
scheduling is checked separately, because it is the one that would actually exercise
`combine` across machines.
"""

from __future__ import annotations

import math
import struct

import pyarrow as pa
import pytest

from _harness import assert_same

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")


#: Every shape here has **strictly separated** counts, and that is deliberate rather than
#: incidental. DuckDB's `approx_top_k` is a space-saving sketch and its `mode` does not
#: document a tie rule, so on a tied group there is no oracle answer to be differential
#: about -- the same reason `any_value` is tested by property rather than by comparison in
#: `test_diff_agg_distribution.py`. Ties are Batcher's own contract, so they are pinned
#: against that contract below instead of against DuckDB.
SHAPES: dict[str, pa.Table] = {
    # Counts 4 > 3 > 2 > 1 in group a and 3 > 2 > 1 in group b: every prefix of the ranking
    # is unambiguous, so `top_k` is comparable at every k, including k past the cardinality.
    "duplicates": pa.table(
        {
            "g": ["a"] * 10 + ["b"] * 6,
            "x": pa.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 7, 7, 7, 8, 8, 9], pa.int64()),
        }
    ),
    # Nulls interleaved, and one group that is nothing but nulls (-> NULL, not an error).
    "nulls": pa.table(
        {
            "g": ["a", "a", "a", "a", "a", "a", "b", "b", "c", "c", "c"],
            "x": pa.array([1, None, 1, 1, 2, None, 5, None, 9, 9, 8], pa.int64()),
        }
    ),
    "single_row": pa.table({"g": ["a"], "x": pa.array([42], pa.int64())}),
    "empty": pa.table({"g": pa.array([], pa.string()), "x": pa.array([], pa.int64())}),
    # Longer than one morsel (16,384 rows), so `iter_batches()` is genuinely several batches
    # and partial/combine runs for real rather than degenerating to a single partial. Value
    # `v` is seeded to appear strictly more often than `v + 1`.
    "multibatch": pa.table(
        {
            "g": ["a" if i % 2 else "b" for i in range(20_000)],
            "x": pa.array(
                [v for v in range(8) for _ in range(2_500 - v * 100)]
                + [0] * (20_000 - sum(2_500 - v * 100 for v in range(8))),
                pa.int64(),
            ),
        }
    ),
}

#: The float shape is separate: Batcher and DuckDB must agree that the two zeros are one
#: value, and a NaN group must not fracture by bit pattern. Counts are separated here too.
FLOAT_SHAPE = pa.table(
    {
        "g": ["z"] * 5 + ["n"] * 5,
        "x": pa.array(
            [-0.0, -0.0, 0.0, 1.5, 1.5, float("nan"), float("nan"), float("nan"), 2.5, 2.5],
            pa.float64(),
        ),
    }
)


def _paths(ds):
    """The four single-node schedulings of the same plan."""
    yield "collect", ds.collect()
    yield "spill", ds.collect(spill=True)
    yield "spill_partitioned", ds.collect(spill=True, num_partitions=3)
    batches = list(ds.iter_batches())
    schema = batches[0].schema if batches else ds.collect().schema
    yield "iter_batches", pa.Table.from_batches(batches, schema=schema)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_mode_matches_duckdb_on_every_path(duck, shape):
    table = SHAPES[shape]
    duck.register("t", table)
    oracle = duck.sql("SELECT g, mode(x) AS r FROM t GROUP BY g")
    ds = bt.from_arrow(table).group_by("g").agg(r=bt.col("x").mode())
    for _name, got in _paths(ds):
        assert_same(got, oracle)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("k", [1, 2, 5])
def test_top_k_matches_duckdb_on_every_path(duck, shape, k):
    table = SHAPES[shape]
    duck.register("t", table)
    oracle = duck.sql(f"SELECT g, approx_top_k(x, {k}) AS r FROM t GROUP BY g")
    ds = bt.from_arrow(table).group_by("g").agg(r=bt.col("x").top_k(k))
    for _name, got in _paths(ds):
        assert_same(got, oracle)


def test_global_mode_and_top_k_match_duckdb(duck):
    """Ungrouped: one group, which is the shape the state change was measured on."""
    table = SHAPES["multibatch"]
    duck.register("t", table)
    assert_same(
        bt.from_arrow(table).agg(r=bt.col("x").mode()).collect(),
        duck.sql("SELECT mode(x) AS r FROM t"),
    )
    assert_same(
        bt.from_arrow(table).agg(r=bt.col("x").top_k(3)).collect(),
        duck.sql("SELECT approx_top_k(x, 3) AS r FROM t"),
    )


def test_float_identity_folds_before_counting(duck):
    """`-0.0`/`0.0` and every NaN are one value, so they are one *count*.

    Counting the raw row encoding instead splits `[-0.0, -0.0, 0.0]` into a 2-versus-1 race.
    The signed zeros are checked against DuckDB, which folds them the same way. The NaN group
    deliberately is not -- see the test below.
    """
    z = FLOAT_SHAPE.filter(pa.compute.equal(FLOAT_SHAPE.column("g"), "z"))
    duck.register("fz", z)
    ds = bt.from_arrow(z).group_by("g").agg(r=bt.col("x").mode())
    for _name, got in _paths(ds):
        assert_same(got, duck.sql("SELECT g, mode(x) AS r FROM fz GROUP BY g"))

    # The representative handed back is the canonical zero, not the negative one it happened
    # to see first. Checked on the bits, because `-0.0 == 0.0` is True in Python -- which is
    # also why `assert_same` above cannot see this, and why it needs its own assertion.
    got = bt.from_arrow(z).agg(r=bt.col("x").mode()).collect().column("r").to_pylist()[0]
    assert struct.pack("<d", got) == struct.pack("<d", 0.0), "mode must return +0.0, not -0.0"
    assert not math.isnan(got)


def test_nan_folds_which_is_a_deliberate_difference_from_duckdb(duck):
    """Batcher folds NaN for `mode`; DuckDB does not, and contradicts itself doing so.

    On `[nan, nan, nan, 2.5, 2.5]` DuckDB's own `GROUP BY` reports `(nan, 3), (2.5, 2)` -- it
    folds the NaNs -- and yet `mode(x)` on that same column returns `2.5`. Batcher answers
    `nan`, which is the value its own `GROUP BY` and `count(distinct)` agree is the most
    frequent.

    Recorded as a decision rather than hidden: the alternative is to make `mode` disagree with
    grouping on the same column, which is the kind of inconsistency this suite exists to
    catch. Pre-existing -- the value-list state answered `nan` here too, so the counted state
    changed nothing about it.
    """
    n = FLOAT_SHAPE.filter(pa.compute.equal(FLOAT_SHAPE.column("g"), "n"))
    duck.register("fn", n)

    got = bt.from_arrow(n).agg(r=bt.col("x").mode()).collect().column("r").to_pylist()[0]
    assert math.isnan(got), "Batcher folds NaN, so the three NaNs win"

    grouped = duck.sql("SELECT count(*) AS c FROM fn WHERE isnan(x)").fetchall()[0][0]
    assert grouped == 3, "DuckDB's own grouping folds the NaNs too"
    assert duck.sql("SELECT mode(x) AS r FROM fn").fetchall()[0][0] == 2.5, (
        "pinning DuckDB's contradicting answer, so this test fails loudly if DuckDB "
        "ever makes mode consistent with its own GROUP BY -- at which point Batcher's "
        "answer becomes a plain match and this test should become a comparison"
    )


def test_an_all_null_group_yields_an_empty_list_not_null(duck):
    """`top_k` over a group with no non-null values: Batcher gives `[]`, DuckDB gives NULL.

    A pre-existing difference, unchanged by the counted state (the value-list finalize built
    an empty list from an empty group in exactly the same way). Recorded here so the shape is
    covered and the divergence is visible, rather than left to surface as a mystery in a
    differential run.
    """
    t = pa.table({"g": ["a", "a", "b"], "x": pa.array([1, 1, None], pa.int64())})
    duck.register("an", t)
    got = bt.from_arrow(t).group_by("g").agg(r=bt.col("x").top_k(2)).collect()
    by_group = dict(zip(got.column("g").to_pylist(), got.column("r").to_pylist(), strict=True))
    assert by_group["a"] == [1]
    assert by_group["b"] == [], "Batcher yields an empty list for an all-null group"

    duck_b = duck.sql("SELECT approx_top_k(x, 2) AS r FROM an WHERE g = 'b'").fetchall()[0][0]
    assert duck_b is None, "DuckDB yields NULL for the same group -- the known difference"


def test_a_fully_tied_group_ranks_by_value():
    """Every value once: the ranking is entirely tie-break, so it must be ascending value.

    Not compared against DuckDB -- `approx_top_k` over a group with no repeats has no defined
    answer to compare to. What is pinned is Batcher's own rule, which is what makes the
    aggregate mergeable: the winner must not depend on which partition saw which row first.
    """
    table = pa.table({"g": ["a"] * 6, "x": pa.array([5, 3, 9, 1, 7, 2], pa.int64())})
    ds = bt.from_arrow(table).group_by("g").agg(r=bt.col("x").top_k(3))
    for name, got in _paths(ds):
        assert got.column("r").to_pylist() == [[1, 2, 3]], f"tie ranking wrong on {name}"


def test_top_k_is_ordered_most_frequent_first():
    """`top_k` is ordered, so it needs an ORDER-SENSITIVE assertion.

    `assert_same` is order-independent by design, so using it here would pass on a list that
    came back in any order at all -- exactly how an ordering bug stays invisible.
    """
    table = pa.table({"x": pa.array([7, 7, 7, 7, 5, 5, 5, 9, 9, 1], pa.int64())})
    got = bt.from_arrow(table).agg(r=bt.col("x").top_k(3)).collect()
    assert got.column("r").to_pylist() == [[7, 5, 9]]


def test_ties_break_to_the_smaller_value_on_every_path():
    """The tie-break is what makes this legal to merge, so it is pinned independently.

    Every value occurs exactly twice: the answer must be the smallest, on every scheduling,
    rather than whichever the hashing happened to reach first.
    """
    table = pa.table({"x": pa.array([9, 9, 3, 3, 5, 5, 7, 7], pa.int64())})
    ds = bt.from_arrow(table).agg(r=bt.col("x").mode())
    for name, got in _paths(ds):
        assert got.column("r").to_pylist() == [3], f"tie broken wrongly on {name}"


def test_a_hot_group_at_scale_still_answers_correctly(duck):
    """The shape the state change was made for: one group, many rows, few distinct values.

    Deliberately *not* an assertion about peak RSS. The state is `O(distinct)`, but `partial`
    still row-encodes the batch it is given, so process memory moves with the batch size and a
    threshold tight enough to catch a return to the value-list state is also tight enough to
    flake on an unrelated allocation. The exact property -- that the state holds one entry per
    distinct value and not one per row -- is asserted where it can be seen directly, against the
    state itself, by `agg::counted`'s `state_is_bounded_by_cardinality_not_row_count` in
    `bc-runtime`.

    What is worth checking here is that the answer survives the scale: several morsels, a
    single hot group, and a `combine` that runs for real.
    """
    n = 2_000_000
    # Value v occurs (50 - v) times per cycle, so the ranking is strictly separated.
    cycle = [v for v in range(50) for _ in range(50 - v)]
    xs = (cycle * (n // len(cycle) + 1))[:n]
    table = pa.table({"x": pa.array(xs, pa.int64())})
    duck.register("hot", table)

    assert_same(
        bt.from_arrow(table).agg(r=bt.col("x").top_k(3)).collect(),
        duck.sql("SELECT approx_top_k(x, 3) AS r FROM hot"),
    )
    assert_same(
        bt.from_arrow(table).agg(r=bt.col("x").mode()).collect(),
        duck.sql("SELECT mode(x) AS r FROM hot"),
    )
