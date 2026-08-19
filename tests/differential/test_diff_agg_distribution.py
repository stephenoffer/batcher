"""The distribution aggregates added this wave, against DuckDB.

`entropy`, `mad`, `kurtosis_pop` and `quantile_disc` all read a group's whole value list, so
they share MEDIAN's mergeable state and differ only in how they finalize it. `approx_top_k` is
here too, but no longer for that reason: it counts values rather than keeping them
(`bc-runtime`'s `agg::counted`, with its own matrix in `test_diff_agg_counted_state.py`).

Each is checked against DuckDB's own answer, grouped and global, and each is checked to give
the *same* answer through the multi-partition path -- an aggregate that is not
partition-order-independent is the classic silent distributed bug.

`any_value` is the exception, and is tested differently: DuckDB documents the chosen row
as unspecified, so there is no answer to be differential about. What is pinned is that
the engine's choice is a member of the group and is stable across execution modes.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def skewed(duck):
    t = pa.table(
        {
            "g": ["a"] * 7 + ["b"] * 5,
            "x": [1, 2, 2, 3, 3, 3, 10, 4, 4, 5, 6, 100],
        }
    )
    duck.register("skewed", t)
    return t


GROUPED = [
    "SELECT g, entropy(x) AS r FROM skewed GROUP BY g",
    "SELECT g, mad(x) AS r FROM skewed GROUP BY g",
    "SELECT g, kurtosis_pop(x) AS r FROM skewed GROUP BY g",
    "SELECT g, quantile_disc(x, 0.5) AS r FROM skewed GROUP BY g",
    "SELECT g, quantile_disc(x, 0.25) AS r FROM skewed GROUP BY g",
    "SELECT g, quantile_disc(x, 0.9) AS r FROM skewed GROUP BY g",
]

GLOBAL = [
    "SELECT entropy(x) AS r FROM skewed",
    "SELECT mad(x) AS r FROM skewed",
    "SELECT kurtosis_pop(x) AS r FROM skewed",
    "SELECT quantile_disc(x, 0.5) AS r FROM skewed",
]


@pytest.mark.parametrize("q", GROUPED + GLOBAL)
def test_distribution_aggregates_match_duckdb(duck, skewed, q):
    assert_same(bt.sql(q, skewed=skewed).collect(), duck.sql(q))


def test_approx_top_k_matches_duckdb(duck, skewed):
    q = "SELECT approx_top_k(x, 2) AS r FROM skewed WHERE g = 'a'"
    assert_same(bt.sql(q, skewed=skewed).collect(), duck.sql(q))


def test_approx_top_k_is_ordered_most_frequent_first(skewed):
    # DuckDB's own ordering, and the tie-break that makes it deterministic: equal
    # frequencies rank by the smaller value, so a partition split cannot reorder it.
    q = "SELECT approx_top_k(x, 3) AS r FROM skewed WHERE g = 'a'"
    assert bt.sql(q, skewed=skewed).to_pydict()["r"][0] == [3, 2, 1]


def test_entropy_is_zero_for_a_constant_group():
    ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [7, 7, 7]})
    assert ds.group_by("g").agg(r=bt.col("x").entropy()).to_pydict()["r"] == [0.0]


def test_entropy_of_n_distinct_values_is_log2_n():
    ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 4]})
    assert ds.group_by("g").agg(r=bt.col("x").entropy()).to_pydict()["r"] == [2.0]


def test_mad_ignores_an_outlier_a_stddev_would_follow():
    ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 1000]})
    out = ds.group_by("g").agg(m=bt.col("x").mad(), s=bt.col("x").std()).to_pydict()
    assert out["m"] == [1.0]
    assert out["s"][0] > 400


def test_quantile_disc_returns_a_value_that_is_present_where_quantile_interpolates():
    ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 4]})
    out = (
        ds.group_by("g")
        .agg(d=bt.col("x").quantile_disc(0.5), c=bt.col("x").quantile(0.5))
        .to_pydict()
    )
    assert out["d"] == [2.0]  # an element of the input
    assert out["c"] == [2.5]  # between two of them


def test_any_value_is_a_member_of_the_group_and_stable(skewed):
    out = bt.from_arrow(skewed).group_by("g").agg(r=bt.col("x").any_value()).sort("g").to_pydict()
    values = dict(zip(out["g"], out["r"], strict=True))
    assert values["a"] in {1, 2, 3, 10}
    assert values["b"] in {4, 5, 6, 100}
    # The engine resolves "unspecified" to the group minimum so a partitioned run agrees
    # with a single-node one; pinned so the choice cannot drift silently.
    assert values == {"a": 1, "b": 4}


def test_any_value_and_arbitrary_are_the_same_function(duck, skewed):
    q = "SELECT any_value(x) AS a, arbitrary(x) AS b FROM skewed"
    out = bt.sql(q, skewed=skewed).to_pydict()
    assert out["a"] == out["b"]


@pytest.mark.parametrize(
    "agg",
    [
        "entropy(x)",
        "mad(x)",
        "kurtosis_pop(x)",
        "quantile_disc(x, 0.5)",
        "any_value(x)",
    ],
)
def test_the_multi_partition_result_equals_the_single_node_one(skewed, agg):
    # The mergeability invariant, exercised through the engine: the same query over a
    # relation split into many morsels must produce the identical relation.
    q = f"SELECT g, {agg} AS r FROM skewed GROUP BY g"
    one = bt.sql(q, skewed=skewed).sort("g").to_pydict()
    many = bt.sql(q, skewed=bt.from_arrow(skewed).repartition(4).collect()).sort("g").to_pydict()
    assert one == many


# --- compensated summation --------------------------------------------------------------

KAHAN_QUERIES = [
    "SELECT fsum(x) AS r FROM skewed",
    "SELECT kahan_sum(x) AS r FROM skewed",
    "SELECT sumkahan(x) AS r FROM skewed",
    "SELECT g, fsum(x) AS r FROM skewed GROUP BY g",
]


@pytest.mark.parametrize("q", KAHAN_QUERIES)
def test_compensated_sum_matches_duckdb(duck, skewed, q):
    assert_same(bt.sql(q, skewed=skewed).collect(), duck.sql(q))


def test_the_compensated_sum_is_right_where_the_plain_one_drifts(duck):
    # The case the aggregate exists for: addends far below the running total lose their
    # low bits in a plain float sum. Both numbers asserted, and DuckDB agrees on both.
    t = pa.table({"x": [1e16, 1.0, 1.0, -1e16]})
    duck.register("drift", t)
    out = (
        bt.from_arrow(t).select(exact=bt.col("x").kahan_sum(), naive=bt.col("x").sum()).to_pydict()
    )
    assert out == {"exact": [2.0], "naive": [0.0]}
    assert duck.sql("SELECT fsum(x), sum(x) FROM drift").fetchall() == [(2.0, 0.0)]


def test_the_compensated_sum_survives_repartitioning():
    # The compensation has to merge, not just accumulate: a partitioned run folds several
    # (sum, c) states together, and the correction must be applied exactly once.
    t = pa.table({"g": ["a"] * 4, "x": [1e16, 1.0, 1.0, -1e16]})
    one = bt.from_arrow(t).group_by("g").agg(r=bt.col("x").kahan_sum()).to_pydict()
    many = bt.from_arrow(t).repartition(3).group_by("g").agg(r=bt.col("x").kahan_sum()).to_pydict()
    assert one == many == {"g": ["a"], "r": [2.0]}
