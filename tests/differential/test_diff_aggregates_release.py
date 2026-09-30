"""Aggregates and per-group callbacks that had no executed test, against DuckDB.

`array_agg`, `bool_and` and `bool_or` are exact and compared exactly. `approx_median` and
`approx_quantile` are sketches, so each is held to the *exact* DuckDB quantile within a
stated tolerance. `GroupBy.map_groups` is compared on its one promise that a plain
`map_batches` cannot keep: one call per whole group, even over a many-batch input.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

# --- approximate quantiles --------------------------------------------------------------

_N = 20_000
# A deterministic permutation of 0.._N-1 per group, so each value equals its rank and a
# value error is a rank error in the same units. Group "c" is all null.
_PERM = [(i * 7919) % _N for i in range(_N)]
_QUANTILE_TABLE = pa.table(
    {
        "k": ["a"] * _N + ["b"] * _N + ["c"] * 3,
        "x": [float(v) for v in _PERM] + [float(v) * 2.0 for v in _PERM] + [None] * 3,
    }
)

#: Sketch rank error allowed, as a fraction of the group's rows. The quantile sketches
#: promise roughly 1% rank error (`.claude/rules/rust-engine.md` records a worst observed
#: 0.0097 for KLL); 2% leaves headroom without letting a wrong quantile through: the
#: nearest wrong answers these tests guard against (q vs 1-q, a median of the wrong group)
#: are at least 20% of the range away.
_RANK_TOL = 0.02


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_approx_quantile_is_within_rank_tolerance_of_the_exact_quantile(duck, q):
    duck.register("t", _QUANTILE_TABLE)
    got = dict(
        zip(
            *bt.from_arrow(_QUANTILE_TABLE)
            .group_by("k")
            .agg(v=bt.approx_quantile("x", q))
            .to_pydict()
            .values(),
            strict=True,
        )
    )
    exact = dict(duck.execute(f"SELECT k, quantile_cont(x, {q}) FROM t GROUP BY k").fetchall())
    assert got.keys() == exact.keys() == {"a", "b", "c"}
    assert got["c"] is None and exact["c"] is None, "an all-null group has no quantile"
    for k, scale in (("a", 1.0), ("b", 2.0)):
        assert abs(got[k] - exact[k]) <= _RANK_TOL * _N * scale, (k, got[k], exact[k])


def test_approx_median_is_the_half_quantile_within_tolerance(duck):
    duck.register("t", _QUANTILE_TABLE)
    got = (
        bt.from_arrow(_QUANTILE_TABLE)
        .group_by("k")
        .agg(m=bt.approx_median("x"), q=bt.approx_quantile("x", 0.5))
        .sort("k")
        .to_pydict()
    )
    exact = [r[0] for r in duck.execute("SELECT median(x) FROM t GROUP BY k ORDER BY k").fetchall()]
    assert got["k"] == ["a", "b", "c"]
    assert got["m"][2] is None and exact[2] is None
    for i, scale in ((0, 1.0), (1, 2.0)):
        assert abs(got["m"][i] - exact[i]) <= _RANK_TOL * _N * scale, (got["m"][i], exact[i])
    assert got["m"] == got["q"], "approx_median is approx_quantile(0.5)"


def test_approx_quantiles_on_one_row_and_on_no_rows(duck):
    one = bt.from_pydict({"x": [4.5]}).agg(m=bt.approx_median("x"), q=bt.approx_quantile("x", 0.9))
    assert one.to_pydict() == {"m": [4.5], "q": [4.5]}
    empty = pa.table({"x": pa.array([], pa.float64())})
    duck.register("e", empty)
    got = bt.from_arrow(empty).agg(m=bt.approx_median("x")).to_arrow()
    assert_same(got, duck.sql("SELECT median(x) AS m FROM e"))


# --- array_agg --------------------------------------------------------------------------

_LISTS = pa.table(
    {
        "k": ["a", "a", "b", "a", "c", "b", "a", "d", "d"],
        "i": [3, 1, 2, 0, 5, 4, 2, 6, 7],
        "x": [1.5, None, 7.0, 2.5, 9.0, 7.0, -1.0, None, None],
    }
)


@pytest.fixture
def lists(duck):
    duck.register("t", _LISTS)
    return duck


def _by_key(table: pa.Table, col: str) -> dict:
    d = table.to_pydict()
    return dict(zip(d["k"], d[col], strict=True))


@pytest.mark.parametrize("descending", [False, True])
def test_array_agg_with_order_by_matches_duckdb_element_order(lists, descending):
    got = (
        bt.from_arrow(_LISTS)
        .group_by("k")
        .agg(l=bt.array_agg("x", order_by="i", descending=descending))
        .to_arrow()
    )
    direction = "DESC" if descending else "ASC"
    want = lists.sql(f"SELECT k, array_agg(x ORDER BY i {direction}) AS l FROM t GROUP BY k")
    # Compared per key, element for element: a list is ordered even though groups are not.
    assert _by_key(got, "l") == _by_key(want.to_arrow_table(), "l")


def test_array_agg_ignore_nulls_drops_them_and_leaves_an_all_null_group_empty(lists):
    got = (
        bt.from_arrow(_LISTS)
        .group_by("k")
        .agg(l=bt.array_agg("x", order_by="i", ignore_nulls=True))
        .to_arrow()
    )
    with_nulls = _by_key(
        lists.sql("SELECT k, array_agg(x ORDER BY i) AS l FROM t GROUP BY k").to_arrow_table(), "l"
    )
    # Spark's collect_list, as documented: nulls left out, an all-null group is [].
    want = {k: [v for v in vs if v is not None] for k, vs in with_nulls.items()}
    assert _by_key(got, "l") == want
    assert want["d"] == []


def test_array_agg_without_order_by_keeps_each_group_multiset(lists):
    got = bt.from_arrow(_LISTS).group_by("k").agg(l=bt.array_agg("x")).to_arrow()
    want = _by_key(lists.sql("SELECT k, array_agg(x) AS l FROM t GROUP BY k").to_arrow_table(), "l")

    def key(v):
        return (v is None, v or 0.0)

    got_sorted = {k: sorted(vs, key=key) for k, vs in _by_key(got, "l").items()}
    assert got_sorted == {k: sorted(vs, key=key) for k, vs in want.items()}
    assert None in got_sorted["a"], "SQL keeps nulls in the list"


def test_array_agg_over_no_rows_is_null_like_duckdb(duck):
    empty = pa.table({"x": pa.array([], pa.int64())})
    duck.register("e", empty)
    got = bt.from_arrow(empty).agg(l=bt.array_agg("x")).to_arrow()
    assert_same(got, duck.sql("SELECT array_agg(x) AS l FROM e"))


# --- bool_and / bool_or -----------------------------------------------------------------

_BOOLS = pa.table(
    {
        "k": ["a", "a", "a", "b", "b", "c", "c", "d"],
        "v": [True, None, True, True, False, None, None, False],
    }
)


def test_bool_and_bool_or_match_duckdb_with_nulls_and_an_all_null_group(duck):
    duck.register("t", _BOOLS)
    got = (
        bt.from_arrow(_BOOLS)
        .group_by("k")
        .agg(all_=bt.bool_and("v"), any_=bt.bool_or("v"))
        .to_arrow()
    )
    assert_same(
        got, duck.sql("SELECT k, bool_and(v) AS all_, bool_or(v) AS any_ FROM t GROUP BY k")
    )
    assert _by_key(got, "all_")["c"] is None, "the all-null group keeps SQL's null"


def test_bool_aggregates_empty_value_answers_the_all_null_group_only(duck):
    duck.register("t", _BOOLS)
    got = (
        bt.from_arrow(_BOOLS)
        .group_by("k")
        .agg(
            all_=bt.bool_and("v", empty_value=True),
            any_=bt.bool_or("v", empty_value=False),
        )
        .to_arrow()
    )
    assert_same(
        got,
        duck.sql(
            "SELECT k, coalesce(bool_and(v), true) AS all_, coalesce(bool_or(v), false) AS any_"
            " FROM t GROUP BY k"
        ),
    )


def test_bool_aggregates_over_an_expression_and_on_no_rows(duck):
    table = pa.table({"x": [3, -1, None, 8]})
    duck.register("t", table)
    ds = bt.from_arrow(table)
    got = ds.agg(a=bt.bool_and(bt.col("x") > 0), o=bt.bool_or(bt.col("x") > 0)).to_arrow()
    assert_same(got, duck.sql("SELECT bool_and(x > 0) AS a, bool_or(x > 0) AS o FROM t"))
    none = ds.filter(bt.col("x") > 100)
    got = none.agg(a=bt.bool_and(bt.col("x") > 0), o=bt.bool_or(bt.col("x") > 0)).to_arrow()
    assert_same(
        got, duck.sql("SELECT bool_and(x > 0) AS a, bool_or(x > 0) AS o FROM t WHERE x > 100")
    )


# --- GroupBy.map_groups -----------------------------------------------------------------


def _many_batches(rows: int, batch: int) -> pa.Table:
    """A table split into many small record batches, so a group spans several of them."""
    keys = [f"g{i % 7}" for i in range(rows)]
    vals = [(i * 31) % 1000 for i in range(rows)]
    whole = pa.table({"k": keys, "v": vals})
    return pa.Table.from_batches(whole.to_batches(max_chunksize=batch))


def _summary(group: pa.RecordBatch) -> dict:
    v = group.column("v").to_pylist()
    return {
        "k": [group.column("k")[0].as_py()],
        "n": [len(v)],
        "spread": [max(v) - min(v)],
    }


def test_map_groups_calls_once_per_whole_group_across_many_batches(duck):
    table = _many_batches(7_000, 64)
    assert table.column("k").num_chunks > 50, "positive control: groups span many batches"
    duck.register("t", table)
    got = (
        bt.from_arrow(table)
        .group_by("k")
        .map_groups(_summary, output_columns=["k", "n", "spread"])
        .to_arrow()
    )
    assert_same(
        got, duck.sql("SELECT k, count(*) AS n, max(v) - min(v) AS spread FROM t GROUP BY k")
    )


def test_map_groups_concatenates_groups_of_different_output_lengths(duck):
    table = pa.table({"k": ["a", "b", "a", "c", "a", "b"], "v": [5, 1, 3, 9, 4, 2]})
    duck.register("t", table)

    def top_two(group: pa.RecordBatch) -> pa.RecordBatch:
        order = pc.sort_indices(group, sort_keys=[("v", "descending")])
        return group.take(order[:2])

    got = bt.from_arrow(table).group_by("k").map_groups(top_two).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT k, v FROM (SELECT *, row_number() OVER (PARTITION BY k ORDER BY v DESC) AS r"
            " FROM t) WHERE r <= 2"
        ),
    )
    assert got.num_rows == 5, "a:2 + b:2 + c:1 — a one-row group keeps its one row"


def test_map_groups_rejects_a_frame_that_is_all_group_keys():
    with pytest.raises(PlanError):
        bt.from_pydict({"k": ["a"]}).group_by("k").map_groups(lambda b: b).to_arrow()
