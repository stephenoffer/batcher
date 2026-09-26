"""Dataset verbs that had no executed test: sampling, profiling, reshaping, and terminals.

Each verb is held against the DuckDB query its docstring describes. The ones that promise a
row order (`bottom_k` ranked, `gather_every` and `iter_slices` over a sorted input) are
compared row for row; the rest are multisets and use `assert_same`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_LABELLED = pa.table(
    {
        "y": ["a", "b", "a", "c", "a", "b", "a", "c", "b", "a"],
        "x": [9, 4, 2, 7, 5, 1, 8, 3, 6, 0],
        "v": [1.5, None, 2.0, 3.0, None, 4.5, 1.0, 2.5, 0.5, 6.0],
    }
)


@pytest.fixture
def labelled(duck):
    duck.register("t", _LABELLED)
    return duck


# --- sampling -------------------------------------------------------------------------


def test_balance_classes_keeps_the_rarest_class_count_preferring_order_by(labelled):
    got = bt.from_arrow(_LABELLED).balance_classes("y", order_by="x").to_arrow()
    assert_same(
        got,
        labelled.sql(
            "WITH m AS (SELECT min(c) AS m FROM (SELECT count(*) AS c FROM t GROUP BY y)) "
            "SELECT y, x, v FROM (SELECT *, row_number() OVER (PARTITION BY y ORDER BY x) AS rn"
            " FROM t), m WHERE rn <= m.m"
        ),
    )
    counts = bt.from_arrow(got).group_by("y").agg(n=bt.count()).to_pydict()["n"]
    assert set(counts) == {2}, "every class cut to the rarest class's two rows"


@pytest.mark.parametrize("n", [1, 2, 10])
def test_sample_per_group_caps_each_group_at_n_by_order(labelled, n):
    got = bt.from_arrow(_LABELLED).sample_per_group("y", n, order_by="x").to_arrow()
    assert_same(
        got,
        labelled.sql(
            "SELECT y, x, v FROM (SELECT *, row_number() OVER (PARTITION BY y ORDER BY x) AS rn"
            f" FROM t) WHERE rn <= {n}"
        ),
    )


def test_sample_per_group_rejects_a_non_positive_cap():
    with pytest.raises(PlanError):
        bt.from_arrow(_LABELLED).sample_per_group("y", 0)


@pytest.mark.parametrize(("n", "offset"), [(1, 0), (3, 0), (3, 1), (4, 9), (2, 20)])
def test_gather_every_keeps_every_nth_row_of_the_sorted_input(labelled, n, offset):
    got = bt.from_arrow(_LABELLED).sort("x").gather_every(n, offset).to_arrow()
    assert_same_ordered(
        got,
        labelled.sql(
            "SELECT y, x, v FROM (SELECT *, row_number() OVER (ORDER BY x) - 1 AS rn FROM t)"
            f" WHERE rn >= {offset} AND (rn - {offset}) % {n} = 0 ORDER BY x"
        ),
    )


@pytest.mark.parametrize(("n", "offset"), [(0, 0), (2, -1)])
def test_gather_every_rejects_bad_arguments(n, offset):
    with pytest.raises(PlanError):
        bt.from_arrow(_LABELLED).gather_every(n, offset)


def test_bottom_k_returns_the_k_smallest_in_rank_order(labelled):
    got = bt.from_arrow(_LABELLED).bottom_k(4, "x").sort("x").to_arrow()
    assert_same_ordered(got, labelled.sql("SELECT * FROM t ORDER BY x LIMIT 4"))
    assert got.column("x").to_pylist() == [0, 1, 2, 3]


def test_bottom_k_is_top_k_ascending_and_skips_nulls_like_order_by(duck):
    table = pa.table({"x": [5, None, 3, 8, 1, None]})
    duck.register("t", table)
    ds = bt.from_arrow(table)
    got = ds.bottom_k(3, "x").sort("x").to_arrow()
    assert_same_ordered(got, duck.sql("SELECT x FROM t ORDER BY x NULLS LAST LIMIT 3"))
    same = ds.top_k(3, "x", descending=False).sort("x").to_arrow()
    assert got.to_pydict() == same.to_pydict()


@pytest.mark.parametrize("k", [0, 1, 100])
def test_bottom_k_edge_sizes(labelled, k):
    got = bt.from_arrow(_LABELLED).bottom_k(k, "x").sort("x").to_arrow()
    assert_same_ordered(got, labelled.sql(f"SELECT * FROM t ORDER BY x LIMIT {k}"))


# --- profiling ------------------------------------------------------------------------


def test_class_balance_is_each_class_share_of_rows_including_a_null_class(duck):
    table = pa.table({"y": ["a", "a", None, "b", "a", "b", None, "c"]})
    duck.register("t", table)
    got = bt.from_arrow(table).class_balance("y").to_arrow()
    assert_same(
        got, duck.sql("SELECT y, count(*) / sum(count(*)) OVER () AS fraction FROM t GROUP BY y")
    )
    assert sum(got.column("fraction").to_pylist()) == pytest.approx(1.0)


def test_crosstab_counts_co_occurrences_and_leaves_absent_pairs_null(duck):
    table = pa.table({"a": ["x", "x", "y", "z", "x", "z"], "b": ["p", "q", "p", "r", "p", "p"]})
    duck.register("t", table)
    got = bt.from_arrow(table).crosstab("a", "b").to_arrow()
    assert sorted(got.column_names) == ["a", "p", "q", "r"]
    cells = ", ".join(
        f"nullif(count(*) FILTER (WHERE b = '{v}'), 0) AS {v}" for v in ("p", "q", "r")
    )
    assert_same(got, duck.sql(f"SELECT a, {cells} FROM t GROUP BY a"))


def test_drop_constant_columns_drops_what_has_at_most_one_distinct_value(duck):
    table = pa.table(
        {
            "const": [7, 7, 7, 7],
            "const_with_null": [1, None, 1, 1],
            "all_null": pa.array([None] * 4, pa.int64()),
            "varies": [1, 2, 3, 4],
            "label": ["a", "b", "a", "a"],
        }
    )
    duck.register("t", table)
    # SQL's COUNT(DISTINCT) ignores nulls, which is the reading the engine takes.
    distinct = duck.sql(
        "SELECT " + ", ".join(f"count(DISTINCT {c})" for c in table.column_names) + " FROM t"
    ).fetchone()
    want = [c for c, n in zip(table.column_names, distinct, strict=True) if n > 1]
    got = bt.from_arrow(table).drop_constant_columns()
    assert got.columns == want == ["varies", "label"]
    assert_same(got.to_arrow(), duck.sql("SELECT varies, label FROM t"))


_NUMERIC = pa.table(
    {
        "x": [5.0, None, 3.0, 8.0, 1.0, 2.0],
        "y": [1.0, 2.0, None, 4.0, 5.0, 3.0],
        "z": [2.0, 1.0, 3.0, None, 0.0, 7.0],
        "s": ["a", "b", "c", "d", "e", "f"],
    }
)


@pytest.mark.parametrize(
    ("method", "duck_fn"), [("cov_matrix", "covar_samp"), ("corr_matrix", "corr")]
)
@pytest.mark.parametrize("columns", [None, ["z", "x"]])
def test_pairwise_matrix_matches_duckdb_pairwise_aggregates(duck, method, duck_fn, columns):
    duck.register("t", _NUMERIC)
    got = getattr(bt.from_arrow(_NUMERIC), method)(columns).to_arrow()
    cols = columns or ["x", "y", "z"]  # the string column is skipped when not named
    assert got.column_names == ["column", *cols]
    rows = " UNION ALL ".join(
        f"SELECT '{r}' AS column, "
        + ", ".join(f"{duck_fn}({r}, {c}) AS {c}" for c in cols)
        + " FROM t"
        for r in cols
    )
    assert_same(got, duck.sql(rows))


# --- terminals and properties -----------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1, 3, 10, 64])
def test_iter_slices_yields_the_sorted_result_in_order_in_bounded_slices(labelled, n_rows):
    slices = list(bt.from_arrow(_LABELLED).sort("x").iter_slices(n_rows))
    assert all(0 < s.num_rows <= n_rows for s in slices)
    assert len(slices) >= -(-_LABELLED.num_rows // n_rows), "the bound actually split it"
    assert_same_ordered(pa.Table.from_batches(slices), labelled.sql("SELECT * FROM t ORDER BY x"))


def test_iter_slices_of_an_empty_result_yields_no_rows():
    ds = bt.from_arrow(_LABELLED).filter(bt.col("x") > 100)
    assert sum(s.num_rows for s in ds.iter_slices(4)) == 0


def test_to_polars_carries_the_same_rows_and_columns(labelled):
    pl = pytest.importorskip("polars")
    ds = bt.from_arrow(_LABELLED).sort("x")
    frame = ds.to_polars()
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["y", "x", "v"]
    assert frame.to_dict(as_series=False) == ds.to_pydict()
    assert_same_ordered(frame.to_arrow(), labelled.sql("SELECT * FROM t ORDER BY x"))


def _schema() -> pa.Schema:
    return pa.schema([("x", pa.int64())])


def _batches():
    yield pa.record_batch({"x": [1, 2]}, schema=_schema())


def test_is_streaming_reports_an_unbounded_source_through_derived_plans():
    bounded = bt.from_pydict({"x": [1, 2, 3]})
    unbounded = bt.from_batches(_batches, _schema(), bounded=False)
    assert bounded.is_streaming is False
    assert bounded.filter(bt.col("x") > 1).is_streaming is False
    assert unbounded.is_streaming is True
    assert unbounded.filter(bt.col("x") > 1).is_streaming is True
    assert bt.from_batches(_batches, _schema()).is_streaming is False
