"""``array_agg(order_by=...)`` and the positions composed over it, vs DuckDB, on every path.

An ordered list is the one aggregate whose answer is a *sequence*, so every comparison here
is order-sensitive on the list: the per-group lists are read into a ``{group: list}`` map
and compared with ``==``, never through ``assert_same``'s multiset view of the rows.

DuckDB leaves rows that tie on every ``ORDER BY`` key in no particular order, and Batcher
breaks those ties by the value, ascending with nulls last. So each DuckDB query spells that
tiebreak as a trailing ``x ASC NULLS LAST`` key: it is the engine's documented rule, and
without it the oracle itself would have no single answer to agree with.

The input is past ``MIN_ROWS_TO_SHARD`` and split into several record batches, and it is
shuffled, so the parallel, spilled and repartitioned paths each see a group's rows in a
different arrival order. That is what makes the paths worth comparing: an ordered
aggregate that fell back to arrival order would disagree with DuckDB on at least one.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_ROWS = 90_000


def _big() -> pa.Table:
    """Shuffled rows with ties and nulls in the key, nulls and duplicates in the value."""
    rng = random.Random(7)
    g = [None if i % 97 == 0 else f"g{i % 23}" for i in range(_ROWS)]
    k = [None if rng.random() < 0.05 else rng.randrange(50) for _ in range(_ROWS)]
    x = [None if rng.random() < 0.05 else rng.randrange(1_000) for _ in range(_ROWS)]
    order = list(range(_ROWS))
    rng.shuffle(order)
    table = pa.table(
        {
            "g": pa.array([g[i] for i in order], pa.string()),
            "k": pa.array([k[i] for i in order], pa.int64()),
            "x": pa.array([x[i] for i in order], pa.int64()),
        }
    )
    return pa.Table.from_batches(table.to_batches(max_chunksize=20_000))


#: The small edge-case table: an all-null value group, an all-null key group, a single row,
#: and a group whose rows tie on the key with a null value among them.
_EDGES = pa.table(
    {
        "g": ["a", "a", "a", "a", "b", "b", "c", "d", "d", None],
        "k": pa.array([2, None, 2, 1, 5, 5, 9, None, None, 3], pa.int64()),
        "x": pa.array([7, 3, None, 4, None, None, 1, 8, 2, 6], pa.int64()),
    }
)

BIG = _big()

#: How each execution path turns a dataset into a table.
PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(num_partitions=4),
    "spill_forced": lambda ds: ds.collect(spill=True, num_partitions=3),
    "iter_batches": lambda ds: pa.Table.from_batches(list(ds.iter_batches())),
    "repartition": lambda ds: ds.repartition(4).collect(),
}

#: (descending, nulls_last) for the one order key.
ORDERINGS = [(False, True), (True, True), (False, False), (True, False)]


def _duck_order(descending: bool, nulls_last: bool) -> str:
    direction = "DESC" if descending else "ASC"
    nulls = "NULLS LAST" if nulls_last else "NULLS FIRST"
    return f"k {direction} {nulls}, x ASC NULLS LAST"


def _lists(table: pa.Table, column: str = "xs") -> dict:
    """``{group: value}`` from a result, so a list is compared as the sequence it is."""
    groups = table.column("g").to_pylist() if "g" in table.column_names else [None]
    return dict(zip(groups, table.column(column).to_pylist(), strict=True))


@pytest.fixture
def tables(duck):
    duck.register("big", BIG)
    duck.register("edges", _EDGES)
    return {"big": BIG, "edges": _EDGES}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize(("descending", "nulls_last"), ORDERINGS)
@pytest.mark.parametrize("name", ["big", "edges"])
def test_grouped_ordered_array_agg_matches_duckdb(duck, tables, name, descending, nulls_last, path):
    agg = col("x").array_agg(order_by="k", descending=descending, nulls_last=nulls_last)
    got = PATHS[path](bt.from_arrow(tables[name]).group_by("g").agg(xs=agg))
    want = duck.sql(
        f"SELECT g, array_agg(x ORDER BY {_duck_order(descending, nulls_last)}) AS xs "
        f"FROM {name} GROUP BY g"
    ).to_arrow_table()
    assert _lists(got) == _lists(want)


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize(("descending", "nulls_last"), ORDERINGS)
def test_global_ordered_array_agg_matches_duckdb(duck, tables, descending, nulls_last, path):
    agg = col("x").array_agg(order_by="k", descending=descending, nulls_last=nulls_last)
    got = PATHS[path](bt.from_arrow(BIG).agg(xs=agg))
    want = duck.sql(
        f"SELECT array_agg(x ORDER BY {_duck_order(descending, nulls_last)}) AS xs FROM big"
    ).to_arrow_table()
    assert _lists(got) == _lists(want)


def test_the_ordering_is_not_the_arrival_order(tables):
    """Positive control: the fixture's arrival order is not the key order, so the tests above
    would fail an aggregate that ignored ``order_by``."""
    ascending = _lists(bt.from_arrow(BIG).agg(xs=col("x").array_agg(order_by="k")).collect())
    descending = _lists(
        bt.from_arrow(BIG).agg(xs=col("x").array_agg(order_by="k", descending=True)).collect()
    )
    arrival = BIG.column("x").to_pylist()
    assert ascending[None] != arrival
    assert ascending[None] != descending[None]
    assert sorted(ascending[None], key=lambda v: (v is None, v)) == sorted(
        arrival, key=lambda v: (v is None, v)
    )


@pytest.mark.parametrize("path", sorted(PATHS))
def test_two_keys_with_per_key_flags(duck, tables, path):
    agg = col("x").array_agg(
        order_by=["g", "k"], descending=[True, False], nulls_last=[False, True]
    )
    got = PATHS[path](bt.from_arrow(BIG).agg(xs=agg))
    want = duck.sql(
        "SELECT array_agg(x ORDER BY g DESC NULLS FIRST, k ASC NULLS LAST, x ASC NULLS LAST) "
        "AS xs FROM big"
    ).to_arrow_table()
    assert _lists(got) == _lists(want)


def test_ignore_nulls_keeps_the_order(duck, tables):
    agg = col("x").array_agg(order_by="k", ignore_nulls=True)
    got = bt.from_arrow(_EDGES).group_by("g").agg(xs=agg).collect()
    want = duck.sql(
        "SELECT g, coalesce(array_agg(x ORDER BY k, x) FILTER (WHERE x IS NOT NULL), []) AS xs "
        "FROM edges GROUP BY g"
    ).to_arrow_table()
    assert _lists(got) == _lists(want)


def test_empty_input_is_null(duck, tables):
    empty = _EDGES.slice(0, 0)
    got = bt.from_arrow(empty).agg(xs=col("x").array_agg(order_by="k")).collect()
    assert got.column("xs").to_pylist() == [None]


def test_the_function_and_group_by_spellings_agree(tables):
    ds = bt.from_arrow(_EDGES)
    fluent = _lists(ds.group_by("g").agg(x=col("x").array_agg(order_by="k")).collect(), "x")
    function = _lists(ds.group_by("g").agg(x=bt.array_agg("x", order_by="k")).collect(), "x")
    reduced = _lists(ds.group_by("g").array_agg("x", order_by="k").collect(), "x")
    assert fluent == function == reduced
    assert fluent["a"] == [4, 7, None, 3]


@pytest.mark.parametrize("path", sorted(PATHS))
def test_sql_array_agg_order_by(duck, tables, path):
    """SQL ``array_agg(x ORDER BY ...)`` lowers to the ordered aggregate, two orders at once."""
    query = (
        "SELECT g, array_agg(x ORDER BY k DESC NULLS FIRST, x) AS xs, "
        "array_agg(x ORDER BY x DESC) AS ys FROM big GROUP BY g"
    )
    got = PATHS[path](bt.sql(query, big=BIG))
    want = duck.sql(
        "SELECT g, array_agg(x ORDER BY k DESC NULLS FIRST, x) AS xs, "
        "array_agg(x ORDER BY x DESC NULLS LAST) AS ys FROM big GROUP BY g"
    ).to_arrow_table()
    assert _lists(got, "xs") == _lists(want, "xs")
    assert _lists(got, "ys") == _lists(want, "ys")


# --- positions: arg_min / arg_max along an order ----------------------------------------------


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize(("descending", "nulls_last"), ORDERINGS)
@pytest.mark.parametrize("name", ["big", "edges"])
def test_arg_min_arg_max_positions_match_duckdb(duck, tables, name, descending, nulls_last, path):
    ordered = {"order_by": "k", "descending": descending, "nulls_last": nulls_last}
    ds = bt.from_arrow(tables[name]).group_by("g")
    got = PATHS[path](ds.agg(lo=col("x").arg_min(**ordered), hi=col("x").arg_max(**ordered)))
    order = _duck_order(descending, nulls_last)
    want = duck.sql(
        f"SELECT g, "
        f"CASE WHEN min(x) IS NULL THEN NULL "
        f"ELSE list_position(array_agg(x ORDER BY {order}), min(x)) - 1 END AS lo, "
        f"CASE WHEN max(x) IS NULL THEN NULL "
        f"ELSE list_position(array_agg(x ORDER BY {order}), max(x)) - 1 END AS hi "
        f"FROM {name} GROUP BY g"
    ).to_arrow_table()
    assert _lists(got, "lo") == _lists(want, "lo")
    assert _lists(got, "hi") == _lists(want, "hi")


def test_arg_min_matches_polars_over_a_sorted_frame(tables):
    """Polars numbers a group's rows in frame order, so over a frame sorted by the key its
    ``arg_min`` is the position along that key."""
    pl = pytest.importorskip("polars")
    data = {"g": ["a", "a", "a", "b", "b"], "k": [3, 1, 2, 2, 1], "x": [5, 9, 1, 4, 4]}
    want = (
        pl.DataFrame(data)
        .sort("g", "k")
        .group_by("g", maintain_order=True)
        .agg(lo=pl.col("x").arg_min(), hi=pl.col("x").arg_max())
        .to_dict(as_series=False)
    )
    got = (
        bt.from_pydict(data)
        .group_by("g")
        .agg(lo=col("x").arg_min(order_by="k"), hi=col("x").arg_max(order_by="k"))
        .sort("g")
        .to_pydict()
    )
    assert got == want


# --- refusals ----------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["arg_min", "arg_max"])
def test_a_position_without_an_order_is_refused(fn):
    with pytest.raises(PlanError, match=f"{fn} depends on row order and requires order_by"):
        getattr(col("x"), fn)()
    with pytest.raises(PlanError, match="requires order_by"):
        getattr(col("x"), fn)(order_by=[])


def test_the_old_positional_key_still_does_not_mean_a_position():
    """``arg_min(key)`` was the value-by-key aggregate; it must not silently become a position."""
    with pytest.raises(TypeError):
        col("x").arg_min("k")  # type: ignore[misc]


def test_a_flag_list_of_the_wrong_length_is_refused():
    with pytest.raises(PlanError, match="descending has 1 flag"):
        col("x").array_agg(order_by=["g", "k"], descending=[True])


def test_order_by_on_another_aggregate_is_refused():
    from batcher.plan.expr_ir import AggExpr

    ds = bt.from_arrow(_EDGES)
    with pytest.raises(PlanError, match="does not take order_by"):
        ds.agg(s=AggExpr("sum", col("x"), order_by=[(col("k"), False, False)]))


def test_an_ordered_array_agg_has_no_window_form():
    with pytest.raises(PlanError, match="no window form"):
        col("x").array_agg(order_by="k").over("g")


def test_an_order_key_must_exist():
    with pytest.raises(PlanError, match="unknown column"):
        bt.from_arrow(_EDGES).agg(xs=col("x").array_agg(order_by="nope"))
