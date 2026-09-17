"""The `.list` parameters that restore another engine's meaning, and the list bugfix, vs DuckDB.

Each default is DuckDB's list function, pinned beside the parameter. The parameterised form
is checked against the DuckDB spelling of the same meaning: ``list_sort``'s null-order
argument, the ``||`` operator (which, unlike ``list_concat``, propagates a null list),
``v IN (SELECT unnest(l))`` for Spark's three-valued ``array_contains``, and a lambda over
the positions for the non-zero Jaccard index and for an exact integer first difference. The
fixture holds an empty list, a null list, an all-null list, duplicates, ``-0.0``/``0.0``,
and 64-bit boundaries.

A sort within a list is compared as a list value, so element order is checked; the rows
carry an ``id`` so each list is compared with its own oracle row. The competitor half is
`test_diff_scalar_list_competitor_params.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

_INTS = [[3, None, 1, 1], [], None, [None], [None, None, 2], [5, -5, 0, 5]]
# No NaN here: the oracle compares list values with `==`, under which NaN never equals itself.
# NaN's place in a sort and a distinct is pinned by `eval::list` unit tests.
_FLOATS = [[1.0, -0.0, 0.0, None, 2.5], [], None, [None, -1.5, None]]


@pytest.fixture
def ints(duck):
    tbl = pa.table({"id": list(range(len(_INTS))), "l": pa.array(_INTS, pa.list_(pa.int64()))})
    duck.register("li", tbl)
    return tbl


@pytest.fixture
def floats(duck):
    tbl = pa.table(
        {"id": list(range(len(_FLOATS))), "l": pa.array(_FLOATS, pa.list_(pa.float64()))}
    )
    duck.register("lf", tbl)
    return tbl


# --- sort(descending=, nulls_last=) ---------------------------------------------------------


@pytest.mark.parametrize("table", ["li", "lf"])
def test_sort_null_placement_and_direction_match_list_sort(duck, ints, floats, table):
    source = ints if table == "li" else floats
    out = bt.from_arrow(source).select(
        "id",
        asc=col("l").list.sort(),
        asc_first=col("l").list.sort(nulls_last=False),
        desc=col("l").list.sort(descending=True),
        desc_first=col("l").list.sort(descending=True, nulls_last=False),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, list_sort(l, 'ASC', 'NULLS LAST') AS asc, "
            "list_sort(l, 'ASC', 'NULLS FIRST') AS asc_first, "
            "list_sort(l, 'DESC', 'NULLS LAST') AS desc, "
            f"list_sort(l, 'DESC', 'NULLS FIRST') AS desc_first FROM {table}"
        ),
    )


# --- unique(drop_nulls=) / n_unique(count_nulls=) -------------------------------------------


@pytest.mark.parametrize("table", ["li", "lf"])
def test_unique_and_n_unique_null_handling(duck, ints, floats, table):
    source = ints if table == "li" else floats
    got = bt.from_arrow(source).select(
        "id",
        n=col("l").list.n_unique(),
        n_null=col("l").list.n_unique(count_nulls=True),
        # Order-free compare: DuckDB's `list_distinct` does not promise first-seen order.
        u=col("l").list.unique().list.sort(),
        u_null=col("l").list.unique(drop_nulls=False).list.sort(),
    )
    has_null = "(len(list_filter(l, x -> x IS NULL)) > 0)"
    assert_same(
        got.collect(),
        duck.sql(
            f"SELECT id, list_unique(l) AS n, list_unique(l) + {has_null}::BIGINT AS n_null, "
            "list_sort(list_distinct(l)) AS u, "
            f"CASE WHEN {has_null} THEN list_append(list_sort(list_distinct(l)), NULL) "
            f"ELSE list_sort(list_distinct(l)) END AS u_null FROM {table}"
        ),
    )


def test_unique_keeps_the_null_at_its_first_position():
    got = bt.from_pydict({"l": [[2, None, 1, None, 2]]}).select(
        u=col("l").list.unique(drop_nulls=False)
    )
    assert got.to_pydict() == {"u": [[2, None, 1]]}


# --- sum(empty_as_zero=) ---------------------------------------------------------------------


@pytest.mark.parametrize("table", ["li", "lf"])
def test_sum_empty_as_zero(duck, ints, floats, table):
    source = ints if table == "li" else floats
    out = (
        bt.from_arrow(source)
        .select("id", s=col("l").list.sum(), z=col("l").list.sum(empty_as_zero=True))
        .collect()
    )
    assert out.schema.field("z").type == out.schema.field("s").type
    assert_same(
        out,
        duck.sql(
            "SELECT id, list_sum(l) AS s, "
            f"CASE WHEN l IS NULL THEN NULL ELSE coalesce(list_sum(l), 0) END AS z FROM {table}"
        ),
    )


# --- contains(propagate_nulls=) / position(zero_if_absent=) ---------------------------------


@pytest.mark.parametrize("value", [1, 2, 9, 0])
def test_contains_three_valued_and_position_zero(duck, ints, value):
    out = bt.from_arrow(ints).select(
        "id",
        c=col("l").list.contains(value),
        c3=col("l").list.contains(value, propagate_nulls=True),
        p=col("l").list.position(value),
        p0=col("l").list.position(value, zero_if_absent=True),
    )
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, list_contains(l, {value}) AS c, "
            f"CASE WHEN l IS NULL THEN NULL ELSE {value} IN (SELECT unnest(l)) END AS c3, "
            f"list_position(l, {value}) AS p, "
            f"CASE WHEN l IS NULL THEN NULL ELSE coalesce(list_position(l, {value}), 0) END AS p0 "
            "FROM li"
        ),
    )


# --- concat(propagate_nulls=) / flatten(propagate_nulls=) -----------------------------------


def test_concat_propagating_nulls_matches_the_concat_operator(duck):
    left = [[1], None, [2, None], [], None]
    right = [[2], [3], None, [], None]
    tbl = pa.table(
        {
            "id": list(range(5)),
            "a": pa.array(left, pa.list_(pa.int64())),
            "b": pa.array(right, pa.list_(pa.int64())),
        }
    )
    duck.register("cc", tbl)
    out = bt.from_arrow(tbl).select(
        "id",
        dflt=col("a").list.concat(col("b")),
        strict=col("a").list.concat(col("b"), propagate_nulls=True),
    )
    assert_same(
        out.collect(),
        duck.sql("SELECT id, list_concat(a, b) AS dflt, a || b AS strict FROM cc"),
    )


def test_flatten_propagating_a_null_inner_list(duck):
    rows = [[[1], None, [2]], [[1, None], [3]], None, [], [None], [[]]]
    tbl = pa.table({"id": list(range(6)), "l": pa.array(rows, pa.list_(pa.list_(pa.int64())))})
    duck.register("fl", tbl)
    out = bt.from_arrow(tbl).select(
        "id", f=col("l").list.flatten(), strict=col("l").list.flatten(propagate_nulls=True)
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, flatten(l) AS f, CASE WHEN len(list_filter(l, x -> x IS NULL)) > 0 "
            "THEN NULL ELSE flatten(l) END AS strict FROM fl"
        ),
    )


# --- join(null_replacement=) -----------------------------------------------------------------


def test_join_with_a_null_replacement(duck):
    rows = [["a", None, "b"], [], None, [None], [None, None], ["x"]]
    tbl = pa.table({"id": list(range(6)), "l": pa.array(rows, pa.list_(pa.string()))})
    duck.register("lj", tbl)
    out = bt.from_arrow(tbl).select(
        "id",
        skip=col("l").list.join("-"),
        empty=col("l").list.join("-", null_replacement=""),
        marked=col("l").list.join(",", null_replacement="NA"),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, array_to_string(l, '-') AS skip, "
            "array_to_string(list_transform(l, x -> coalesce(x, '')), '-') AS empty, "
            "array_to_string(list_transform(l, x -> coalesce(x, 'NA')), ',') AS marked FROM lj"
        ),
    )


def test_join_null_replacement_renders_numbers_as_join_does():
    ds = bt.from_pydict({"l": [[1.5, None, -0.0]]})
    got = ds.select(r=col("l").list.join("|", null_replacement="?")).to_pydict()["r"]
    plain = ds.select(r=col("l").list.join("|")).to_pydict()["r"]
    first, last = plain[0].split("|")
    assert got == [f"{first}|?|{last}"]


# --- jaccard(mode="nonzero") ------------------------------------------------------------------


def test_jaccard_nonzero_matches_the_position_set_definition(duck):
    a = [[1.0, 2.0, 3.0], [0.0, 0.0], [1.0, float("nan"), -0.0], [1.0, None, 0.0], None]
    b = [[3.0, 1.0, 5.0], [0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0]]
    tbl = pa.table(
        {
            "id": list(range(5)),
            "a": pa.array(a, pa.list_(pa.float64())),
            "b": pa.array(b, pa.list_(pa.float64())),
        }
    )
    duck.register("jj", tbl)
    got = bt.from_arrow(tbl).select("id", nz=col("a").list.jaccard(col("b"), mode="nonzero"))

    def nonzero(side: str) -> str:
        return f"list_filter(range(1, len({side}) + 1), i -> coalesce({side}[i] <> 0, false))"

    inter = f"len(list_intersect({nonzero('a')}, {nonzero('b')}))"
    union = f"len(list_distinct(list_concat({nonzero('a')}, {nonzero('b')})))"
    assert_same(
        got.collect(),
        duck.sql(
            f"SELECT id, CASE WHEN a IS NULL OR b IS NULL OR {union} = 0 THEN NULL "
            f"ELSE {inter} / {union} END AS nz FROM jj"
        ),
    )


def test_jaccard_rejects_an_unknown_mode():
    with pytest.raises(bt.PlanError, match="nonzero"):
        col("a").list.jaccard(col("b"), mode="sets")


# --- diff keeps an integer list integral (bugfix) --------------------------------------------


def test_integer_diff_is_exact_int64(duck):
    big = 2**53
    rows = [[1, 3, 6], [], None, [None, 1], [big + 1, big + 4, None, 7], [-(2**63) + 1, 5]]
    tbl = pa.table({"id": list(range(6)), "l": pa.array(rows, pa.list_(pa.int64()))})
    duck.register("dl", tbl)
    out = bt.from_arrow(tbl).select("id", d=col("l").list.diff()).collect()
    assert out.schema.field("d").type == pa.list_(pa.int64())
    assert bt.from_arrow(tbl).select(d=col("l").list.diff()).schema.field("d").type == pa.list_(
        pa.int64()
    )
    assert_same(
        out.filter(pa.compute.less(out["id"], 5)),
        duck.sql(
            "SELECT id, list_transform(range(len(l)), i -> CASE WHEN i = 0 THEN NULL "
            "ELSE l[i + 1] - l[i] END) AS d FROM dl WHERE id < 5"
        ),
    )
    # DuckDB's checked BIGINT subtraction overflows here; the engine wraps, as its scalar
    # `-` does, so the one overflowing delta is pinned directly.
    assert out.to_pydict()["d"][5] == [None, (5 - (-(2**63) + 1) + 2**63) % 2**64 - 2**63]


def test_float_diff_stays_float64():
    ds = bt.from_pydict({"l": [[0.5, 2.0]]})
    assert ds.select(d=col("l").list.diff()).collect().schema.field("d").type == pa.list_(
        pa.float64()
    )


# --- the value aggregates' list compositions: top_k and arg_max/arg_min ----------------------


def test_top_k_is_the_largest_values_and_arg_extremes_are_positions(duck):
    xs = [4, None, 9, 9, -3, -3, None, 7]
    gs = ["a", "a", "a", "a", "b", "b", "c", "c"]
    tbl = pa.table({"id": list(range(8)), "g": gs, "x": pa.array(xs, pa.int64())})
    duck.register("tk", tbl)
    got = (
        bt.from_arrow(tbl)
        .sort("id")
        .group_by("g")
        .agg(
            top=col("x").top_k(2),
            freq=col("x").mode_top_k(1),
            imax=col("x").arg_max(),
            imin=col("x").arg_min(),
        )
    )
    assert_same(
        got.collect(),
        duck.sql(
            "SELECT g, max(x, 2) AS top, approx_top_k(x, 1) AS freq, "
            "list_position(list(x ORDER BY id), max(x)) - 1 AS imax, "
            "list_position(list(x ORDER BY id), min(x)) - 1 AS imin FROM tk GROUP BY g"
        ),
    )
