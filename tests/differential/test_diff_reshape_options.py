"""The reshaping keywords vs DuckDB: pivot's several values, aggregates and `fill_value`,
`unpivot(include_nulls=)`, `rollup/cube/grouping_sets(grouping_id=)`, and
`unnest(separator=, max_depth=)`.

Every fixture carries the edge that makes the keyword matter: a pivot cell with no rows
next to one whose rows are all NULL (the two `fill_value` must not merge), a grouping key
holding a genuine NULL (the one `grouping_id` exists to tell from a subtotal), and a NULL
struct beside a populated one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


def _long() -> pa.Table:
    # (r, a): two rows. (r, b): one. (s, a): rows whose values are all NULL. (s, b): no
    # rows at all. (t, *): b only, with a duplicate. A NULL category row must be dropped.
    return pa.table(
        {
            "idx": ["r", "r", "r", "s", "s", "t", "t", "t"],
            "k": ["a", "a", "b", "a", "a", "b", "b", None],
            "v": pa.array([1, 2, 3, None, None, 5, 5, 9], pa.int64()),
            "w": pa.array([1.5, None, 2.5, 3.0, None, 1.0, 1.0, 7.0], pa.float64()),
        }
    )


def _filtered(agg: str, column: str, cat: str, fill: object | None = None) -> str:
    cell = f"{agg}({column}) FILTER (WHERE k = '{cat}')"
    if fill is None:
        return cell
    return f"CASE WHEN count(*) FILTER (WHERE k = '{cat}') = 0 THEN {fill} ELSE {cell} END"


@pytest.mark.parametrize(
    ("values", "aggs", "fill"),
    [
        (["v"], ["sum", "count"], None),
        (["v", "w"], ["sum"], None),
        (["v", "w"], ["min", "mean"], None),
        (["v"], ["sum"], 0),
        (["v", "w"], ["max", "count"], -1),
    ],
)
def test_pivot_several_values_and_aggregates(duck, values, aggs, fill):
    """Each `{value}_{aggregate}_{category}` cell equals DuckDB's filtered aggregate."""
    duck.register("t", _long())
    out = bt.from_arrow(_long()).pivot(
        index="idx",
        on="k",
        values=values if len(values) > 1 else values[0],
        aggregate=aggs if len(aggs) > 1 else aggs[0],
        fill_value=fill,
    )
    items = []
    for v in values:
        for agg in aggs:
            prefix = [v] * (len(values) > 1) + [agg] * (len(aggs) > 1)
            for cat in ("a", "b"):
                name = "_".join([*prefix, cat])
                sql_agg = "avg" if agg == "mean" else agg
                items.append(f'{_filtered(sql_agg, v, cat, fill)} AS "{name}"')
    expected = [
        "_".join([*([v] * (len(values) > 1)), *([a] * (len(aggs) > 1)), c])
        for v in values
        for a in aggs
        for c in ("a", "b")
    ]
    assert out.columns == ["idx", *expected]
    assert_same(out.collect(), duck.sql(f"SELECT idx, {', '.join(items)} FROM t GROUP BY idx"))


def test_fill_value_leaves_an_all_null_cell_null(duck):
    """`fill_value` fills (s, b), which has no rows, and not (s, a), whose rows are NULL."""
    out = bt.from_arrow(_long()).pivot(index="idx", on="k", values="v", fill_value=0)
    got = {r["idx"]: r for r in out.collect().to_pylist()}
    assert got["s"]["a"] is None
    assert got["s"]["b"] == 0


def test_pivot_of_empty_and_one_row_inputs(duck):
    """One row widens to one cell; an empty input with fixed columns has no groups."""
    one = _long().slice(0, 1)
    duck.register("one", one)
    out = bt.from_arrow(one).pivot(
        index="idx", on="k", values="v", columns=["a", "b"], fill_value=0
    )
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT idx, {_filtered('sum', 'v', 'a', 0)} AS a, "
            f"{_filtered('sum', 'v', 'b', 0)} AS b FROM one GROUP BY idx"
        ),
    )
    empty = bt.from_arrow(_long()).filter(col("v") > 100)
    assert empty.pivot(index="idx", on="k", values="v", columns=["a"]).collect().num_rows == 0


def test_pivot_columns_are_type_checked_at_plan_time():
    """A category that cannot match `on`, or two naming one column, fail before execution."""
    ds = bt.from_arrow(_long())
    with pytest.raises(PlanError, match="cannot match column 'k'"):
        ds.pivot(index="idx", on="k", values="v", columns=[1])
    ints = bt.from_pydict({"i": [1], "k": [1], "v": [1]})
    with pytest.raises(PlanError, match="same output column or the same rows"):
        ints.pivot(index="i", on="k", values="v", columns=[1, 1.0])
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.pivot(index="idx", on="k", values="v", columns=["idx"])


def test_discovered_pivot_columns_ascend_and_drop_the_null_category():
    """Discovered categories become columns in ascending order; a NULL category is dropped."""
    out = bt.from_arrow(_long()).pivot(index="idx", on="k", values="v", aggregate="count")
    assert out.columns == ["idx", "a", "b"]


@pytest.mark.parametrize("include_nulls", [True, False])
def test_unpivot_include_nulls_matches_duckdb(duck, include_nulls):
    """`include_nulls=False` is SQL's bare UNPIVOT; the default is INCLUDE NULLS."""
    wide = pa.table(
        {
            "id": pa.array([1, 2, 2, 3], pa.int64()),
            "a": pa.array([1.0, None, None, float("nan")], pa.float64()),
            "b": pa.array([None, 4.0, 4.0, None], pa.float64()),
        }
    )
    duck.register("wide", wide)
    out = bt.from_arrow(wide).unpivot(index="id", include_nulls=include_nulls)
    nulls = "INCLUDE NULLS" if include_nulls else "EXCLUDE NULLS"
    assert out.columns == ["id", "variable", "value"]
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, variable, value FROM wide UNPIVOT {nulls} (value FOR variable IN (a, b))"
        ),
    )


def _keys() -> pa.Table:
    # `b` holds a genuine NULL, which reads like a rolled-up `b` without the id.
    return pa.table(
        {
            "a": ["x", "x", "y", "y", None],
            "b": ["p", None, "p", "q", "q"],
            "v": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )


@pytest.mark.parametrize(
    ("build", "sql_group"),
    [
        (lambda ds: ds.rollup("a", "b", grouping_id="gid"), "ROLLUP (a, b)"),
        (lambda ds: ds.cube("a", "b", grouping_id="gid"), "CUBE (a, b)"),
        (
            lambda ds: ds.grouping_sets(["a", "b"], ["b"], [], grouping_id="gid"),
            "GROUPING SETS ((a, b), (b), ())",
        ),
    ],
)
def test_grouping_id_matches_duckdb(duck, build, sql_group):
    """The id column equals DuckDB's GROUPING_ID(a, b), first key the high bit."""
    duck.register("t", _keys())
    out = build(bt.from_arrow(_keys())).agg(s=col("v").sum(), n=bt.count())
    assert out.columns == ["a", "b", "s", "n", "gid"]
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT a, b, sum(v) AS s, count(*) AS n, GROUPING_ID(a, b) AS gid "
            f"FROM t GROUP BY {sql_group}"
        ),
    )


def test_grouping_id_over_an_empty_input(duck):
    """The grand total of an empty input still carries its id (all keys rolled up)."""
    empty = _keys().slice(0, 0)
    duck.register("e", empty)
    out = bt.from_arrow(empty).rollup("a", "b", grouping_id="gid").agg(s=col("v").sum())
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT a, b, sum(v) AS s, GROUPING_ID(a, b) AS gid FROM e GROUP BY ROLLUP (a, b)"
        ),
    )


def test_grouping_id_name_collision_is_refused():
    with pytest.raises(PlanError, match="collides"):
        bt.from_arrow(_keys()).rollup("a", grouping_id="a").agg(s=col("v").sum())
    with pytest.raises(PlanError, match="collides"):
        bt.from_arrow(_keys()).cube("a", grouping_id="s").agg(s=col("v").sum())


def _structs() -> pa.Table:
    inner = pa.struct([("c", pa.int64()), ("d", pa.string())])
    s_type = pa.struct([("a", pa.int64()), ("b", inner)])
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "s": pa.array(
                [{"a": 1, "b": {"c": 10, "d": "x"}}, None, {"a": 3, "b": None}], type=s_type
            ),
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "select"),
    [
        ({}, "id, s.a AS a, s.b AS b"),
        ({"separator": "."}, 'id, s.a AS "s.a", s.b AS "s.b"'),
        ({"separator": "_", "max_depth": 2}, "id, s.a AS s_a, s.b.c AS s_b_c, s.b.d AS s_b_d"),
        ({"max_depth": 2}, "id, s.a AS a, s.b.c AS c, s.b.d AS d"),
        (
            {"separator": ".", "max_depth": 5},
            'id, s.a AS "s.a", s.b.c AS "s.b.c", s.b.d AS "s.b.d"',
        ),
    ],
)
def test_unnest_separator_and_depth_match_duckdb(duck, kwargs, select):
    """Nested fields expand to the requested depth and are named by their path."""
    duck.register("t", _structs())
    out = bt.from_arrow(_structs()).unnest("s", **kwargs)
    exp = duck.sql(f"SELECT {select} FROM t")
    assert out.columns == exp.columns
    assert_same(out.collect(), exp)


def test_unnest_collision_and_bad_options():
    """A collision names the fix; `max_depth` and `separator` are validated."""
    t = _structs().append_column("c", pa.array([0, 0, 0], pa.int64()))
    with pytest.raises(PlanError, match=r"collide.*separator"):
        bt.from_arrow(t).unnest("s", max_depth=2)
    with pytest.raises(PlanError, match="max_depth"):
        bt.from_arrow(t).unnest("s", max_depth=0)
    with pytest.raises(PlanError, match="separator"):
        bt.from_arrow(t).unnest("s", separator="")
