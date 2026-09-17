"""`Dataset.join_where`, `Dataset.update` and `GroupBy.having` against DuckDB, on every path.

None of the three adds an operator. `join_where` is the cartesian join plus filter that Kyber
turns into a `RangeJoin` or a hash join, `update` is a join plus a coalesce, and `having` is an
aggregate with hidden predicate columns plus a filter. What needs pinning is therefore not a
kernel but the composition: that the rewrite Kyber picks still returns DuckDB's rows, and that
it does so on each scheduling of the same semantics.

Every case runs ``collect()``, a spilled ``collect(num_partitions=4)`` and ``iter_batches()``
over inputs carrying nulls in keys and values, an empty side, a one-row side, duplicate keys,
a descending arrival order, and a ``multibatch`` shape past two morsels so the paths are
genuinely different. The results are multisets, so `assert_same` is the right comparison: none
of these verbs promises an order.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

#: The three schedulings of one semantics (invariant #7).
PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True, num_partitions=4),
    "iter_batches": lambda ds: _stream(ds),
}


def _stream(ds: bt.Dataset) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


def _events(n: int) -> pa.Table:
    """`n` events with a nullable time, a nullable float payload and duplicate times."""
    t = [None if i % 11 == 0 else (i * 7) % 50 for i in range(n)]
    v = [None if i % 13 == 0 else float((i * 3) % 17) - 4.0 for i in range(n)]
    return pa.table({"t": pa.array(t, pa.int64()), "v": pa.array(v, pa.float64())})


def _spans() -> pa.Table:
    """Overlapping intervals, a null bound, a duplicate interval and an empty one."""
    return pa.table(
        {
            "lo": pa.array([0, 10, 10, None, 30, 45, 20], pa.int64()),
            "hi": pa.array([15, 25, 25, 40, 30, 60, None], pa.int64()),
            "v": pa.array([1.0, None, 5.0, 2.0, 0.0, -3.0, 9.0], pa.float64()),
        }
    )


EVENT_SHAPES = {
    "base": _events(60),
    "empty": _events(0),
    "single": _events(2).slice(1, 1),
    "descending": _events(60).sort_by([("t", "descending")]),
    "multibatch": _events(40_000),
}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(EVENT_SHAPES))
def test_join_where_interval_matches_duckdb(duck, shape, path):
    """Two inequalities (the IEJoin shape) plus a residual predicate on colliding names."""
    events, spans = EVENT_SHAPES[shape], _spans()
    duck.register("e", events)
    duck.register("s", spans)
    ds = bt.from_arrow(events).join_where(
        bt.from_arrow(spans),
        bt.col("t") >= bt.col("lo"),
        bt.col("t") < bt.col("hi"),
        bt.col("v") < bt.col("v_right"),
    )
    expected = duck.sql(
        "SELECT e.t, e.v, s.lo, s.hi, s.v AS v_right FROM e, s "
        "WHERE e.t >= s.lo AND e.t < s.hi AND e.v < s.v"
    )
    assert_same(PATHS[path](ds), expected)


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(EVENT_SHAPES))
def test_join_where_single_inequality_and_equality_match_duckdb(duck, shape, path):
    """One inequality (a sorted-suffix range join) and an equality (a hash join)."""
    events, spans = EVENT_SHAPES[shape], _spans()
    duck.register("e", events)
    duck.register("s", spans)
    right = bt.from_arrow(spans)
    below = bt.from_arrow(events).join_where(right, bt.col("t") > bt.col("hi"))
    assert_same(
        PATHS[path](below),
        duck.sql("SELECT e.*, s.lo, s.hi, s.v AS v_right FROM e, s WHERE e.t > s.hi"),
    )
    equal = bt.from_arrow(events).join_where(right, [bt.col("t") == bt.col("lo")])
    assert_same(
        PATHS[path](equal),
        duck.sql("SELECT e.*, s.lo, s.hi, s.v AS v_right FROM e, s WHERE e.t = s.lo"),
    )


def _target(n: int) -> pa.Table:
    """Rows to update: duplicate keys, a null key, nulls in both value columns."""
    ids = [None if i % 17 == 5 else i % 23 for i in range(n)]
    x = [None if i % 9 == 0 else i * 10 for i in range(n)]
    y = [None if i % 7 == 3 else f"y{i}" for i in range(n)]
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "x": pa.array(x, pa.int64()),
            "y": pa.array(y, pa.string()),
        }
    )


def _source() -> pa.Table:
    """New values: a null value (kept out unless `include_nulls`), a null key, an unmatched key,
    a duplicated key, and a column only this side has."""
    return pa.table(
        {
            "id": pa.array([1, 2, 2, 5, None, 99, 7], pa.int64()),
            "x": pa.array([100, None, 200, 500, 1, 9900, None], pa.int64()),
            "y": pa.array(["A", "B", None, None, "N", "Z", "G"]),
            "w": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
        }
    )


TARGET_SHAPES = {
    "base": _target(60),
    "empty": _target(0),
    "single": _target(3).slice(1, 1),
    "descending": _target(60).sort_by([("id", "descending")]),
    "multibatch": _target(40_000),
}

_JOIN_SQL = {"left": "LEFT JOIN", "inner": "JOIN", "full": "FULL JOIN"}


def _update_sql(how: str, include_nulls: bool) -> str:
    key = "coalesce(t.id, s.id)" if how == "full" else "t.id"
    if include_nulls:
        x = "CASE WHEN s.matched THEN s.x ELSE t.x END"
        y = "CASE WHEN s.matched THEN s.y ELSE t.y END"
    else:
        x, y = "coalesce(s.x, t.x)", "coalesce(s.y, t.y)"
    return (
        f"SELECT {key} AS id, {x} AS x, {y} AS y FROM t "
        f"{_JOIN_SQL[how]} (SELECT *, true AS matched FROM s) s ON t.id = s.id"
    )


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("include_nulls", [False, True])
@pytest.mark.parametrize("how", sorted(_JOIN_SQL))
@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_update_matches_duckdb(duck, shape, how, include_nulls, path):
    """Every `how` and both null policies, as the join-plus-coalesce DuckDB spells it."""
    target, source = TARGET_SHAPES[shape], _source()
    duck.register("t", target)
    duck.register("s", source)
    ds = bt.from_arrow(target).update(
        bt.from_arrow(source), on="id", how=how, include_nulls=include_nulls
    )
    out = PATHS[path](ds)
    assert out.column_names == ["id", "x", "y"]
    assert_same(out, duck.sql(_update_sql(how, include_nulls)))


def test_update_with_differing_key_names_matches_on_key(duck):
    """`left_on`/`right_on` name the key on each side; the result keeps the left name."""
    target = _target(40)
    source = _source().rename_columns(["key", "x", "y", "w"])
    duck.register("t", target)
    duck.register("s", source)
    ds = bt.from_arrow(target).update(bt.from_arrow(source), left_on="id", right_on="key")
    expected = duck.sql(
        "SELECT t.id, coalesce(s.x, t.x) AS x, coalesce(s.y, t.y) AS y "
        "FROM t LEFT JOIN s ON t.id = s.key"
    )
    assert_same(ds.collect(), expected)


def _grouped(n: int) -> pa.Table:
    """Groups of different sizes, a null group key, an all-null value group, duplicates."""
    g = [None if i % 10 == 0 else "abcde"[i % 5] for i in range(n)]
    v = [None if "abcde"[i % 5] == "e" else (i * 7) % 13 for i in range(n)]
    return pa.table({"g": pa.array(g, pa.string()), "v": pa.array(v, pa.int64())})


GROUP_SHAPES = {
    "base": _grouped(47),
    "empty": _grouped(0),
    "single": _grouped(2).slice(1, 1),
    "descending": _grouped(47).sort_by([("v", "descending")]),
    "multibatch": _grouped(40_000),
}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(GROUP_SHAPES))
def test_having_matches_duckdb(duck, shape, path):
    """Two stacked predicates, one null for the all-null group, over a composite output."""
    table = GROUP_SHAPES[shape]
    duck.register("t", table)
    grouped = (
        bt.from_arrow(table)
        .group_by("g")
        .having(bt.count() > 1)
        .having(bt.col("v").sum() >= 10)
        .agg(s=bt.col("v").sum(), spread=bt.col("v").max() - bt.col("v").min())
    )
    expected = duck.sql(
        "SELECT g, sum(v) AS s, max(v) - min(v) AS spread FROM t GROUP BY g "
        "HAVING count(*) > 1 AND sum(v) >= 10"
    )
    assert_same(PATHS[path](grouped), expected)


@pytest.mark.parametrize("shape", sorted(GROUP_SHAPES))
def test_having_filters_the_shortcut_reductions(duck, shape):
    """`len`/`sum` finish through a different lowering than `agg`, and `having` covers both."""
    table = GROUP_SHAPES[shape]
    duck.register("t", table)
    base = bt.from_arrow(table).group_by("g").having(bt.col("v").max() > 5)
    assert_same(
        base.len().collect(),
        duck.sql('SELECT g, count(*) AS "len" FROM t GROUP BY g HAVING max(v) > 5'),
    )
    assert_same(
        base.sum("v").collect(),
        duck.sql("SELECT g, sum(v) AS v FROM t GROUP BY g HAVING max(v) > 5"),
    )


def test_having_keeps_first_appearance_order_under_maintain_order():
    """`maintain_order` sorts the groups; the filter above must not disturb that order."""
    table = _grouped(47)
    kept = (
        bt.from_arrow(table)
        .group_by("g", maintain_order=True)
        .having(bt.count() > 9)
        .len()
        .collect()
    )
    unfiltered = bt.from_arrow(table).group_by("g", maintain_order=True).len().collect()
    expected = unfiltered.filter(pa.compute.greater(unfiltered.column("len"), 9))
    assert_tables_equal(kept, expected, ordered=True)
