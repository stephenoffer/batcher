"""`join`'s opt-in keywords vs DuckDB, across execution paths and edge-case inputs.

Each keyword is a control-plane rewrite onto the one equi-join operator, so each is checked
the way `test_diff_operator_matrix.py` checks an operator: every path (`collect`, spill,
spill into a forced bucket count, `iter_batches`) against DuckDB, on inputs carrying nulls,
an empty side, a single row, duplicate keys and NaN / signed-zero float keys.

* ``nulls_equal=True`` is SQL ``IS NOT DISTINCT FROM``, the oracle for it, both for the
  DataFrame keyword and for the SQL form, which now plans as a hash join instead of a
  cross join filtered on the null-safe comparison.
* ``indicator=`` is checked against DuckDB marker columns, on rows whose payload is null.
* ``coalesce=False`` is checked against selecting ``l.k`` and ``r.k`` separately.
* expression keys are checked against ``ON lower(l.s) = lower(r.s)``.

Every result here is a join with no outer ORDER BY, so the order-independent `assert_same`
is the right comparison throughout.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, duck_materialize

pytestmark = pytest.mark.differential

#: Integer keys: nulls on both sides, duplicates on both sides, and a zero that a null
#: filled with the type's zero must not be confused with.
INTS = (
    pa.table(
        {
            "k": pa.array([1, 2, None, 2, 0, None], pa.int64()),
            "lv": pa.array([10, 20, 30, None, 50, 60], pa.int64()),
        }
    ),
    pa.table(
        {
            "k": pa.array([1, None, 5, 0, 2, None], pa.int64()),
            "rv": pa.array([None, 200, 300, 400, 500, 600], pa.int64()),
        }
    ),
)
#: Float keys: NaN on both sides, -0.0 against 0.0, and nulls.
FLOATS = (
    pa.table(
        {
            "k": pa.array([1.0, float("nan"), None, -0.0, 2.5], pa.float64()),
            "lv": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    ),
    pa.table(
        {
            "k": pa.array([float("nan"), 0.0, None, 1.0, 7.0], pa.float64()),
            "rv": pa.array([10, 20, 30, 40, 50], pa.int64()),
        }
    ),
)
#: String keys with nulls and an empty string, which is the fill value for a null.
STRINGS = (
    pa.table({"k": pa.array(["a", "", None, "b"]), "lv": pa.array([1, 2, 3, 4], pa.int64())}),
    pa.table({"k": pa.array(["", None, "b", "c"]), "rv": pa.array([5, 6, 7, 8], pa.int64())}),
)
#: A single row on each side, both null keys.
ONE_ROW = (
    pa.table({"k": pa.array([None], pa.int64()), "lv": pa.array([1], pa.int64())}),
    pa.table({"k": pa.array([None], pa.int64()), "rv": pa.array([2], pa.int64())}),
)
#: An empty right side.
EMPTY = (INTS[0], INTS[1].slice(0, 0))
#: Past several morsels, with nulls, so the spilled and streamed paths see more than one batch.
_N = 40_000
MULTIBATCH = (
    pa.table(
        {
            "k": pa.array([None if i % 97 == 0 else i % 9_000 for i in range(_N)], pa.int64()),
            "lv": pa.array(range(_N), pa.int64()),
        }
    ),
    pa.table(
        {
            "k": pa.array([None if i % 89 == 0 else i for i in range(9_000)], pa.int64()),
            "rv": pa.array(range(9_000), pa.int64()),
        }
    ),
)

SHAPES = {
    "ints": INTS,
    "floats": FLOATS,
    "strings": STRINGS,
    "one_row": ONE_ROW,
    "empty": EMPTY,
    "multibatch": MULTIBATCH,
}


def _stream(ds: bt.Dataset) -> pa.Table:
    """`iter_batches()` collected back into one table."""
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches)


PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True),
    "spill_partitioned": lambda ds: ds.collect(spill=True, num_partitions=3),
    "iter_batches": _stream,
}

#: The key column each join type reports when keys are coalesced.
_KEY = {
    "inner": "l.k",
    "left": "l.k",
    "right": "r.k",
    "full": "coalesce(l.k, r.k)",
}


def _register(duck, shape: str) -> tuple[pa.Table, pa.Table]:
    """Copy a shape into DuckDB's own storage, so a NaN key compares by DuckDB's rules.

    A registered Arrow table has the join filter pushed into its scan, where NaN compares
    unequal to itself; DuckDB's executor (and Batcher) match NaN with NaN.
    """
    left, right = SHAPES[shape]
    duck_materialize(duck, "l", left)
    duck_materialize(duck, "r", right)
    return left, right


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_nulls_equal_matches_is_not_distinct_from(duck, how, shape, path):
    left, right = _register(duck, shape)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how, nulls_equal=True)
    sql = (
        f"SELECT {_KEY[how]} AS k, l.lv, r.rv FROM l {how.upper()} JOIN r "
        "ON l.k IS NOT DISTINCT FROM r.k"
    )
    assert_same(PATHS[path](ds.select("k", "lv", "rv")), duck.sql(sql))


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_nulls_equal_semi_anti(duck, how, shape):
    left, right = _register(duck, shape)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how, nulls_equal=True)
    sql = f"SELECT l.k, l.lv FROM l {how.upper()} JOIN r ON l.k IS NOT DISTINCT FROM r.k"
    assert_same(out.collect(), duck.sql(sql))


def test_nulls_equal_false_is_the_default_sql_equality(duck):
    """The control: without the keyword a null key still matches nothing."""
    left, right = _register(duck, "ints")
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="full")
    sql = "SELECT coalesce(l.k, r.k) AS k, l.lv, r.rv FROM l FULL JOIN r ON l.k = r.k"
    assert_same(out.select("k", "lv", "rv").collect(), duck.sql(sql))


def test_nulls_equal_per_key(duck):
    left = pa.table({"a": [1, 1, None], "b": [None, 2, None], "lv": [1, 2, 3]})
    right = pa.table({"a": [1, None, 1], "b": [None, None, 2], "rv": [4, 5, 6]})
    duck.register("l", left)
    duck.register("r", right)
    out = bt.from_arrow(left).join(
        bt.from_arrow(right), on=["a", "b"], how="left", nulls_equal=[False, True]
    )
    sql = (
        "SELECT l.a, l.b, l.lv, r.rv FROM l LEFT JOIN r "
        "ON l.a = r.a AND l.b IS NOT DISTINCT FROM r.b"
    )
    assert_same(out.select("a", "b", "lv", "rv").collect(), duck.sql(sql))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_indicator(duck, how, shape, path):
    left, right = _register(duck, shape)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how, indicator="src")
    sql = (
        f"SELECT {_KEY[how]} AS k, l.lv, r.rv, CASE WHEN l.m AND r.m THEN 'both' "
        "WHEN l.m THEN 'left_only' ELSE 'right_only' END AS src "
        f"FROM (SELECT *, true AS m FROM l) l {how.upper()} JOIN "
        "(SELECT *, true AS m FROM r) r ON l.k = r.k"
    )
    assert_same(PATHS[path](ds.select("k", "lv", "rv", "src")), duck.sql(sql))


def test_indicator_with_all_null_payload(duck):
    """A matched row whose every payload column is null is still `both`."""
    left = pa.table({"k": [1, 2], "x": pa.array([None, None], pa.int64())})
    right = pa.table({"k": [1, 3], "y": pa.array([None, None], pa.int64())})
    duck.register("l", left)
    duck.register("r", right)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="full", indicator="src")
    sql = (
        "SELECT coalesce(l.k, r.k) AS k, l.x, r.y, CASE WHEN l.m AND r.m THEN 'both' "
        "WHEN l.m THEN 'left_only' ELSE 'right_only' END AS src "
        "FROM (SELECT *, true AS m FROM l) l FULL JOIN (SELECT *, true AS m FROM r) r "
        "ON l.k = r.k"
    )
    assert_same(out.collect(), duck.sql(sql))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_coalesce_false_keeps_both_keys(duck, how, shape, path):
    left, right = _register(duck, shape)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how=how, coalesce=False)
    assert ds.columns == ["k", "lv", "k_right", "rv"]
    sql = f"SELECT l.k, l.lv, r.k AS k_right, r.rv FROM l {how.upper()} JOIN r ON l.k = r.k"
    assert_same(PATHS[path](ds), duck.sql(sql))


def test_coalesce_false_with_distinct_key_names(duck):
    left = pa.table({"k": [1, 2, None], "lv": [1, 2, 3]})
    right = pa.table({"rk": [2, 3, None], "rv": [4, 5, 6]})
    duck.register("l", left)
    duck.register("r", right)
    ds = bt.from_arrow(left).join(
        bt.from_arrow(right), left_on="k", right_on="rk", how="full", coalesce=False
    )
    assert ds.columns == ["k", "lv", "rk", "rv"]
    sql = "SELECT l.k, l.lv, r.rk, r.rv FROM l FULL JOIN r ON l.k = r.rk"
    assert_same(ds.collect(), duck.sql(sql))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_expression_keys(duck, how, path):
    left = pa.table({"s": ["Ann", "BOB", None, "cy", "bob"], "lv": [1, 2, 3, 4, 5]})
    right = pa.table({"s": ["ann", "Bob", "dee", None], "rv": [6, 7, 8, 9]})
    duck.register("l", left)
    duck.register("r", right)
    key = bt.col("s").str.lower()
    ds = bt.from_arrow(left).join(bt.from_arrow(right), left_on=key, right_on=key, how=how)
    # A computed key adds no column: both sides' `s` survive as payload.
    assert ds.columns == ["s", "lv", "s_right", "rv"]
    sql = (
        f"SELECT l.s, l.lv, r.s AS s_right, r.rv FROM l {how.upper()} JOIN r "
        "ON lower(l.s) = lower(r.s)"
    )
    assert_same(PATHS[path](ds), duck.sql(sql))


def test_expression_key_mixed_with_a_name(duck):
    left = pa.table({"g": [1, 1, 2], "s": ["A", "b", "C"], "lv": [1, 2, 3]})
    right = pa.table({"g": [1, 2, 2], "s": ["a", "c", "x"], "rv": [4, 5, 6]})
    duck.register("l", left)
    duck.register("r", right)
    key = bt.col("s").str.lower()
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on=["g", key], how="left")
    sql = (
        "SELECT l.g, l.s, l.lv, r.s AS s_right, r.rv FROM l LEFT JOIN r "
        "ON l.g = r.g AND lower(l.s) = lower(r.s)"
    )
    assert_same(ds.collect(), duck.sql(sql))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_validate_passing_join_matches(duck, shape):
    """Every right side here repeats only null keys, so ``m:1`` passes and changes nothing."""
    left, right = _register(duck, shape)
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="left", validate="m:1")
    sql = "SELECT l.k, l.lv, r.rv FROM l LEFT JOIN r ON l.k = r.k"
    assert_same(ds.select("k", "lv", "rv").collect(), duck.sql(sql))


@pytest.mark.parametrize("how", ["inner", "left", "full", "semi", "anti"])
def test_sql_is_not_distinct_from(duck, how):
    left, right = _register(duck, "ints")
    keys = "l.k, l.lv" if how in {"semi", "anti"} else "l.k, l.lv, r.k AS rk, r.rv"
    sql = f"SELECT {keys} FROM l {how.upper()} JOIN r ON l.k IS NOT DISTINCT FROM r.k"
    assert_same(bt.sql(sql, l=left, r=right).collect(), duck.sql(sql))


def test_sql_mixed_equality_and_is_not_distinct_from(duck):
    left = pa.table({"a": [1, 1, None, 2], "b": [None, 2, None, 3], "lv": [1, 2, 3, 4]})
    right = pa.table({"a": [1, None, 1, 2], "b": [None, None, 2, 3], "rv": [4, 5, 6, 7]})
    duck.register("l", left)
    duck.register("r", right)
    sql = (
        "SELECT l.a, l.b, l.lv, r.rv FROM l FULL JOIN r "
        "ON l.a = r.a AND l.b IS NOT DISTINCT FROM r.b"
    )
    assert_same(bt.sql(sql, l=left, r=right).collect(), duck.sql(sql))


def test_sql_is_not_distinct_from_plans_a_keyed_hash_join():
    """The null-safe ON plans on real keys; an expression operand is the positive control.

    The cross join `explain()` renders as ``__cross_key`` is what an ON predicate the
    translator cannot key on still becomes, so its presence there proves the token is
    reachable and its absence for the null-safe column comparison means something.
    """
    left, right = INTS
    keyed = bt.sql(
        "SELECT * FROM l JOIN r ON l.k IS NOT DISTINCT FROM r.k", l=left, r=right
    ).explain()
    computed = bt.sql(
        "SELECT * FROM l JOIN r ON l.k + 0 IS NOT DISTINCT FROM r.k", l=left, r=right
    ).explain()
    assert "__cross_key" in computed
    assert "__cross_key" not in keyed
    assert "hash_join" in keyed
