"""Range (inequality) joins vs DuckDB, on the plan shape that actually runs them.

``test_diff_theta_join.py`` pins that a non-equi join is *correct*. This file pins the
`RangeJoin` rewrite that makes it *fast*: the optimizer moves one or two inequality
conjuncts into the join, which the engine then executes with a sorted-suffix scan or
IEJoin instead of materializing the cartesian product.

Two things are checked for every case, because either alone would be misleading. The
result must match DuckDB — and the plan must actually contain a ``range_join``, since a
rewrite that silently declines would pass every result assertion while changing nothing.

The edge cases are where a range algorithm can quietly differ from a filter over a cross
product: NULL and NaN keys (both never match, for different reasons), ``-0.0`` against
``0.0`` (equal under IEEE, distinct to a row encoder), ties on either or both axes, and
an empty side.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.kyber.optimizer import optimize_logical


def _ir_ops(ir: object) -> set[str]:
    """Every ``op`` tag appearing in a lowered plan."""
    found: set[str] = set()
    if isinstance(ir, dict):
        op = ir.get("op")
        if isinstance(op, str):
            found.add(op)
        for v in ir.values():
            found |= _ir_ops(v)
    elif isinstance(ir, list):
        for v in ir:
            found |= _ir_ops(v)
    return found


def _assert_range_join(ds: bt.Dataset) -> None:
    """The optimized plan must carry a `range_join`, not a cartesian `hash_join`."""
    ops = _ir_ops(optimize_logical(ds._plan).to_ir())
    assert "range_join" in ops, f"expected a range_join in the plan, got {sorted(ops)}"


def _check(duck, query: str, **tables: pa.Table) -> None:
    ds = bt.sql(query, **tables)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))


@pytest.fixture
def points_and_intervals(duck):
    """Points against half-open intervals — the shape a range join exists for."""
    pt = pa.table({"x": [0, 5, 10, 15, 20, 25], "pid": ["a", "b", "c", "d", "e", "f"]})
    iv = pa.table({"lo": [0, 8, 12, 30], "hi": [10, 16, 13, 40], "iid": ["p", "q", "r", "s"]})
    duck.register("pt", pt)
    duck.register("iv", iv)
    return pt, iv


@pytest.mark.differential
@pytest.mark.parametrize(
    "cond",
    [
        "pt.x >= iv.lo AND pt.x < iv.hi",
        "pt.x > iv.lo AND pt.x <= iv.hi",
        "pt.x > iv.lo AND pt.x < iv.hi",
        "pt.x >= iv.lo AND pt.x <= iv.hi",
        "pt.x < iv.lo AND pt.x > iv.hi",
    ],
)
def test_interval_containment_every_operator_pair(duck, points_and_intervals, cond):
    """All sixteen strict/non-strict combinations reduce to these five shapes."""
    pt, iv = points_and_intervals
    _check(duck, f"SELECT pid, iid FROM pt, iv WHERE {cond}", pt=pt, iv=iv)


@pytest.mark.differential
@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_single_inequality(duck, op):
    """One inequality takes the sorted-suffix path, not IEJoin."""
    a = pa.table({"x": [1, 2, 3, 4], "lab": ["a", "b", "c", "d"]})
    b = pa.table({"y": [2, 3, 10], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, f"SELECT lab, tag FROM a, b WHERE a.x {op} b.y", a=a, b=b)


@pytest.mark.differential
@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_reversed_operand_order(duck, op):
    """`b.y > a.x` is `a.x < b.y`; the rewrite must flip the operator, not the sides."""
    a = pa.table({"x": [1, 2, 3, 4], "lab": ["a", "b", "c", "d"]})
    b = pa.table({"y": [2, 3, 10], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, f"SELECT lab, tag FROM a, b WHERE b.y {op} a.x", a=a, b=b)


@pytest.mark.differential
def test_nulls_never_match(duck):
    """`NULL < y` is UNKNOWN, so the row is dropped — on both sides, on both axes."""
    a = pa.table({"x": [1, None, 3], "z": [10, 20, None], "lab": ["a", "b", "c"]})
    b = pa.table({"lo": [0, None, 2], "hi": [5, 9, None], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x >= b.lo AND a.z < b.hi", a=a, b=b)


@pytest.mark.differential
def test_nan_never_matches_and_negative_zero_equals_zero(duck):
    """IEEE: every comparison with NaN is false; `-0.0` and `0.0` compare equal.

    A row encoder disagrees with both — it orders NaN as the largest value and gives
    `-0.0` different bytes from `0.0` — so this is where a range join built on one would
    diverge from the filter it replaces.
    """
    a = pa.table({"x": [math.nan, -0.0, 1.0, 2.0], "lab": ["a", "b", "c", "d"]})
    b = pa.table({"y": [0.0, math.nan, 1.5], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x <= b.y", a=a, b=b)


@pytest.mark.differential
def test_ties_on_both_axes(duck):
    """Equal keys are where strict and non-strict operators part company."""
    a = pa.table({"x": [1, 1, 1, 2], "z": [5, 5, 6, 5], "lab": ["a", "b", "c", "d"]})
    b = pa.table({"lo": [1, 1, 2], "hi": [5, 6, 5], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x >= b.lo AND a.z <= b.hi", a=a, b=b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x > b.lo AND a.z < b.hi", a=a, b=b)


@pytest.mark.differential
def test_empty_side_yields_no_rows_but_the_right_columns(duck):
    a = pa.table({"x": pa.array([], type=pa.int64()), "lab": pa.array([], type=pa.string())})
    b = pa.table({"y": [1, 2], "tag": ["p", "q"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x < b.y", a=a, b=b)


@pytest.mark.differential
def test_a_predicate_matching_nothing(duck):
    a = pa.table({"x": [100, 200], "lab": ["a", "b"]})
    b = pa.table({"y": [1, 2], "tag": ["p", "q"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.x < b.y", a=a, b=b)


@pytest.mark.differential
def test_extra_conjuncts_stay_in_the_filter(duck):
    """A third inequality and a single-side predicate are post-checks, not join keys."""
    a = pa.table({"x": [1, 2, 3, 4, 5], "z": [9, 8, 7, 6, 5], "lab": ["a", "b", "c", "d", "e"]})
    b = pa.table({"lo": [0, 2, 4], "hi": [9, 8, 7], "m": [3, 3, 3], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(
        duck,
        "SELECT lab, tag FROM a, b WHERE a.x > b.lo AND a.z < b.hi AND a.x < b.m AND a.x > 1",
        a=a,
        b=b,
    )


@pytest.mark.differential
def test_string_and_date_keys(duck):
    """Any type with a total order works; the row encoder supplies it."""
    a = pa.table({"s": ["apple", "pear", "zebra"], "lab": ["a", "b", "c"]})
    b = pa.table({"t": ["banana", "quince"], "tag": ["p", "q"]})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, "SELECT lab, tag FROM a, b WHERE a.s < b.t", a=a, b=b)


@pytest.mark.differential
def test_an_equality_conjunct_keeps_the_hash_join(duck):
    """An equality is worth more than an inequality — the rewrite must stand down.

    Absorbed into the join keys, `a.k = b.k` makes the whole thing a hash join, which
    beats any range algorithm. This is the one case where *not* firing is the win.
    """
    a = pa.table({"k": [1, 1, 2], "x": [1, 5, 9], "lab": ["a", "b", "c"]})
    b = pa.table({"k": [1, 2, 2], "y": [4, 8, 12], "tag": ["p", "q", "r"]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT lab, tag FROM a, b WHERE a.k = b.k AND a.x < b.y"
    ds = bt.sql(query, a=a, b=b)
    ops = _ir_ops(optimize_logical(ds._plan).to_ir())
    assert "range_join" not in ops, "an equi-conjunct must win the join keys"
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
def test_mismatched_key_types_decline_the_rewrite(duck):
    """`Int64` against `Float64` cannot share a row encoding, so it stays a filter."""
    a = pa.table({"x": pa.array([1, 2, 3], type=pa.int64()), "lab": ["a", "b", "c"]})
    b = pa.table({"y": pa.array([1.5, 2.5], type=pa.float64()), "tag": ["p", "q"]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT lab, tag FROM a, b WHERE a.x < b.y"
    ds = bt.sql(query, a=a, b=b)
    ops = _ir_ops(optimize_logical(ds._plan).to_ir())
    assert "range_join" not in ops
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("mode", ["collect", "iter_batches"])
def test_execution_modes_agree(duck, points_and_intervals, mode):
    """A breaker must produce the same relation streamed as materialized."""
    pt, iv = points_and_intervals
    query = "SELECT pid, iid FROM pt, iv WHERE pt.x >= iv.lo AND pt.x < iv.hi"
    ds = bt.sql(query, pt=pt, iv=iv)
    _assert_range_join(ds)
    if mode == "collect":
        got = ds.collect()
    else:
        batches = list(ds.iter_batches())
        got = pa.Table.from_batches(batches, schema=batches[0].schema) if batches else ds.collect()
    assert_same(got, duck.sql(query))


@pytest.mark.differential
def test_many_rows_matches_duckdb(duck):
    """A size where the cartesian plan would materialize 4M pairs for ~7K results."""
    n = 2_000
    pt = pa.table({"x": [(i * 7919) % 20_000 for i in range(n)], "pid": list(range(n))})
    iv = pa.table(
        {
            "lo": [(i * 13) % 20_000 for i in range(n)],
            "hi": [(i * 13) % 20_000 + 40 for i in range(n)],
            "iid": list(range(n)),
        }
    )
    duck.register("pt", pt)
    duck.register("iv", iv)
    _check(
        duck,
        "SELECT count(*) c, sum(pid) s FROM pt, iv WHERE pt.x >= iv.lo AND pt.x < iv.hi",
        pt=pt,
        iv=iv,
    )


@pytest.mark.differential
def test_distributed_equals_single_node(duck):
    """The single-node == distributed invariant, for the new operator.

    A range join is decomposable — a left row's matches depend on the whole right side
    and on nothing else about the left — so no amount of partitioning may change the
    answer. Today the distributed planner has no range-join staging and falls back to
    executing it whole, which satisfies the invariant; this test is what would fail if a
    future staging rewrite got the decomposition wrong.
    """
    n = 200
    pt = pa.table({"x": [(i * 37) % 500 for i in range(n)], "pid": list(range(n))})
    iv = pa.table(
        {
            "lo": [(i * 11) % 500 for i in range(50)],
            "hi": [(i * 11) % 500 + 25 for i in range(50)],
            "iid": list(range(50)),
        }
    )
    duck.register("pt", pt)
    duck.register("iv", iv)
    query = "SELECT pid, iid FROM pt, iv WHERE pt.x >= iv.lo AND pt.x < iv.hi"
    ds = bt.sql(query, pt=pt, iv=iv)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))
    assert_same(
        bt.sql(query, pt=pt, iv=iv).collect(distributed=True, num_workers=3), duck.sql(query)
    )


@pytest.mark.differential
def test_temporal_overlap_of_media_segments(duck):
    """Two spans overlap iff `a.start < b.end AND a.end > b.start`.

    This is the join at the heart of aligning ASR spans to detected video scenes, of
    windowing event logs against sessions, and of any "which of these intervals touch
    which of those" question. It is the same operator as a numeric band join, which is
    why it is worth pinning here: the shape a multimodal pipeline reaches for is one an
    analytics engine already had to get right.
    """
    seg = pa.table(
        {"s_start": [0, 100, 250, 900], "s_end": [120, 200, 400, 950], "sid": list("abcd")}
    )
    tr = pa.table({"t_start": [50, 210, 500], "t_end": [110, 300, 600], "tid": list("xyz")})
    duck.register("seg", seg)
    duck.register("tr", tr)
    _check(
        duck,
        "SELECT sid, tid FROM seg, tr WHERE seg.s_start < tr.t_end AND seg.s_end > tr.t_start",
        seg=seg,
        tr=tr,
    )


@pytest.mark.differential
def test_bounding_box_overlap(duck):
    """Two axis-aligned boxes intersect iff they overlap on *both* axes — four inequalities.

    Object-detection dedup and IoU blocking are this shape. Two conditions go into the
    join and the other two stay in the filter above, which is the designed behaviour: the
    join cuts the candidate set without materializing a cartesian product, and the
    remaining pair is a cheap post-check on what survives.
    """
    a = pa.table(
        {
            "ax1": [0, 10, 100],
            "ax2": [20, 30, 120],
            "ay1": [0, 10, 100],
            "ay2": [20, 30, 120],
            "aid": list("abc"),
        }
    )
    b = pa.table(
        {"bx1": [5, 200], "bx2": [25, 220], "by1": [5, 200], "by2": [25, 220], "bid": list("xy")}
    )
    duck.register("a", a)
    duck.register("b", b)
    _check(
        duck,
        "SELECT aid, bid FROM a, b WHERE a.ax1 < b.bx2 AND a.ax2 > b.bx1 "
        "AND a.ay1 < b.by2 AND a.ay2 > b.by1",
        a=a,
        b=b,
    )


@pytest.mark.differential
@pytest.mark.parametrize(
    "cond",
    [
        # The canonical temporal proximity join: "events within a window of each other".
        "a.ts - 5 < b.ts AND a.ts + 5 > b.ts",
        "a.ts + 1 < b.ts",
        "b.ts > a.ts * 2",
        "a.ts < b.ts - 3",
    ],
)
def test_a_computed_operand_is_hoisted_into_the_join(duck, cond):
    """`a.ts - w < b.ts` has no column to sort until the expression is materialized.

    The rule computes it in a hidden column beneath the join, which is the same per-row
    work the filter over the cartesian product was already doing on the same rows. Without
    this the single most common temporal join shape stayed quadratic.
    """
    a = pa.table({"ts": [10, 20, 30, 40], "lab": list("abcd")})
    b = pa.table({"ts": [12, 25, 100], "tag": list("xyz")})
    duck.register("a", a)
    duck.register("b", b)
    _check(duck, f"SELECT lab, tag FROM a, b WHERE {cond}", a=a, b=b)


@pytest.mark.differential
def test_a_raising_computed_operand_is_not_hoisted(duck):
    """Integer division can raise, and the cartesian plan never evaluates it on an empty side.

    Hoisting it below the join would evaluate it on every left row even when the right side
    is empty — turning an empty result into an error. The rule must decline, which means the
    plan keeps the filter and the result still matches.
    """
    a = pa.table({"ts": [10, 20], "lab": ["a", "b"]})
    b = pa.table({"ts": [12], "tag": ["x"]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT lab, tag FROM a, b WHERE a.ts / 2 < b.ts"
    ds = bt.sql(query, a=a, b=b)
    ops = _ir_ops(optimize_logical(ds._plan).to_ir())
    assert "range_join" not in ops, "a raising expression must not be hoisted below the join"
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
def test_hoisting_does_not_leak_the_hidden_column(duck):
    """The hidden key is an implementation detail; the output columns must be unchanged."""
    a = pa.table({"ats": [10, 20, 30], "lab": list("abc")})
    b = pa.table({"bts": [12, 25], "tag": list("xy")})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT * FROM a, b WHERE a.ats + 1 < b.bts"
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    got = ds.collect()
    assert not [c for c in got.column_names if c.startswith("__rj_key")]
    assert_same(got, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    "cond",
    ["a.x < b.y", "a.x <= b.y", "a.x > b.y", "a.x >= b.y", "b.y > a.x", "b.y <= a.x"],
)
@pytest.mark.parametrize("negate", [False, True])
def test_inequality_correlated_exists(duck, cond, negate):
    """`EXISTS (SELECT 1 FROM b WHERE a.x < b.y)` is a range **semi** join.

    Before the range join existed this raised `NotImplementedError: correlated subqueries
    not supported` — only *equality*-correlated `EXISTS` decorrelated — so a shape DuckDB
    answers had no plan at all. It is the first caller to emit a `RangeJoin` whose
    `join_type` is not `inner`; the runtime already implemented `Semi`/`Anti` and fuzzed
    both against the cross-product oracle.
    """
    a = pa.table({"x": [1, 2, 3, 9, None], "lab": list("abcde")})
    b = pa.table({"y": [2, 5, None], "tag": list("xyz")})
    duck.register("a", a)
    duck.register("b", b)
    kw = "NOT EXISTS" if negate else "EXISTS"
    query = f"SELECT lab FROM a WHERE {kw} (SELECT 1 FROM b WHERE {cond})"
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
def test_correlated_exists_keeps_local_predicates_inside(duck):
    """A predicate on the inner table alone stays the inner relation's filter."""
    a = pa.table({"x": [1, 2, 3, 9], "lab": list("abcd")})
    b = pa.table({"y": [2, 5, 7], "tag": list("xyz")})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT lab FROM a WHERE EXISTS (SELECT 1 FROM b WHERE a.x < b.y AND b.y > 3)"
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
def test_correlated_exists_preserves_outer_duplicates(duck):
    """A semi join keeps each qualifying outer row, duplicates included.

    The tempting shortcut — cross join, filter, `DISTINCT` — collapses them, which is why
    this needs a real semi join rather than a rewrite over the operators already present.
    """
    a = pa.table({"x": [1, 1, 1, 9], "lab": ["a", "a", "a", "d"]})
    b = pa.table({"y": [5]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT lab FROM a WHERE EXISTS (SELECT 1 FROM b WHERE a.x < b.y)"
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    got = ds.collect()
    assert got.num_rows == 3, "three identical outer rows must survive as three"
    assert_same(got, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize(
    "cond",
    [
        "b.y > a.lo AND b.y < a.hi",
        "a.lo <= b.y AND a.hi >= b.y",
        "b.y >= a.lo AND a.hi > b.y",
    ],
)
def test_two_inequality_correlated_exists(duck, cond, negate):
    """Interval containment as an `EXISTS` — two correlations, one range semi join.

    ``EXISTS (SELECT 1 FROM b WHERE b.y > a.lo AND b.y < a.hi)`` is "does any inner row fall
    in this outer row's interval", which is the `EXISTS` spelling of the shape the operator
    was built for. Two conditions is the engine's ceiling, and a semi join cannot carry a
    residual (it emits no right columns to filter on), so a third inequality declines the
    whole shape rather than silently dropping one.
    """
    a = pa.table({"lo": [0, 2, 5, 8, None], "hi": [4, 6, 9, 20, 3], "lab": list("abcde")})
    b = pa.table({"y": [2, 5, 7, None], "tag": list("wxyz")})
    duck.register("a", a)
    duck.register("b", b)
    kw = "NOT EXISTS" if negate else "EXISTS"
    query = f"SELECT lab FROM a WHERE {kw} (SELECT 1 FROM b WHERE {cond})"
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))


@pytest.mark.differential
def test_two_correlations_with_a_local_predicate(duck):
    """A predicate on the inner table alone stays inside; only the correlations join."""
    a = pa.table({"lo": [0, 2, 5, 8], "hi": [4, 6, 9, 20], "lab": list("abcd")})
    b = pa.table({"y": [2, 5, 7], "tag": list("xyz")})
    duck.register("a", a)
    duck.register("b", b)
    query = (
        "SELECT lab FROM a WHERE EXISTS "
        "(SELECT 1 FROM b WHERE b.y > a.lo AND b.y < a.hi AND b.y <> 5)"
    )
    ds = bt.sql(query, a=a, b=b)
    _assert_range_join(ds)
    assert_same(ds.collect(), duck.sql(query))
