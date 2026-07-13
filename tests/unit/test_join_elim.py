"""Plan-shape unit tests for the `join_elim` (join-elimination) rules.

Each rule gets a *fires* test (the intended shape, with the proof present) and the
negative tests that matter for a family this dangerous: the proof is only *estimated*
(a SKETCH ndv / non-EXACT bounds), a column of the removed side **is** used, the join is
an **inner** join (which has no referential-integrity guarantee and so is never
eliminable), or the two "self-join" sides are not actually the same relation.
Result-correctness vs DuckDB lives in `tests/differential/test_diff_join_elim.py`.
"""

from __future__ import annotations

import dataclasses

import batcher as bt
import batcher.kyber.rules.extra.join_elim as je  # importing registers the rules
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import lit
from batcher.plan.logical import Filter, Join, Limit, Project
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

RULE_NAMES = {
    "anti_join_of_nonempty_cartesian_to_empty",
    "eliminate_cross_join_of_single_row",
    "eliminate_left_join_under_distinct",
    "inner_join_to_semi_when_right_unique",
    "join_disjoint_keys_to_empty",
    "no_match_join_to_preserved_side",
    "self_anti_join_to_null_keys",
    "self_join_elimination",
    "self_semi_join_to_filter",
    "semi_join_of_nonempty_cartesian",
}


# --- fixtures / helpers -------------------------------------------------------


def _fact():
    return bt.from_pydict({"k": [1, 2, 2, 3], "v": [10, 20, 30, 40]})


def _dim():
    return bt.from_pydict({"k": [1, 2, 9], "w": [5, 6, 7]})


def _ctx(ds, stats=None):
    return Optimizer(sources=ds._sources, source_stats=stats)._context()


def _rewrite(ds, stats=None):
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _has(plan, kind) -> bool:
    return any(isinstance(n, kind) for n in walk(plan))


def _join_node(plan) -> Join:
    """The plan's `Join` (a `full` join is wrapped in a key-coalescing `Project`)."""
    return next(n for n in walk(plan) if isinstance(n, Join))


def _single_side(plan, side: str) -> Join:
    """The plan's join with its output pruned to one side (what column pruning produces)."""
    join = _join_node(plan)
    return dataclasses.replace(join, output=tuple(o for o in join.output if o.side == side))


def _unique_key(rows: int, key: str, *, null_count: int = 0, exact: bool = True):
    """A source declaring `key` as a primary key: ndv == row count, EXACT (or SKETCH)."""
    prov = Provenance.EXACT if exact else Provenance.SKETCH
    return SourceStatistics(
        row_count=rows,
        columns={key: ColumnStat(ndv=rows, null_count=null_count, provenance=prov)},
    )


def _bounds(lo, hi, *, exact: bool = True) -> ColumnStat:
    prov = Provenance.EXACT if exact else Provenance.DEFAULT
    return ColumnStat(min=lo, max=hi, null_count=0, provenance=prov)


def _ranged_join(how: str, *, exact: bool = True, overlap: bool = False):
    """A join of two sources whose key ranges are declared disjoint (or overlapping)."""
    left = bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]})
    right = bt.from_pydict({"k": [100, 101], "w": [1, 2]})
    ds = left.join(right, on="k", how=how)
    hi = 3 if not overlap else 200
    stats = [
        SourceStatistics(row_count=3, columns={"k": _bounds(1, hi, exact=exact)}),
        SourceStatistics(row_count=2, columns={"k": _bounds(100, 101, exact=exact)}),
    ]
    return ds, stats


def test_rules_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= RULE_NAMES


# --- eliminate_left_join_under_distinct ---------------------------------------


def test_left_join_under_distinct_fires():
    ds = _fact().join(_dim(), on="k", how="left").select("k", "v").distinct()
    out = je.eliminate_left_join_under_distinct(ds._plan, None)
    assert out is not None and not _has(out, Join)  # the join is gone; no key proof needed


def test_left_join_under_distinct_right_join_mirror():
    ds = _fact().join(_dim(), on="k", how="right").select("w").distinct()
    out = je.eliminate_left_join_under_distinct(ds._plan, None)
    assert out is not None and not _has(out, Join)  # a right join preserves its right side


def test_left_join_under_distinct_negatives():
    # A right column is read → the join carries data and must stay.
    used = _fact().join(_dim(), on="k", how="left").select("k", "w").distinct()
    assert je.eliminate_left_join_under_distinct(used._plan, None) is None
    # An INNER join also *filters* (no referential integrity) → never eliminable here.
    inner = _fact().join(_dim(), on="k", how="inner").select("k", "v").distinct()
    assert je.eliminate_left_join_under_distinct(inner._plan, None) is None
    # A FULL join null-extends right-only rows too → it would add an all-null left tuple.
    full = _fact().join(_dim(), on="k", how="full").select("k", "v").distinct()
    assert je.eliminate_left_join_under_distinct(full._plan, None) is None


# --- inner_join_to_semi_when_right_unique -------------------------------------


def test_inner_join_to_semi_fires_on_structural_unique():
    dim = bt.from_pydict({"k": [1, 1, 2]}).distinct()  # unique on exactly the join key
    ds = _fact().join(dim, on="k", how="inner")
    join = _single_side(ds._plan, "left")
    out = je.inner_join_to_semi_when_right_unique(join, _ctx(ds))
    assert isinstance(out, Join) and out.join_type == "semi"
    assert out.output == join.output  # schema untouched
    # Idempotent: a semi join is not an inner join.
    assert je.inner_join_to_semi_when_right_unique(out, _ctx(ds)) is None


def test_inner_join_to_semi_fires_on_exact_ndv():
    ds = _fact().join(_dim(), on="k", how="inner")
    stats = [None, _unique_key(3, "k")]
    out = je.inner_join_to_semi_when_right_unique(_single_side(ds._plan, "left"), _ctx(ds, stats))
    assert isinstance(out, Join) and out.join_type == "semi"


def test_inner_join_to_semi_negatives():
    ds = _fact().join(_dim(), on="k", how="inner")
    left_only = _single_side(ds._plan, "left")
    # Uniqueness merely *estimated* (a sketched ndv) → not a proof → no rewrite.
    sketched = [None, _unique_key(3, "k", exact=False)]
    assert je.inner_join_to_semi_when_right_unique(left_only, _ctx(ds, sketched)) is None
    # No stats at all → uniqueness unprovable.
    assert je.inner_join_to_semi_when_right_unique(left_only, _ctx(ds)) is None
    # A right column is in the output → a semi join could not produce it.
    stats = [None, _unique_key(3, "k")]
    assert je.inner_join_to_semi_when_right_unique(ds._plan, _ctx(ds, stats)) is None
    # A left join is not an inner join (it is `eliminate_left_join`'s business).
    left = _single_side(_fact().join(_dim(), on="k", how="left")._plan, "left")
    assert je.inner_join_to_semi_when_right_unique(left, _ctx(ds, stats)) is None


# --- self_join_elimination ----------------------------------------------------


def _self_join(how: str = "inner"):
    ds = bt.from_pydict({"id": [1, 2, 3], "v": [10, 20, 30]})
    return ds.join(ds, on="id", how=how)


def test_self_join_elimination_fires():
    ds = _self_join()
    stats = [_unique_key(3, "id")] * 2
    out = je.self_join_elimination(_single_side(ds._plan, "left"), _ctx(ds, stats))
    assert isinstance(out, Project) and not _has(out, Join)
    assert [i.alias for i in out.items] == ["id", "v"]  # schema preserved, in order


def test_self_join_elimination_fires_on_full_join():
    ds = _self_join("full")  # the full join sits under a key-coalescing Project
    stats = [_unique_key(3, "id")] * 2
    out = je.self_join_elimination(_single_side(ds._plan, "left"), _ctx(ds, stats))
    assert isinstance(out, Project)  # every row matches itself → nothing to null-extend


def test_self_join_elimination_negatives():
    ds = _self_join()
    left_only = _single_side(ds._plan, "left")
    # Uniqueness only estimated → no proof, no rewrite.
    assert (
        je.self_join_elimination(left_only, _ctx(ds, [_unique_key(3, "id", exact=False)] * 2))
        is None
    )
    # Unique but *nullable*: a null-keyed row does not match itself → the inner join
    # would drop it. Uniqueness alone is not enough.
    nullable = [_unique_key(4, "id", null_count=1)] * 2
    assert je.self_join_elimination(left_only, _ctx(ds, nullable)) is None
    # No stats → nothing proven.
    assert je.self_join_elimination(left_only, _ctx(ds)) is None
    # A two-sided output cannot be carried by one input.
    assert je.self_join_elimination(ds._plan, _ctx(ds, [_unique_key(3, "id")] * 2)) is None
    # Two *different* relations that merely look alike are not a self-join.
    a = bt.from_pydict({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = bt.from_pydict({"id": [1, 2, 3], "v": [10, 20, 30]})
    other = a.join(b, on="id", how="inner")
    stats = [_unique_key(3, "id")] * 2
    assert je.self_join_elimination(_single_side(other._plan, "left"), _ctx(other, stats)) is None


# --- self_semi_join_to_filter / self_anti_join_to_null_keys -------------------


def test_self_semi_join_to_filter_fires_without_stats():
    ds = _self_join("semi")
    out = je.self_semi_join_to_filter(ds._plan, _ctx(ds))
    assert isinstance(out, Project) and isinstance(out.input, Filter)  # `id IS NOT NULL`
    assert not _has(out, Join)
    assert [i.alias for i in out.items] == ["id", "v"]


def test_self_semi_join_to_filter_drops_filter_when_non_null_proven():
    ds = _self_join("semi")
    out = je.self_semi_join_to_filter(ds._plan, _ctx(ds, [_unique_key(3, "id")] * 2))
    assert isinstance(out, Project) and not _has(out, Filter)  # provably no null key


def test_self_anti_join_fires_without_stats():
    ds = _self_join("anti")
    out = je.self_anti_join_to_null_keys(ds._plan, _ctx(ds))
    assert isinstance(out, Project) and isinstance(out.input, Filter)  # `id IS NULL`
    assert not _has(out, Join)


def test_self_anti_join_is_empty_when_non_null_proven():
    ds = _self_join("anti")
    out = je.self_anti_join_to_null_keys(ds._plan, _ctx(ds, [_unique_key(3, "id")] * 2))
    assert isinstance(out, Limit) and out.n == 0  # no row can fail to match itself


def test_self_semi_anti_negatives():
    ds = _self_join("semi")
    # A different relation on the right is not a self-join.
    a = bt.from_pydict({"id": [1, 2], "v": [1, 2]})
    b = bt.from_pydict({"id": [1, 2], "v": [1, 2]})
    other = a.join(b, on="id", how="semi")
    assert je.self_semi_join_to_filter(other._plan, _ctx(other)) is None
    # An inner self-join is not a semi join (it can duplicate) → different rule.
    assert je.self_semi_join_to_filter(_self_join("inner")._plan, _ctx(ds)) is None
    assert je.self_anti_join_to_null_keys(_self_join("semi")._plan, _ctx(ds)) is None


# --- cartesian joins ----------------------------------------------------------


def _cartesian(how: str, right):
    left = bt.from_pydict({"x": [1, 2, 3]})
    key = "ck"
    return left.with_columns(**{key: lit(1)}).join(
        right.with_columns(**{key: lit(1)}), on=key, how=how
    )


def test_cross_join_of_single_row_eliminated():
    scalar = bt.from_pydict({"y": [5, 6]}).agg(m=col("y").max())  # EXACT one row
    out = _rewrite(bt.from_pydict({"x": [1, 2, 3]}).cross_join(scalar).select("x"))
    assert not _has(out, Join)


def test_cross_join_of_single_row_negatives():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    two_rows = bt.from_pydict({"y": [5, 6]})
    assert _has(_rewrite(ds.cross_join(two_rows).select("x")), Join)  # 2 rows → fans out
    scalar = two_rows.agg(m=col("y").max())
    assert _has(_rewrite(ds.cross_join(scalar).select("x", "m")), Join)  # `m` is read


def test_semi_join_of_nonempty_cartesian_fires():
    ds = _cartesian("semi", bt.from_pydict({"y": [5, 6]}))
    out = je.semi_join_of_nonempty_cartesian(ds._plan, _ctx(ds))
    assert isinstance(out, Project) and not _has(out, Join)


def test_anti_join_of_nonempty_cartesian_to_empty_fires():
    ds = _cartesian("anti", bt.from_pydict({"y": [5, 6]}))
    out = je.anti_join_of_nonempty_cartesian_to_empty(ds._plan, _ctx(ds))
    assert isinstance(out, Limit) and out.n == 0


def test_cartesian_semi_anti_negatives():
    # An empty right side proves the *opposite* verdict → these rules stand down.
    empty = bt.from_pydict({"y": []})
    semi = _cartesian("semi", empty)
    assert je.semi_join_of_nonempty_cartesian(semi._plan, _ctx(semi)) is None
    anti = _cartesian("anti", empty)
    assert je.anti_join_of_nonempty_cartesian_to_empty(anti._plan, _ctx(anti)) is None
    # A real (non-constant) key is not cartesian: matching is not guaranteed.
    real = _fact().join(_dim(), on="k", how="semi")
    assert je.semi_join_of_nonempty_cartesian(real._plan, _ctx(real)) is None


# --- disjoint key ranges ------------------------------------------------------


def test_join_disjoint_keys_to_empty_fires():
    ds, stats = _ranged_join("inner")
    ctx = _ctx(ds, stats)
    out = je.join_disjoint_keys_to_empty(ds._plan, ctx)
    assert isinstance(out, Join) and isinstance(out.left, Limit) and out.left.n == 0
    assert out.output == ds._plan.output  # schema untouched
    assert je.join_disjoint_keys_to_empty(out, ctx) is None  # idempotent


def test_join_disjoint_keys_to_empty_negatives():
    # Overlapping ranges → a match is possible.
    ds, stats = _ranged_join("inner", overlap=True)
    assert je.join_disjoint_keys_to_empty(ds._plan, _ctx(ds, stats)) is None
    # Disjoint but only *estimated* bounds → not a proof.
    ds, stats = _ranged_join("inner", exact=False)
    assert je.join_disjoint_keys_to_empty(ds._plan, _ctx(ds, stats)) is None
    # An anti join keeps *every* left row when nothing matches → emptying it is wrong.
    ds, stats = _ranged_join("anti")
    assert je.join_disjoint_keys_to_empty(ds._plan, _ctx(ds, stats)) is None


def test_no_match_join_to_preserved_side_fires():
    for how in ("left", "anti"):
        ds, stats = _ranged_join(how)
        join = _single_side(ds._plan, "left")
        out = je.no_match_join_to_preserved_side(join, _ctx(ds, stats))
        assert isinstance(out, Project) and not _has(out, Join)
        assert [i.alias for i in out.items] == [o.alias for o in join.output]


def test_no_match_join_to_preserved_side_negatives():
    # A FULL join preserves both sides → |L| + |R| rows, not a passthrough of either.
    ds, stats = _ranged_join("full")
    assert (
        je.no_match_join_to_preserved_side(_single_side(ds._plan, "left"), _ctx(ds, stats)) is None
    )
    # Output spans both sides → the null-extended right columns cannot be fabricated.
    ds, stats = _ranged_join("left")
    assert je.no_match_join_to_preserved_side(ds._plan, _ctx(ds, stats)) is None
    # Overlapping ranges → matches are possible.
    ds, stats = _ranged_join("left", overlap=True)
    assert (
        je.no_match_join_to_preserved_side(_single_side(ds._plan, "left"), _ctx(ds, stats)) is None
    )
