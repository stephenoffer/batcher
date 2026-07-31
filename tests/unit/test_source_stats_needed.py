"""A rule that cannot see is indistinguishable from a rule that is absent.

`collect_source_stats` narrows the per-column bounds it computes for a resident source to the
columns `column_bounds_needed` names, because building a zone map costs a pass over the column
(~46 ms per column at 10M rows). That narrowing is a silent trap: a rule reading a column the
list forgot sees `Provenance.DEFAULT` and declines, so it is written, tested in isolation,
correct — and dead on every real query. Two consumers were missed that way already, and the
second had the sharper symptom: `SELECT min(x), max(x) FROM t` has no filter and no join, so the
needed set came back empty, every column's min/max was narrowed away, and the rule whose whole
job is to answer that query from metadata scanned the table instead.

These tests close the loop the unit tests around each rule cannot: they drive each shortcut
through the *narrowed* statistics the execution path really passes, so a consumer added without
its column being fetched fails here rather than in production.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.plan.logical import Aggregate, Filter, Limit, Sort
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _narrowed_plan(dataset):
    """The optimized plan, using exactly the statistics `collect()` would collect.

    This is the whole point: passing the *full* statistics would make every one of these pass
    whether or not `column_bounds_needed` names the column.
    """
    from batcher import core, kyber

    hub = core.default_hub()
    need = column_bounds_needed(dataset._plan)
    stats = collect_source_stats(dataset._sources, hub, need_columns=need)
    return kyber.optimize_logical(
        dataset._plan, sources=dataset._sources, hub=hub, source_stats=stats
    )


def _ds():
    """Four rows: distinct `x`, duplicated `g` (a real group key), constant `c`."""
    return bt.from_pydict(
        {"x": [5, 1, 9, 3], "y": [1, 2, 3, 4], "g": [1, 1, 2, 2], "c": [7, 7, 7, 7]}
    )


def test_global_min_max_is_answered_from_metadata_when_narrowed():
    # The regression this file exists for. `min`/`max` reference no filter and no join, so the
    # needed set was empty, the bounds were narrowed away, and the scan happened anyway.
    plan = _narrowed_plan(_ds().agg(mn=col("x").min(), mx=col("x").max()))
    assert not any(isinstance(n, Aggregate) for n in walk(plan)), (
        f"the aggregate survived, so the bounds were not fetched: "
        f"{[type(n).__name__ for n in walk(plan)]}"
    )


def test_aggregate_input_columns_are_named_as_needed():
    assert column_bounds_needed(_ds().agg(mn=col("x").min())._plan) == {"x"}


def test_group_keys_are_named_as_needed():
    # `drop_constant_group_key` proves a key constant from min == max; `x` comes along because
    # `count_of_non_null_column` reads its null count to rewrite `count(x)` as `count(*)`.
    assert column_bounds_needed(_ds().group_by("g").agg(n=col("x").count())._plan) == {"g", "x"}


def test_only_the_aggregates_that_read_column_statistics_name_their_input():
    """An aggregate whose input no rule reads must not pull that column's statistics in.

    Fetching a column materializes its whole `ColumnStat`, quantile grid included, and the
    `approx_*` terminals answer from a sketch *when one exists* — so requesting a column for an
    unrelated reason changes what `ds.approx_percentile` on that column returns. Asking only for
    the inputs a rule actually reads is what keeps that from happening.
    """
    assert column_bounds_needed(_ds().agg(m=col("x").min())._plan) == {"x"}
    assert column_bounds_needed(_ds().agg(s=col("x").sum())._plan) == {"x"}
    assert column_bounds_needed(_ds().agg(n=col("x").count())._plan) == {"x"}
    # No rule reads the input column's statistics for these, so neither does the collector.
    assert column_bounds_needed(_ds().agg(q=col("x").quantile(0.5))._plan) == set()
    assert column_bounds_needed(_ds().agg(d=col("x").n_unique())._plan) == set()


def test_sort_keys_are_named_as_needed():
    # `prune_constant_sort_keys`, `prune_sort_keys_after_unique_key` and
    # `skip_sort_of_single_row` all read the key column's statistics.
    assert column_bounds_needed(_ds().sort("x", "y")._plan) == {"x", "y"}


def test_filter_columns_are_still_named_as_needed():
    assert column_bounds_needed(_ds().filter(col("x") > 2)._plan) == {"x"}


def test_join_keys_are_still_named_as_needed():
    other = bt.from_pydict({"x": [1, 2], "z": [3, 4]})
    assert column_bounds_needed(_ds().join(other, on="x")._plan) == {"x"}


def test_zonemap_pruning_still_fires_when_narrowed():
    # The consumer that was never broken — pinned so a future narrowing cannot break it either.
    plan = _narrowed_plan(_ds().filter(col("x") > 100))
    assert any(isinstance(n, Limit) and n.n == 0 for n in walk(plan)), (
        f"the always-false filter was not pruned: {[type(n).__name__ for n in walk(plan)]}"
    )


def test_sargable_transposition_still_fires_when_narrowed():
    # `sarg_bounded_ordered` needs the column's range to prove its arithmetic cannot wrap, and
    # reaches it through the same narrowed statistics.
    plan = _narrowed_plan(_ds().filter((col("x") + 1) > 5))
    predicates = [n.predicate.to_ir() for n in walk(plan) if isinstance(n, Filter)]
    assert predicates and predicates[0]["left"] == {"e": "col", "name": "x"}, (
        f"the arithmetic was not transposed off the column: {predicates}"
    )


def test_constant_leading_sort_key_is_pruned_when_narrowed():
    # A key whose column is constant contributes no ordering, and `prune_constant_sort_keys`
    # proves that from min == max — which the narrowing used to remove, so the query sorted on a
    # useless leading key.
    plan = _narrowed_plan(bt.from_pydict({"c": [7, 7, 7, 7], "x": [5, 1, 9, 3]}).sort("c", "x"))
    keys = [k.expr.name for n in walk(plan) if isinstance(n, Sort) for k in n.keys]
    assert keys == ["x"], f"the constant key survived: {keys}"


def test_single_row_sort_is_skipped_when_narrowed():
    plan = _narrowed_plan(bt.from_pydict({"x": [5]}).sort("x"))
    assert not any(isinstance(n, Sort) for n in walk(plan)), (
        f"the sort of a one-row relation survived: {[type(n).__name__ for n in walk(plan)]}"
    )


@pytest.mark.parametrize(
    ("build", "label"),
    [
        (lambda ds: ds.agg(mn=col("x").min()), "global-min"),
        (lambda ds: ds.filter(col("x") > 100), "always-false-filter"),
        (lambda ds: ds.filter((col("x") + 1) > 5), "sargable-transposition"),
        (lambda ds: ds.sort("x", "y"), "sort-keys"),
        (lambda ds: ds.group_by("g").agg(n=col("x").count()), "group-keys"),
        (lambda ds: ds.sort("c", "x"), "constant-sort-key"),
    ],
)
def test_narrowed_and_full_statistics_produce_the_same_plan(build, label):
    """The narrowing is a *cost* optimization, so it must not change the plan.

    Any divergence means `column_bounds_needed` is missing a column the optimizer reads — the
    exact failure mode this module guards, stated as an equivalence rather than per-rule.
    """
    from batcher import core, kyber

    hub = core.default_hub()
    dataset = build(_ds())
    full = collect_source_stats(dataset._sources, hub, need_columns=None)
    with_full = kyber.optimize_logical(
        dataset._plan, sources=dataset._sources, hub=hub, source_stats=full
    )
    assert _narrowed_plan(dataset).to_ir() == with_full.to_ir(), (
        f"{label}: narrowed statistics produced a different plan than full ones"
    )
