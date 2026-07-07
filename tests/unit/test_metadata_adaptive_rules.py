"""Plan-shape unit tests for the `metadata_adaptive` EXACT-gated rewrites.

Each rule gets a fires test (the rewrite yields the intended shape), a does-not-fire
test (the proof is only LEARNED/SKETCH/absent, or the operator legitimately applies),
and an idempotence check. Result-correctness vs DuckDB lives in the differential suite
(`tests/differential/test_diff_metadata_adaptive.py`).
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.metadata_adaptive import (  # importing registers the rules
    drop_distinct_when_unique,
    prune_constant_sort_keys,
    prune_filter_col_comparison,
    skip_sort_of_single_row,
)
from batcher.plan.logical import Distinct, Filter, Limit, Sort
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk


def _rewrite(ds, stats=None):
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _ctx(ds, stats=None):
    return Optimizer(sources=ds._sources, source_stats=stats)._context()


def _has(out, kind) -> bool:
    return any(isinstance(n, kind) for n in walk(out))


def _exact_col(**kw) -> ColumnStat:
    return ColumnStat(provenance=Provenance.EXACT, **kw)


# --- registration --------------------------------------------------------------


def test_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {
        "skip_sort_of_single_row",
        "prune_constant_sort_keys",
        "drop_distinct_when_unique",
        "prune_filter_col_comparison",
    } <= names


# --- skip_sort_of_single_row ---------------------------------------------------


def test_skip_sort_of_single_row_fires_on_one_row_source():
    ds = bt.from_pydict({"x": [5]})  # exactly one row, EXACT
    out = _rewrite(ds.sort("x"))
    assert not _has(out, Sort)


def test_skip_sort_of_single_row_fires_over_global_aggregate():
    ds = bt.from_pydict({"x": [1, 2, 3, 4]})
    out = _rewrite(ds.agg(s=col("x").sum()).sort("s"))  # aggregate is EXACTLY one row
    assert not _has(out, Sort)


def test_skip_sort_of_single_row_kept_when_multi_row():
    ds = bt.from_pydict({"x": [3, 1, 2]})  # 3 rows → the sort really orders
    assert skip_sort_of_single_row(ds.sort("x")._plan, _ctx(ds)) is None


def test_skip_sort_of_single_row_kept_without_exact_size():
    ds = bt.from_pydict({"x": [3, 1, 2]})
    # A filter downgrades provenance away from EXACT, so the size is only estimated.
    plan = ds.filter(col("x") > 0).sort("x")._plan
    assert skip_sort_of_single_row(plan, _ctx(ds)) is None


def test_skip_sort_of_single_row_idempotent():
    ds = bt.from_pydict({"x": [5]})
    once = _rewrite(ds.sort("x"))
    twice = Optimizer(sources=ds._sources).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- prune_constant_sort_keys --------------------------------------------------


def test_prune_constant_sort_keys_drops_the_constant_key():
    ds = bt.from_pydict({"x": [3, 1, 2]})
    out = _rewrite(ds.with_columns(k=7).sort("k", "x"))  # k is a literal constant
    sorts = [n for n in walk(out) if isinstance(n, Sort)]
    assert len(sorts) == 1
    assert [key.expr.name for key in sorts[0].keys] == ["x"]


def test_prune_constant_sort_keys_all_constant_drops_sort():
    ds = bt.from_pydict({"x": [3, 1, 2]})
    out = _rewrite(ds.with_columns(k=7).sort("k"))  # only key is constant, no limit
    assert not _has(out, Sort)


def test_prune_constant_sort_keys_kept_for_non_constant_keys():
    ds = bt.from_pydict({"x": [3, 1, 2], "y": [1, 2, 3]})
    # Neither column has EXACT constant stats (in-memory source declares none).
    assert prune_constant_sort_keys(ds.sort("x", "y")._plan, _ctx(ds)) is None


def test_prune_constant_sort_keys_idempotent():
    ds = bt.from_pydict({"x": [3, 1, 2]})
    once = _rewrite(ds.with_columns(k=7).sort("k", "x"))
    twice = Optimizer(sources=ds._sources).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- drop_distinct_when_unique -------------------------------------------------


def _unique_stats(ndv, provenance=Provenance.EXACT, rows=3):
    return [
        SourceStatistics(
            row_count=rows,
            columns={"id": ColumnStat(ndv=ndv, null_count=0, provenance=provenance)},
        )
    ]


def test_drop_distinct_when_unique_fires():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    out = _rewrite(ds.distinct(), stats=_unique_stats(3))  # EXACT ndv 3 == 3 rows
    assert not _has(out, Distinct)


def test_drop_distinct_when_unique_kept_when_ndv_below_rows():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    # EXACT but only 2 distinct over 3 rows → a duplicate may exist → keep the dedup.
    assert drop_distinct_when_unique(ds.distinct()._plan, _ctx(ds, _unique_stats(2))) is None


def test_drop_distinct_when_unique_kept_when_ndv_only_sketch():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    # A sketch/HLL ndv is an estimate — never enough to drop a dedup.
    stats = _unique_stats(3, provenance=Provenance.SKETCH)
    assert drop_distinct_when_unique(ds.distinct()._plan, _ctx(ds, stats)) is None


def test_drop_distinct_when_unique_kept_without_stats():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    assert drop_distinct_when_unique(ds.distinct()._plan, _ctx(ds)) is None


def test_drop_distinct_when_unique_idempotent():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    stats = _unique_stats(3)
    once = _rewrite(ds.distinct(), stats=stats)
    twice = Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- prune_filter_col_comparison -----------------------------------------------


def _two_col_stats(a_range, b_range, rows=2):
    return [
        SourceStatistics(
            row_count=rows,
            columns={
                "a": _exact_col(min=a_range[0], max=a_range[1], null_count=0),
                "b": _exact_col(min=b_range[0], max=b_range[1], null_count=0),
            },
        )
    ]


def test_prune_filter_col_comparison_always_true_drops_filter():
    ds = bt.from_pydict({"a": [0, 1], "b": [10, 20]})
    stats = _two_col_stats((0, 1), (10, 20))  # max(a)=1 < min(b)=10 → a<b always
    out = _rewrite(ds.filter(col("a") < col("b")), stats=stats)
    assert not _has(out, Filter)


def test_prune_filter_col_comparison_always_false_is_empty():
    ds = bt.from_pydict({"a": [10, 20], "b": [0, 1]})
    stats = _two_col_stats((10, 20), (0, 1))  # min(a)=10 >= max(b)=1 → a<b never
    out = _rewrite(ds.filter(col("a") < col("b")), stats=stats)
    assert isinstance(out, Limit) and out.n == 0


def test_prune_filter_col_comparison_undecidable_kept():
    ds = bt.from_pydict({"a": [0, 50], "b": [10, 20]})
    stats = _two_col_stats((0, 50), (10, 20))  # ranges overlap → undecidable
    assert (
        prune_filter_col_comparison(ds.filter(col("a") < col("b"))._plan, _ctx(ds, stats)) is None
    )


def test_prune_filter_col_comparison_kept_without_exact_bounds():
    ds = bt.from_pydict({"a": [0, 1], "b": [10, 20]})
    # No source stats → no EXACT bounds → the col-vs-col filter is left to execute.
    assert prune_filter_col_comparison(ds.filter(col("a") < col("b"))._plan, _ctx(ds)) is None


def test_prune_filter_col_comparison_idempotent():
    ds = bt.from_pydict({"a": [0, 1], "b": [10, 20]})
    stats = _two_col_stats((0, 1), (10, 20))
    once = _rewrite(ds.filter(col("a") < col("b")), stats=stats)
    twice = Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()
