"""Plan-shape, idempotence, and negative tests for the `projection_scan` rules."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.projection_scan import (
    _non_nullable_cols,
    dedupe_sort_keys,
    drop_always_true_null_check,
    drop_self_cast_in_filter,
    drop_self_cast_in_projection,
    drop_self_cast_in_sort_key,
    empty_on_impossible_null_check,
    empty_sample_n,
    fold_nested_sample_same_seed,
    identity_full_sample,
    merge_projection_renames,
)
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Filter,
    Limit,
    Project,
    Projection,
    Sample,
    Scan,
    Sort,
    SortKeySpec,
)

_RULE_NAMES = {
    "dedupe_sort_keys",
    "drop_always_true_null_check",
    "drop_self_cast_in_filter",
    "drop_self_cast_in_projection",
    "drop_self_cast_in_sort_key",
    "empty_on_impossible_null_check",
    "empty_sample_n",
    "fold_nested_sample_same_seed",
    "identity_full_sample",
    "merge_projection_renames",
}


def _scan() -> Scan:
    return bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})._plan


def _nonnull_scan() -> Scan:
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False), pa.field("y", pa.int64())])
    return bt.from_arrow(pa.table({"x": [1, 2, 3], "y": [10, 20, 30]}, schema=schema))._plan


def test_all_rules_registered():
    registered = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert registered >= _RULE_NAMES


# --- the removed `eliminate_sort_before_sample` -------------------------------
#
# That rule rewrote `Sample(Sort(x))` -> `Sample(x)`, which is wrong: `Sample` is
# order-preserving in the engine, so it changed the order of the rows the user got back.
# The sound form of the optimization now lives on the order-indifferent consumer
# (`eliminate_sort_before_aggregate`, which looks through a `Sample`). This asserts the
# rule is *gone*, so reinstating it fails here rather than silently reordering results.


def test_sort_before_sample_rule_is_not_registered():
    assert "eliminate_sort_before_sample" not in {r.name for r in DEFAULT_REGISTRY.rules()}


# --- dedupe_sort_keys --------------------------------------------------------


def test_dedupe_sort_keys():
    scan = _scan()
    node = Sort(scan, (SortKeySpec(Col("x")), SortKeySpec(Col("y")), SortKeySpec(Col("x"))))
    out = dedupe_sort_keys(node, None)
    assert isinstance(out, Sort)
    assert [k.expr.name for k in out.keys] == ["x", "y"]
    assert dedupe_sort_keys(out, None) is None


def test_dedupe_sort_keys_negative():
    scan = _scan()
    two = Sort(scan, (SortKeySpec(Col("x")), SortKeySpec(Col("y"))))
    assert dedupe_sort_keys(two, None) is None
    # same column but a different direction is not an exact duplicate → kept.
    node = Sort(scan, (SortKeySpec(Col("x")), SortKeySpec(Col("x"), descending=True)))
    assert dedupe_sort_keys(node, None) is None


# --- null-check rules (schema NOT NULL) --------------------------------------


def test_non_nullable_cols_from_scan():
    assert _non_nullable_cols(_nonnull_scan()) == frozenset({"x"})
    assert _non_nullable_cols(_scan()) == frozenset()


def test_drop_always_true_not_null_only_conjunct():
    scan = _nonnull_scan()
    node = Filter(scan, col("x").is_not_null())
    assert drop_always_true_null_check(node, None) is scan


def test_drop_always_true_not_null_keeps_other_conjunct():
    scan = _nonnull_scan()
    node = Filter(scan, col("x").is_not_null() & (col("y") > 1))
    out = drop_always_true_null_check(node, None)
    assert isinstance(out, Filter)
    assert out.predicate.to_ir() == (col("y") > 1).to_ir()
    assert drop_always_true_null_check(out, None) is None  # idempotent


def test_drop_always_true_not_null_negative():
    scan = _nonnull_scan()
    # `y` is nullable → its IS NOT NULL is not provably true.
    assert drop_always_true_null_check(Filter(scan, col("y").is_not_null()), None) is None
    # a non-null-check predicate is untouched.
    assert drop_always_true_null_check(Filter(scan, col("x") > 1), None) is None


def test_empty_on_impossible_is_null():
    scan = _nonnull_scan()
    out = empty_on_impossible_null_check(Filter(scan, col("x").is_null()), None)
    assert isinstance(out, Limit) and out.n == 0 and out.input is scan
    # inside a conjunction, still impossible → empty.
    node = Filter(scan, col("x").is_null() & (col("y") > 1))
    assert isinstance(empty_on_impossible_null_check(node, None), Limit)


def test_empty_on_impossible_is_null_negative():
    scan = _nonnull_scan()
    assert empty_on_impossible_null_check(Filter(scan, col("y").is_null()), None) is None
    assert empty_on_impossible_null_check(Filter(scan, col("x").is_not_null()), None) is None


# --- sample bounds -----------------------------------------------------------


def test_empty_sample_n_zero():
    scan = _scan()
    out = empty_sample_n(Sample(scan, fraction=1.0, seed=0, n=0), None)
    assert isinstance(out, Limit) and out.n == 0 and out.input is scan


def test_empty_sample_n_negative():
    scan = _scan()
    assert empty_sample_n(Sample(scan, fraction=0.5, seed=0), None) is None  # fraction mode
    assert empty_sample_n(Sample(scan, fraction=1.0, seed=0, n=3), None) is None


def test_identity_full_sample():
    scan = _scan()
    assert identity_full_sample(Sample(scan, fraction=1.0, seed=0), None) is scan


def test_identity_full_sample_negative():
    scan = _scan()
    assert identity_full_sample(Sample(scan, fraction=0.5, seed=0), None) is None
    assert identity_full_sample(Sample(scan, fraction=1.0, seed=0, n=2), None) is None


def test_fold_nested_sample_same_seed():
    scan = _scan()
    node = Sample(Sample(scan, 0.6, 5), 0.4, 5)
    out = fold_nested_sample_same_seed(node, None)
    assert isinstance(out, Sample) and out.input is scan
    assert out.fraction == 0.4 and out.seed == 5
    assert fold_nested_sample_same_seed(out, None) is None


def test_fold_nested_sample_negative():
    scan = _scan()
    assert fold_nested_sample_same_seed(Sample(Sample(scan, 0.6, 5), 0.4, 6), None) is None  # seed
    assert fold_nested_sample_same_seed(Sample(Sample(scan, 0.6, 5, n=3), 0.4, 5), None) is None
    assert fold_nested_sample_same_seed(Sample(scan, 0.5, 5), None) is None  # not nested


# --- merge_projection_renames ------------------------------------------------


def _rename_stack() -> Project:
    scan = _scan()
    inner = Project(scan, (Projection("a", Col("x")), Projection("keep", Col("y"))))
    # `a` (a bare rename) is referenced twice; `keep` once.
    return Project(inner, (Projection("p", col("a") + col("keep")), Projection("q", col("a") * 2)))


def test_merge_projection_renames_inlines_bare_col():
    node = _rename_stack()
    out = merge_projection_renames(node, None)
    assert isinstance(out, Project) and isinstance(out.input, Scan)
    assert [it.alias for it in out.items] == ["p", "q"]
    # `a` → `x` inlined into both outputs.
    assert out.items[0].expr.to_ir() == (col("x") + col("y")).to_ir()
    assert out.items[1].expr.to_ir() == (col("x") * 2).to_ir()
    assert merge_projection_renames(out, None) is None


def test_merge_projection_renames_skips_multi_ref_computed():
    scan = _scan()
    inner = Project(scan, (Projection("a", col("x") + 1), Projection("keep", Col("y"))))
    node = Project(inner, (Projection("p", col("a") + col("keep")), Projection("q", col("a") * 2)))
    # `a` is computed and referenced twice → inlining would duplicate work.
    assert merge_projection_renames(node, None) is None


def test_merge_projection_renames_defers_single_ref_to_merge():
    scan = _scan()
    inner = Project(scan, (Projection("a", Col("x")), Projection("keep", Col("y"))))
    node = Project(inner, (Projection("p", col("a")), Projection("q", col("keep"))))
    # every inner column referenced once → this is `merge_projections`' job, not ours.
    assert merge_projection_renames(node, None) is None


# --- self-cast stripping -----------------------------------------------------


def test_drop_self_cast_in_projection():
    scan = _scan()
    node = Project(scan, (Projection("r", col("x").cast("int64")),))
    out = drop_self_cast_in_projection(node, None)
    assert isinstance(out, Project)
    assert out.items[0].expr.to_ir() == Col("x").to_ir()
    assert drop_self_cast_in_projection(out, None) is None


def test_drop_self_cast_in_projection_negative():
    scan = _scan()
    node = Project(scan, (Projection("r", col("x").cast("float64")),))
    assert drop_self_cast_in_projection(node, None) is None


def test_drop_self_cast_in_filter():
    scan = _scan()
    node = Filter(scan, col("x").cast("int64") > 2)
    out = drop_self_cast_in_filter(node, None)
    assert isinstance(out, Filter)
    assert out.predicate.to_ir() == (col("x") > 2).to_ir()
    assert drop_self_cast_in_filter(out, None) is None


def test_drop_self_cast_in_filter_negative():
    scan = _scan()
    assert drop_self_cast_in_filter(Filter(scan, col("x").cast("float64") > 2.0), None) is None
    assert drop_self_cast_in_filter(Filter(scan, col("x") > 2), None) is None


def test_drop_self_cast_in_sort_key():
    scan = _scan()
    node = Sort(scan, (SortKeySpec(col("x").cast("int64"), descending=True),))
    out = drop_self_cast_in_sort_key(node, None)
    assert isinstance(out, Sort)
    assert out.keys[0].expr.to_ir() == Col("x").to_ir()
    assert out.keys[0].descending is True  # direction preserved
    assert drop_self_cast_in_sort_key(out, None) is None


def test_drop_self_cast_in_sort_key_negative():
    scan = _scan()
    assert (
        drop_self_cast_in_sort_key(Sort(scan, (SortKeySpec(col("x").cast("float64")),)), None)
        is None
    )
    assert drop_self_cast_in_sort_key(Sort(scan, (SortKeySpec(Col("x")),)), None) is None
