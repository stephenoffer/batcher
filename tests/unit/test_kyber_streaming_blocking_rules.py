"""Kyber's blocking-operator-avoidance rules for unbounded (streaming) plans.

Plan-shape assertions only; result correctness against DuckDB lives in
`tests/differential/`. Every rule gets three tests, not one:

1. it **fires** on the shape it targets over a stream;
2. it **declines** on the near-miss where firing would change the result — for these
   rules that is the test that protects a correct answer, since each rewrite discards
   an ordering and the near-miss is the case where the ordering is observable;
3. it is a **no-op on a bounded plan**, because the unboundedness gate is a scope
   limit rather than a heuristic, and a rule that quietly fired in batch would be
   changing plans nobody asked it to touch.

The boundedness signal comes from the *bound source* (`ctx.sources`), never a row
estimate, so an unbounded fixture needs a real source with ``bounded=False``. An
`Optimizer` with no sources sees a wholly bounded plan — which is precisely what makes
it the bounded-control fixture here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io import IteratorSource
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.streaming.blocking import _at_most_one_row, _strip_full_sort
from batcher.kyber.streaming import blocking_operators, has_unbounded_input
from batcher.plan.expr_ir import AggExpr, col, lit
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Limit,
    Project,
    Sample,
    Scan,
    Sort,
    Union,
    WatermarkDedup,
)
from batcher.plan.logical.aggregate import AggregateSpec, SortKeySpec
from batcher.plan.logical.relational import Projection
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("k", pa.int64()), ("v", pa.int64()), ("t", pa.int64())])

# One ascending sort key over `k`, the ordering every test below asks about.
KEYS = (SortKeySpec(col("k")),)


def _scan() -> Scan:
    """A scan of source 0, whose boundedness the optimizer fixture decides."""
    return Scan(0, SchemaRef.from_arrow(_SCHEMA))


def _stream() -> Optimizer:
    """An optimizer whose one source is a declared-unbounded stream."""
    source = IteratorSource(lambda: iter([]), _SCHEMA, bounded=False)
    return Optimizer(None, [source], None)


def _batch() -> Optimizer:
    """An optimizer whose one source is an ordinary bounded relation."""
    source = IteratorSource(lambda: iter([]), _SCHEMA, bounded=True)
    return Optimizer(None, [source], None)


def _shape(plan) -> list[str]:
    return [type(n).__name__ for n in walk(plan)]


def _global_agg(inner=None) -> Aggregate:
    """A keyless (global) aggregate — exactly one output row, by construction."""
    return Aggregate(
        inner if inner is not None else _scan(),
        (),
        (AggregateSpec("c", AggExpr("count_star", None)),),
    )


# --- the fixtures themselves ---------------------------------------------------


def test_the_stream_fixture_is_actually_unbounded():
    """Pin the fixture: everything below is vacuous if this signal does not work.

    `has_unbounded_input` reads the bound source, not a cardinality estimate, so a
    fixture that forgot `bounded=False` would silently make every "fires" test below
    assert nothing at all.
    """
    assert has_unbounded_input(_scan(), _stream()._context())
    assert not has_unbounded_input(_scan(), _batch()._context())
    assert not has_unbounded_input(_scan(), Optimizer(None, [], None)._context())


# --- stream_drop_keyless_sort --------------------------------------------------


def test_keyless_sort_over_a_stream_is_dropped():
    """A sort with no keys orders nothing, yet blocks the stream forever."""
    plan = Sort(_scan(), keys=(), limit=None)
    assert blocking_operators(plan), "the fixture must start out blocking"
    assert _shape(_stream().logical_rewrite(plan)) == ["Scan"]


def test_keyless_topn_over_a_stream_is_kept():
    """A keyless top-N selects rows; which ones depends on a stability guarantee.

    Rewriting it away (to a `Limit`, say) would assert that the engine's partial sort
    is stable with no keys to compare. It is also not blocking, so there is no hang to
    justify the risk.
    """
    plan = Sort(_scan(), keys=(), limit=5)
    assert isinstance(_stream().logical_rewrite(plan), Sort)


def test_keyed_sort_over_a_stream_is_kept():
    """A sort with real keys imposes a real ordering — it is not this rule's business."""
    plan = Sort(_scan(), KEYS)
    assert isinstance(_stream().logical_rewrite(plan), Sort)


def test_keyless_sort_is_untouched_on_a_bounded_plan():
    plan = Sort(_scan(), keys=(), limit=None)
    assert isinstance(_batch().logical_rewrite(plan), Sort)


# --- stream_drop_sort_under_distinct -------------------------------------------


def test_sort_under_distinct_over_a_stream_is_dropped():
    """`Distinct` is a set operation, so its input's order is not observable."""
    plan = Distinct(Sort(_scan(), KEYS))
    assert _shape(_stream().logical_rewrite(plan)) == ["Distinct", "Scan"]


def test_sort_under_distinct_is_reached_through_project_and_filter():
    """`Project`/`Filter` are row-wise, so they cannot make the ordering observable."""
    inner = Filter(Sort(_scan(), KEYS), col("v") > lit(1))
    plan = Distinct(Project(inner, (Projection("k", col("k")),)))
    shape = _shape(_stream().logical_rewrite(plan))
    assert "Sort" not in shape, shape
    assert shape[0] == "Distinct", shape


def test_topn_under_distinct_is_kept():
    """A top-N sort selects *which* rows exist; dropping it changes the result set."""
    plan = Distinct(Sort(_scan(), KEYS, limit=3))
    assert "Sort" in _shape(_stream().logical_rewrite(plan))


def test_sort_under_distinct_is_not_reached_through_a_limit():
    """`Limit` takes a positional prefix, so which rows it keeps depends on the order.

    Removing the sort beneath it would change *which* rows reach the `Distinct` — a
    different result, not merely a different order. This is the near-miss that makes
    `_strip_full_sort`'s descent list a proof obligation rather than a convenience.
    """
    plan = Distinct(Limit(Sort(_scan(), KEYS), 10))
    assert "Sort" in _shape(_stream().logical_rewrite(plan))


def test_sort_under_watermark_dedup_is_never_dropped():
    """A dedup keeps the *first* row per key and advances a watermark from arrival order.

    Reordering its input therefore changes which row survives per key and which rows are
    discarded as late. No rule in this family may strip that sort, and this pins it.
    """
    plan = WatermarkDedup(Sort(_scan(), KEYS), subset=("k",), event_time="t", lateness_micros=1_000)
    assert "Sort" in _shape(_stream().logical_rewrite(plan))


def test_sort_under_distinct_is_untouched_on_a_bounded_plan():
    plan = Distinct(Sort(_scan(), KEYS))
    assert "Sort" in _shape(_batch().logical_rewrite(plan))


# --- stream_drop_sort_in_distinct_union_branch ---------------------------------


def _two_branches():
    """Two genuinely different branches, so branch-dedup rules cannot collapse them."""
    left = Sort(Filter(_scan(), col("v") > lit(1)), KEYS)
    right = Filter(_scan(), col("v") > lit(2))
    return left, right


def test_sort_in_a_distinct_union_branch_over_a_stream_is_dropped():
    """A distinct union dedupes the concatenation, so branch order is not observable."""
    left, right = _two_branches()
    out = _stream().logical_rewrite(Union((left, right), distinct=True))
    assert "Sort" not in _shape(out), _shape(out)


def test_sort_in_a_union_all_branch_is_kept():
    """`UNION ALL` concatenates, so a branch's internal ordering survives into the result.

    This is the whole gate for the rule, and the reason it is separate from the
    `Distinct` one rather than a case of it.
    """
    left, right = _two_branches()
    out = _stream().logical_rewrite(Union((left, right), distinct=False))
    assert "Sort" in _shape(out), _shape(out)


def test_distinct_union_with_no_sorted_branch_is_unchanged():
    """No branch has a removable sort → the rule returns None and stays idempotent."""
    right = Filter(_scan(), col("v") > lit(2))
    plan = Union((Filter(_scan(), col("v") > lit(1)), right), distinct=True)
    assert "Sort" not in _shape(_stream().logical_rewrite(plan))


def test_sort_in_a_distinct_union_branch_is_untouched_on_a_bounded_plan():
    left, right = _two_branches()
    out = _batch().logical_rewrite(Union((left, right), distinct=True))
    assert "Sort" in _shape(out), _shape(out)


# --- stream_drop_distinct_over_at_most_one_row ---------------------------------


@pytest.mark.parametrize(
    "inner",
    [
        Limit(_scan(), 1),
        Sort(_scan(), KEYS, limit=1),
        Sample(_scan(), fraction=1.0, seed=7, n=1),
        _global_agg(),
    ],
    ids=["limit_1", "top_1", "sample_1", "global_aggregate"],
)
def test_distinct_over_an_at_most_one_row_input_is_dropped(inner):
    """A relation of ≤ 1 row cannot hold a duplicate pair, so the dedup is an identity."""
    assert "Distinct" not in _shape(_stream().logical_rewrite(Distinct(inner)))


@pytest.mark.parametrize(
    "inner",
    [
        Limit(_scan(), 2),
        Sort(_scan(), KEYS, limit=2),
        Sample(_scan(), fraction=0.5, seed=7),
    ],
    ids=["limit_2", "top_2", "fraction_sample"],
)
def test_distinct_over_a_multi_row_input_is_kept(inner):
    """Two rows may be equal, so the dedup is load-bearing — the near-miss by one row."""
    assert "Distinct" in _shape(_stream().logical_rewrite(Distinct(inner)))


def test_distinct_over_a_grouped_aggregate_is_bounded_by_groups_not_by_one_row():
    """A *grouped* aggregate has a row per group; only a keyless one is a single row.

    `remove_redundant_distinct` drops this `Distinct` for its own (correct) reason —
    an aggregate's group keys are in its output — so the assertion here is on
    `_at_most_one_row`, which must not be the thing that claims it.
    """
    grouped = Aggregate(
        _scan(),
        (Projection("k", col("k")),),
        (AggregateSpec("c", AggExpr("count_star", None)),),
    )
    assert not _at_most_one_row(grouped)
    assert _at_most_one_row(_global_agg())


def test_distinct_over_one_row_is_untouched_on_a_bounded_plan():
    """The bounded path has `remove_redundant_distinct`'s exact-stats proof; not ours.

    An `IteratorSource` reports no exact row count, so no stats-gated rule fires here
    either — which makes this a clean read of *this* rule's gate.
    """
    plan = Distinct(Limit(_scan(), 1))
    assert "Distinct" in _shape(_batch().logical_rewrite(plan))


# --- stream_drop_sort_over_at_most_one_row -------------------------------------


@pytest.mark.parametrize(
    "inner",
    [Limit(_scan(), 1), Sort(_scan(), KEYS, limit=1)],
    ids=["limit_1", "top_1"],
)
def test_sort_over_an_at_most_one_row_input_is_dropped(inner):
    """A ≤ 1-row relation is already ordered under every key set and direction."""
    out = _stream().logical_rewrite(Sort(inner, KEYS))
    assert _shape(out) == _shape(inner), _shape(out)


def test_sort_over_a_global_aggregate_is_dropped():
    """A keyless aggregate is exactly one row, so ordering it is a no-op.

    Sorted on the aggregate's own output column, since a global aggregate projects
    away every input column — the reason this case cannot ride the parametrization
    above.
    """
    inner = _global_agg()
    out = _stream().logical_rewrite(Sort(inner, (SortKeySpec(col("c")),)))
    assert _shape(out) == _shape(inner), _shape(out)


def test_sort_over_a_multi_row_input_is_kept():
    """Two rows can be out of order, so the sort does real work."""
    plan = Sort(Limit(_scan(), 2), KEYS)
    assert isinstance(_stream().logical_rewrite(plan), Sort)


def test_topn_of_zero_over_one_row_is_kept():
    """`limit == 0` yields the empty relation — a real change, not an identity.

    Refused here and left to the empty-propagation rules, which model emptiness
    properly instead of silently returning a one-row input for a zero-row request.
    """
    plan = Sort(Limit(_scan(), 1), KEYS, limit=0)
    out = _stream().logical_rewrite(plan)
    # This rule declines; `empty_topn_to_empty` then models the top-0 as a `Limit(0)`.
    # What must NOT happen is the one-row input coming back for a zero-row request.
    assert isinstance(out, Limit) and out.n == 0, _shape(out)
    assert _shape(out) != ["Limit", "Scan"], _shape(out)


def test_topn_of_one_over_one_row_is_dropped():
    """`limit >= 1` cannot remove the single row, so the sort is still an identity."""
    plan = Sort(Limit(_scan(), 1), KEYS, limit=1)
    assert _shape(_stream().logical_rewrite(plan)) == ["Limit", "Scan"]


def test_sort_over_one_row_is_untouched_on_a_bounded_plan():
    plan = Sort(Limit(_scan(), 1), KEYS)
    assert isinstance(_batch().logical_rewrite(plan), Sort)


# --- the structural helpers ----------------------------------------------------


def test_strip_full_sort_refuses_the_operators_it_cannot_cross():
    """The descent list is a proof obligation: everything not proven transparent stops it.

    Each of these either selects rows by position/relative order or combines rows across
    the stream, so reordering its input changes which rows come out.
    """
    sorted_scan = Sort(_scan(), KEYS)
    assert _strip_full_sort(Limit(sorted_scan, 5)) is None
    assert _strip_full_sort(Distinct(sorted_scan)) is None
    assert _strip_full_sort(Sample(sorted_scan, fraction=0.5, seed=1)) is None
    assert _strip_full_sort(_global_agg(sorted_scan)) is None
    assert _strip_full_sort(_scan()) is None


def test_strip_full_sort_refuses_a_topn():
    """A top-N is a row filter, not an ordering — and it does not block a stream."""
    assert _strip_full_sort(Sort(_scan(), KEYS, limit=5)) is None
    assert _strip_full_sort(Sort(_scan(), KEYS)) is not None


def test_at_most_one_row_is_structural_and_needs_no_statistics():
    """The bound reads plan shape only, which is why it works where exact stats cannot."""
    assert _at_most_one_row(Limit(_scan(), 0))
    assert _at_most_one_row(Limit(_scan(), 1))
    assert not _at_most_one_row(Limit(_scan(), 2))
    assert not _at_most_one_row(Sort(_scan(), KEYS))
    assert not _at_most_one_row(_scan())
    assert not _at_most_one_row(Filter(Limit(_scan(), 1), col("v") > lit(0)))
