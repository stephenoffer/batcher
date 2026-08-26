"""The row-wise tail above a streaming fold is identified once, in the neutral layer.

`split_streaming_tail` is the streaming counterpart of what the distributed dispatcher
does with `_split_at`/`_apply_above`: the fold runs incrementally, and the row-wise
operators *above* it are re-applied to its snapshot. It lives in `plan` because the three
callers — the single-node processor, the distributed eligibility gate, and the distributed
launcher — are in packages that may not import one another, and a second copy is exactly
how the streaming and distributed tails would come to disagree.

These pin the split itself, the `rebuild_over_scan` half that pairs with it, and that
`streaming_fold_target` still means what it always did (a *bare* fold), so a caller that
cannot apply a tail is never handed a plan that has one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.logical import (
    rebuild_over_scan,
    split_streaming_tail,
    streaming_fold_target,
)


@pytest.fixture
def ds():
    return bt.from_pydict({"k": ["a", "a", "b"], "v": [1.0, 2.0, 3.0], "i": [1, 2, 3]})


def _names(nodes):
    return [type(n).__name__ for n in nodes]


@pytest.mark.unit
def test_a_bare_aggregate_has_an_empty_tail(ds):
    tail, agg = split_streaming_tail(ds.group_by("k").agg(s=bt.col("v").sum())._plan)
    assert tail == ()
    assert type(agg).__name__ == "Aggregate"


@pytest.mark.unit
def test_a_projection_above_the_aggregate_is_the_tail(ds):
    plan = ds.group_by("k").agg(s=bt.col("v").sum()).select("s")._plan
    tail, agg = split_streaming_tail(plan)
    assert _names(tail) == ["Project"]
    assert type(agg).__name__ == "Aggregate"


@pytest.mark.unit
def test_an_expression_over_aggregates_lowers_to_a_tail(ds):
    # `sum(v) / count()` is one keyword to the user and a Project over an Aggregate to the
    # engine. That lowering is the reason most composite aggregates could not stream.
    tail, agg = split_streaming_tail(ds.group_by("k").agg(a=bt.col("v").sum() / bt.count())._plan)
    assert _names(tail) == ["Project"]
    assert type(agg).__name__ == "Aggregate"


@pytest.mark.unit
def test_a_having_filter_is_the_tail(ds):
    plan = ds.group_by("k").agg(s=bt.col("v").sum()).filter(bt.col("s") > 1)._plan
    tail, _ = split_streaming_tail(plan)
    assert _names(tail) == ["Filter"]


@pytest.mark.unit
def test_a_multi_node_tail_is_returned_outermost_first(ds):
    plan = ds.group_by("k").agg(s=bt.col("v").sum()).filter(bt.col("s") > 0).select("s")._plan
    tail, _ = split_streaming_tail(plan)
    assert _names(tail) == ["Project", "Filter"]


@pytest.mark.unit
def test_a_whole_column_distinct_still_folds_as_an_aggregate(ds):
    tail, agg = split_streaming_tail(ds.select("k").distinct()._plan)
    assert tail == ()
    assert type(agg).__name__ == "Aggregate"


@pytest.mark.unit
def test_a_distinct_under_a_projection_folds_with_a_tail(ds):
    tail, agg = split_streaming_tail(ds.select("k").distinct().select("k")._plan)
    assert _names(tail) == ["Project"]
    assert type(agg).__name__ == "Aggregate"


@pytest.mark.unit
def test_a_stateless_plan_is_not_a_fold(ds):
    # Walks the tail to the `Scan` and finds no fold, so the router falls through to the
    # stateless processor rather than building an aggregate over nothing.
    assert split_streaming_tail(ds.filter(bt.col("v") > 0).select("k")._plan) is None


@pytest.mark.unit
def test_a_keyed_distinct_is_still_refused(ds):
    assert split_streaming_tail(ds.distinct(subset=["k"])._plan) is None


@pytest.mark.unit
def test_a_breaker_beneath_the_fold_is_still_refused(ds):
    # The fold re-runs its input per micro-batch, so a breaker beneath it would be re-run
    # per batch — the silent wrong answer the predicate exists to prevent.
    plan = ds.distinct().group_by("k").agg(n=bt.count())._plan
    assert split_streaming_tail(plan) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "build",
    [
        lambda d: d.group_by("k").agg(s=bt.col("v").sum()).sort("s"),
        lambda d: d.group_by("k").agg(s=bt.col("v").sum()).limit(1),
    ],
)
def test_sort_and_limit_are_not_tail_nodes(ds, build):
    # Neither is row-wise, and neither has a batch meaning to match on a *running* result.
    assert split_streaming_tail(build(ds)._plan) is None


# --- streaming_fold_target keeps its old contract -------------------------


@pytest.mark.unit
def test_fold_target_still_answers_only_for_a_bare_fold(ds):
    assert streaming_fold_target(ds.group_by("k").agg(s=bt.col("v").sum())._plan) is not None
    assert streaming_fold_target(ds.select("k").distinct()._plan) is not None
    # A tail makes it None, so a caller that cannot apply one is never handed it.
    with_tail = ds.group_by("k").agg(s=bt.col("v").sum()).select("s")._plan
    assert streaming_fold_target(with_tail) is None


# --- rebuild_over_scan ----------------------------------------------------


@pytest.mark.unit
def test_rebuild_over_scan_reroots_the_chain(ds):
    plan = ds.group_by("k").agg(s=bt.col("v").sum()).select("s")._plan
    tail, _ = split_streaming_tail(plan)
    schema = pa.schema([("k", pa.string()), ("s", pa.float64())])
    rebuilt = rebuild_over_scan(tail, schema)
    assert type(rebuilt).__name__ == "Project"
    assert type(rebuilt.input).__name__ == "Scan"


@pytest.mark.unit
def test_rebuild_over_scan_with_an_empty_tail_is_the_bare_scan():
    schema = pa.schema([("s", pa.int64())])
    assert type(rebuild_over_scan((), schema)).__name__ == "Scan"


@pytest.mark.unit
def test_rebuild_over_scan_preserves_nesting_order(ds):
    plan = ds.group_by("k").agg(s=bt.col("v").sum()).filter(bt.col("s") > 0).select("s")._plan
    tail, _ = split_streaming_tail(plan)
    schema = pa.schema([("k", pa.string()), ("s", pa.float64())])
    rebuilt = rebuild_over_scan(tail, schema)
    # Outermost first in, outermost first out: Project(Filter(Scan)).
    assert type(rebuilt).__name__ == "Project"
    assert type(rebuilt.input).__name__ == "Filter"
    assert type(rebuilt.input.input).__name__ == "Scan"
