"""Kyber's decisions about a keyed dedup — what it may rewrite, and what it must not.

A keyed dedup (`distinct(subset=…)`) looks like a `Distinct` and is not one. The whole-row
form is a *set* operation: it removes rows that duplicate another row exactly, so any rewrite
justified by "the dedup erases multiplicities" is sound. The keyed form removes rows that
differ from the survivor in every column outside the key, so those same rewrites change the
answer rather than the cost.

Every rule below reads a `Distinct` and once treated the two as the same thing. Each test
pairs the rewrite that must still fire on the whole-row form with the one that must not fire
on the keyed form — a one-sided test would pass against a rule that had simply stopped
working. The wrong-answer direction is the point: none of these produce an error, they
produce a plausible relation with the wrong rows in it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import Config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.algebraic.identities import (
    collapse_full_key_distinct,
    remove_redundant_distinct,
)
from batcher.kyber.rules.extra.agg_extra import drop_distinct_before_agg
from batcher.kyber.rules.extra.setops import push_filter_through_distinct
from batcher.kyber.rules.joins.rewrites import drop_redundant_distinct_build, join_to_semijoin
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Distinct,
    Filter,
    Join,
    JoinOutputCol,
    Project,
    Projection,
    Scan,
    SortKeySpec,
)
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = SchemaRef.from_arrow(
    pa.schema([("k", pa.int64()), ("v", pa.int64()), ("ts", pa.int64())])
)


def _scan() -> Scan:
    return Scan(0, _SCHEMA)


def _keyed(inp=None) -> Distinct:
    """`distinct(["k"], keep="first", order_by="ts")` over a scan."""
    return Distinct(inp or _scan(), ("k",), (SortKeySpec(Col("ts")),))


def _ctx() -> OptimizerContext:
    """A context with no learned statistics — every rule here is structural, not stats-gated,
    so an estimator that knows nothing is the right one to assert against."""
    return OptimizerContext(
        config=Config(), sources=[], hub=None, estimator=CardinalityEstimator([], None)
    )


def test_a_keyed_dedup_is_not_a_group_by():
    """`as_aggregate` is the derivation three subsystems reuse to distribute a whole-row
    DISTINCT. Handing it a keyed one would build a group-by over every column — a *whole-row*
    dedup — so it refuses rather than returning something that runs and is wrong."""
    from batcher._internal.errors import PlanError

    assert Distinct(_scan()).as_aggregate().group_keys  # the whole-row form still derives
    with pytest.raises(PlanError, match="not a group-by"):
        _keyed().as_aggregate()


def test_an_ordering_without_a_key_is_refused():
    """Rows that agree on every column are indistinguishable, so an ordering over a whole-row
    dedup could not choose anything. Accepting it would silently ignore the argument."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="no payload to order"):
        Distinct(_scan(), (), (SortKeySpec(Col("ts")),))


def test_redundant_distinct_removal_respects_the_key():
    """`Distinct(Aggregate)` is redundant only for the whole-row form.

    An aggregate emits one row per group key, so its rows are already distinct — but they are
    not distinct on an arbitrary *subset* of its columns, which is exactly what a keyed dedup
    above it collapses further.
    """
    agg = Aggregate(
        _scan(),
        (Projection("k", Col("k")), Projection("v", Col("v"))),
        (AggregateSpec("s", Col("ts").sum()),),
    )
    assert remove_redundant_distinct(Distinct(agg), _ctx()) is agg
    assert remove_redundant_distinct(Distinct(agg, ("k",)), _ctx()) is None


def test_an_identical_keyed_dedup_is_still_idempotent():
    """Same key, same ordering: the inner one already chose the survivor."""
    inner = _keyed()
    outer = Distinct(inner, inner.keys, inner.order)
    assert remove_redundant_distinct(outer, _ctx()) is inner
    # A *different* key is not idempotent — the outer one collapses further.
    other = Distinct(inner, ("v",))
    assert remove_redundant_distinct(other, _ctx()) is None


def test_a_dedup_keyed_on_every_column_collapses_to_the_whole_row_form():
    """With no payload the two forms return the same relation, and the whole-row one is the
    cheaper operator (a presence bitmap or a single-pass hash, and no gather by index)."""
    every = Distinct(_scan(), ("k", "v", "ts"))
    out = collapse_full_key_distinct(every, _ctx())
    assert isinstance(out, Distinct) and not out.keys
    # A proper subset is left alone: it still has a payload to carry.
    assert collapse_full_key_distinct(Distinct(_scan(), ("k", "v")), _ctx()) is None


def test_a_filter_does_not_commute_with_a_keyed_dedup():
    """Filtering first changes which rows are *available* to be chosen as the survivor."""
    whole = Filter(Distinct(_scan()), Col("v") > 5)
    assert isinstance(push_filter_through_distinct(whole, _ctx()), Distinct)
    keyed = Filter(_keyed(), Col("v") > 5)
    assert push_filter_through_distinct(keyed, _ctx()) is None


def test_a_keyed_dedup_beneath_a_duplicate_insensitive_aggregate_stays():
    """`min`/`max` are unchanged by removing *identical* rows, which is all a whole-row dedup
    removes. A keyed dedup removes rows that differ, so the values reaching each group change."""
    spec = (AggregateSpec("lo", Col("v").min()),)
    keys = (Projection("k", Col("k")),)
    whole = Aggregate(Distinct(_scan()), keys, spec)
    assert isinstance(drop_distinct_before_agg(whole, _ctx()), Aggregate)
    keyed = Aggregate(_keyed(), keys, spec)
    assert drop_distinct_before_agg(keyed, _ctx()) is None


def test_a_keyed_dedup_only_reads_the_columns_it_needs():
    """A keyed dedup carries its payload but does not *read* payload nothing consumes.

    The whole-row form needs every column — dropping one changes which rows are duplicates —
    and that rule used to apply to both, so `distinct(["user_id"]).select("user_id")` over a
    200-column table read all 200. Only the key, the ordering, and what the plan above consumes
    are needed now.

    Asserted on `source_projections`, the thing that decides what comes off disk, because the
    operator-tree rewrite and the per-source analysis are two separate walks: fixing one and
    not the other prunes the payload out of the plan while still paying to read it.
    """
    from batcher.config import active_config
    from batcher.kyber import optimize

    t = pa.table({"k": [1, 1, 2], "a": [1, 2, 3], "b": ["x", "y", "z"]})

    def reads(ds):
        return optimize(ds._plan, active_config(), ds._sources).source_projections[0]

    base = bt.from_arrow(t)
    assert reads(base.distinct(["k"]).select("k")) == ["k"]
    assert reads(base.distinct(["k"]).select("k", "a")) == ["k", "a"]
    # The ordering column is read even when nothing above consumes it: it decides the survivor.
    assert reads(base.distinct(["k"], keep="first", order_by="b").select("k")) == ["k", "b"]
    # A whole-row dedup still needs everything.
    assert reads(base.distinct().select("k")) == ["k", "a", "b"]


def _semi_join(right) -> Join:
    return Join(
        _scan(),
        right,
        ("k",),
        ("k",),
        "semi",
        (JoinOutputCol("left", "k", "k"), JoinOutputCol("left", "v", "v")),
    )


def test_a_keyed_build_side_dedup_is_not_droppable():
    """A semi join is insensitive to duplicate build rows, not to *missing* ones — and a keyed
    dedup on the build side can drop the only row carrying a given join key."""
    whole = _semi_join(Distinct(_scan()))
    assert isinstance(drop_redundant_distinct_build(whole, _ctx()), Join)
    keyed = _semi_join(Distinct(_scan(), ("v",)))
    assert drop_redundant_distinct_build(keyed, _ctx()) is None


def test_join_to_semijoin_only_fires_for_a_whole_row_dedup():
    """Reducing an inner join to a semi join removes the fan-out. A whole-row dedup above it
    collapses that fan-out anyway; a keyed one chooses a survivor *from* it."""
    join = Join(
        _scan(),
        _scan(),
        ("k",),
        ("k",),
        "inner",
        (JoinOutputCol("left", "k", "k"), JoinOutputCol("right", "v", "rv")),
    )
    proj = Project(join, (Projection("k", Col("k")),))
    assert isinstance(join_to_semijoin(Distinct(proj), _ctx()), Distinct)
    assert join_to_semijoin(Distinct(proj, ("k",)), _ctx()) is None
