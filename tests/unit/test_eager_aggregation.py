"""Plan-shape unit tests for `eager_aggregation`."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.agg_pushdown import eager_aggregation
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Aggregate, Join


def _emp():
    # 40 rows over 3 departments so a pushed pre-aggregate (group by dept_id, ndv 3) clears the
    # ~13x reduction the cost guard now requires — the rewrite is only worth its hash table on a
    # large fan-out per key. (A 5-row toy could not express any reduction the guard would accept.)
    dept = [1, 2, 3] * 13 + [1]
    return bt.from_pydict({"dept_id": dept, "sal": list(range(len(dept)))})


def _dept():
    return bt.from_pydict({"dept_id": [1, 2, 3], "name": ["eng", "sales", "ops"]})


def _ctx(ds, ndv=None):
    learned = {"__column_ndv__": ndv} if ndv else {}
    est = StatsEstimator(ds._sources, learned=learned)
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _grouped_max():
    return _emp().join(_dept(), on="dept_id").group_by("name").agg(top=col("sal").max())


def test_rule_registered():
    assert "eager_aggregation" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_pushes_partial_aggregate_below_join():
    ds = _grouped_max()
    out = eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0}))
    assert isinstance(out, Aggregate)
    assert isinstance(out.input, Join)
    assert isinstance(out.input.left, Aggregate)  # the pushed partial aggregate


def test_no_fire_without_reduction():
    # ndv == row count → grouping does not shrink the side → not worth pushing.
    ds = _grouped_max()
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 40.0})) is None


def test_no_fire_on_marginal_reduction():
    # A near-unique key gives only a small reduction (here 40/10 = 4x): building the pre-aggregate's
    # hash table costs more than the join input it shrinks, so the guard must decline. This is the
    # `SUM(...) FROM lineitem JOIN orders` regression in miniature (l_orderkey, ~4 rows/key).
    ds = _grouped_max()
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 10.0})) is None


def test_no_fire_without_stats():
    # No ndv → the estimator can't prove a reduction → conservative no-op.
    ds = _grouped_max()
    assert eager_aggregation(ds._plan, _ctx(ds)) is None


def test_sum_not_pushed():
    # SUM is not fan-out-safe → never pushed.
    ds = _emp().join(_dept(), on="dept_id").group_by("name").agg(total=col("sal").sum())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_right_side_aggregate_not_pushed():
    # The aggregate input is a right column → not a left-side push.
    ds = _emp().join(_dept(), on="dept_id").group_by("dept_id").agg(top=col("name").max())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_outer_join_not_pushed():
    ds = _emp().join(_dept(), on="dept_id", how="left").group_by("name").agg(top=col("sal").max())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 3.0})) is None


def test_idempotent():
    ds = _grouped_max()
    ctx = _ctx(ds, ndv={"dept_id": 3.0})
    once = eager_aggregation(ds._plan, ctx)
    # The pushed side is already reduced; a second push finds no further reduction.
    assert eager_aggregation(once, ctx) is None


def test_a_measured_non_reducing_group_by_vetoes_the_push():
    """`learned_partial_agg` was written on every query and read by nothing.

    `record_group_reduction` records each group-by's real `groups / input_rows`; the reader
    existed but had no production caller, so the measurement never reached a decision. It now
    vetoes pre-aggregation for a group-by a past run measured as barely reducing — the case the
    ratio exists to identify, and where the pushed hash table is pure waste.

    The veto never *licenses* a push: `_reduces_enough` still has to approve it from the
    estimator's `ndv` first, so a stale measurement can only skip a beneficial rewrite, never
    introduce a bad one. Result-invariant either way (the pre-aggregate is an algebraic
    identity).
    """
    from batcher.kyber.learned_tuning import record_group_reduction
    from batcher.kyber.signature import plan_signature
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends.in_process import InProcessBackend

    ds = _grouped_max()
    est = StatsEstimator(ds._sources, learned={"__column_ndv__": {"dept_id": 3.0}})

    # Cold hub: no measurement, so the estimator's ndv decides exactly as before.
    hub = MetadataHub(InProcessBackend())
    ctx = OptimizerContext(config=active_config(), sources=ds._sources, hub=hub, estimator=est)
    assert eager_aggregation(ds._plan, ctx) is not None

    # A past run measured this group-by collapsing 1000 rows into 990 groups — no reduction.
    record_group_reduction(hub, plan_signature(ds._plan), 990.0, 1000.0)
    assert eager_aggregation(ds._plan, ctx) is None

    # A strongly-reducing measurement leaves the push in place.
    hub2 = MetadataHub(InProcessBackend())
    record_group_reduction(hub2, plan_signature(ds._plan), 3.0, 1000.0)
    ctx2 = OptimizerContext(config=active_config(), sources=ds._sources, hub=hub2, estimator=est)
    assert eager_aggregation(ds._plan, ctx2) is not None


def _wide_emp():
    """1,000 rows over 100 departments — a 10x reduction, enough to clear the ratio gate."""
    dept = [i % 100 for i in range(1000)]
    return bt.from_pydict({"dept_id": dept, "sal": list(range(len(dept)))})


def _one_dept():
    """A single department — so the join keeps ~1/100th of the left side."""
    return bt.from_pydict({"dept_id": [7], "name": ["eng"]})


def test_a_more_selective_join_vetoes_the_push():
    """The ratio gate is blind to what the join does with the side it shrank.

    `_reduces_enough` prices the push only against the input it reduces, so a large ratio
    over a huge table sails through even when the join downstream is the far stronger
    reducer. Pre-aggregating in front of one is pure added work: the group-by still reads
    every source row, and the join then emits fewer rows than the group-by produced.

    This is TPC-H Q17 in miniature. There, lineitem pre-aggregated 6,001,215 rows to 201,152
    groups — 29.8x, comfortably past the gate — to feed a join against 195 filtered parts
    whose output was 5,514 rows, and the query went 12 ms -> 242 ms. Here 1,000 rows reduce
    10x to 100 groups to feed a join that emits ~10.
    """
    ds = _wide_emp().join(_one_dept(), on="dept_id").group_by("name").agg(top=col("sal").max())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 100.0})) is None


def test_a_fanning_join_still_pushes():
    """The veto must not fire when the join preserves rows — the classic eager-aggregation win.

    Same reduction as the vetoed case above; the only difference is that every left row finds
    a match, so the join's output is far larger than the pre-aggregate's and the push pays.
    Without this, the guard above would be indistinguishable from disabling the rule.
    """
    dept = bt.from_pydict({"dept_id": list(range(100)), "name": [f"d{i}" for i in range(100)]})
    ds = _wide_emp().join(dept, on="dept_id").group_by("name").agg(top=col("sal").max())
    assert eager_aggregation(ds._plan, _ctx(ds, ndv={"dept_id": 100.0})) is not None
