"""The distributed staging decision must be taken from the plan Kyber produced.

`resolve_adaptive` asks `requires_staging` of the plan the *caller wrote*; the distributed
executor is handed the plan the *optimizer produced*. Those are not the same question, and
one rewrite makes them disagree: eager aggregation pre-reduces a fact table on the join key,
turning `Aggregate(Join(fact, dim))` into `Aggregate(Join(dim, Aggregate(fact)))` — a breaker
beneath a join, which the one-shot dispatcher has no path for.

The rewrite fires on *measured* statistics, so the disagreement appears only once the query
has run: a `join -> group_by -> agg` over parquet distributed fine on its first run and raised
`PlanError` on every run after it. Nothing in the suites could see that, because a
differential test runs each query once.

These tests pin the two halves separately, so a failure says which one broke:

1. the *premise* — that the shape really does flip `requires_staging` — asserted on plans
   built here rather than taken on trust, since a test whose premise has quietly stopped
   holding passes while checking nothing;
2. the *routing* — that `_stage_if_optimization_requires_it` sends exactly that case to the
   staged executor, leaves the ordinary case alone, and never fires inside the staged loop
   itself (`materialize=False`), which is what would make staging re-enter itself.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col

pytest.importorskip("ray", reason="the staging predicate lives behind the optional ray extra")

from batcher.api.orchestration.stages import _stage_if_optimization_requires_it
from batcher.dist import requires_staging

pytestmark = pytest.mark.unit


def _plans():
    """`(raw, rewritten)` — the shape before and after eager aggregation.

    Built through the public API so the nodes are the ones the optimizer really emits: the
    rewritten form is spelled as a pre-aggregated fact side joined to the dimension, which is
    exactly what the rule produces.
    """
    fact = bt.from_pydict({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    dim = bt.from_pydict({"k": [1, 2], "p": ["a", "b"]})
    raw = fact.join(dim, on="k").group_by("p").agg(t=col("v").sum())._plan
    pre = fact.group_by("k").agg(t=col("v").sum())
    rewritten = pre.join(dim, on="k").group_by("p").agg(t=col("t").sum())._plan
    return raw, rewritten


def test_eager_aggregation_is_what_flips_the_staging_answer():
    """The premise: the same query needs no staging before the rewrite and needs it after."""
    raw, rewritten = _plans()
    assert requires_staging(raw) is False
    assert requires_staging(rewritten) is True


class _Ctx:
    """The two fields the routing helper reads off an `ExecutionContext`."""

    hub = None
    num_workers = None
    transport = "auto"


def test_the_rewritten_plan_is_routed_to_the_staged_executor(monkeypatch):
    raw, rewritten = _plans()
    sentinel = object()
    called: list = []

    def fake(plan, sources, hub, **kw):
        called.append((plan, kw))

        class R:
            table = sentinel

        return R()

    monkeypatch.setattr("batcher.api.adaptive.execute_adaptive", fake)
    out = _stage_if_optimization_requires_it(raw, rewritten, [], _Ctx(), materialize=True)
    assert out is sentinel
    # The staged loop re-optimizes from the raw plan, exactly as the ordinary adaptive route
    # does — handing it the already-rewritten one would stage a plan the loop did not choose.
    assert called and called[0][0] is raw
    assert called[0][1]["distributed"] is True


def test_a_plan_the_dispatcher_can_route_is_left_alone(monkeypatch):
    """The common case must not pay for this check, and must not be diverted by it."""
    raw, _ = _plans()

    def fail(*a, **k):  # pragma: no cover - reaching this is the failure
        raise AssertionError("a one-shot-routable plan must not be staged")

    monkeypatch.setattr("batcher.api.adaptive.execute_adaptive", fail)
    assert _stage_if_optimization_requires_it(raw, raw, [], _Ctx(), materialize=True) is None


def test_it_stops_firing_once_the_nesting_budget_is_spent(monkeypatch):
    """Re-entry is bounded, and at the bound the loop stops re-entering itself.

    This is the guard that matters, and `materialize` is not it. Every stage runs through
    `run_relational` and comes back through this function, and the loop's **final** stage is
    the whole residual plan with `materialize=True` — so without a bound a plan the loop
    could not cut re-entered the loop on itself and raised `RecursionError` where the caller
    should have seen a `PlanError`. Caught by `test_diff_exists_mixed_correlation`, not by
    this file's first draft, which asserted the `materialize=False` case and called the
    nesting impossible.

    The bound replaced a flat "never inside the loop", which also refused the one re-entry
    that makes progress — see the test below. What has to stay true is that the budget is
    finite, so an optimizer rewrite that never converges ends in a `PlanError` rather than a
    `RecursionError`.
    """
    raw, rewritten = _plans()

    def fail(*a, **k):  # pragma: no cover - reaching this is the failure
        raise AssertionError("a stage must not re-enter the staged executor past the bound")

    monkeypatch.setattr("batcher.api.adaptive.execute_adaptive", fail)
    monkeypatch.setattr("batcher.api.adaptive.staging.staged_depth_exhausted", lambda: True)
    assert _stage_if_optimization_requires_it(raw, rewritten, [], _Ctx(), materialize=True) is None


def test_one_re_entry_is_allowed_for_an_optimizer_introduced_breaker(monkeypatch):
    """A stage whose *rewrite* needs staging must still be able to stage, once.

    The loop chooses its cuts from the plan the caller wrote; Kyber re-optimizes each stage
    afterwards, and eager aggregation can put a breaker beneath a join in a stage the loop
    had already decided needed no cut. Forbidding every re-entry made that unroutable: a
    four-table star join with an aggregate over it raised `PlanError` on the disk transport
    (`test_distributed_multi_table_join_matches_single_node[disk]`) while the identical query
    ran on Flight.
    """
    raw, rewritten = _plans()
    sentinel = object()
    called = []

    class R:
        table = sentinel

    def fake(*a, **k):
        called.append((a, k))
        return R()

    monkeypatch.setattr("batcher.api.adaptive.execute_adaptive", fake)
    monkeypatch.setattr("batcher.api.adaptive.staging.staged_depth_exhausted", lambda: False)
    out = _stage_if_optimization_requires_it(raw, rewritten, [], _Ctx(), materialize=True)
    assert out is sentinel
    assert called and called[0][0][0] is raw
    # `force_structural` is what makes the re-entry deterministic. The loop is handed the
    # *raw* plan, whose `requires_staging` is False — it is `rewritten` that has no one-shot
    # path — so without this the loop would fall through to `_worth_staging`, which reads the
    # hub. The query would then run or raise depending on what earlier runs had learned,
    # which is precisely the cross-run instability this gate exists to remove.
    assert called[0][1]["force_structural"] is True


def test_an_intermediate_stage_is_left_alone_too(monkeypatch):
    """`materialize=False` is a partitioned intermediate; it has no table to hand back."""
    raw, rewritten = _plans()

    def fail(*a, **k):  # pragma: no cover - reaching this is the failure
        raise AssertionError("an unmaterialized stage must not be staged")

    monkeypatch.setattr("batcher.api.adaptive.execute_adaptive", fail)
    assert _stage_if_optimization_requires_it(raw, rewritten, [], _Ctx(), materialize=False) is None
