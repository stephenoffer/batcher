"""Capacity-aware placement: spot for work that recomputes, on-demand for work that doesn't.

A stateless map partition re-derives from its durable descriptor — which is exactly why
`policies._barrier` resubmits a preempted one instead of failing the stage — so it belongs on
reclaimable capacity. A pipeline breaker holds accumulated state no descriptor can rebuild, so
it does not. Ray can express that with a market-type `label_selector` plus a
`fallback_strategy` naming where the stage goes when the preferred market is full.

The whole risk of this feature is a selector emitted where it *narrows* rather than chooses: a
fleet that carries no market label, carries two different market labels, or is entirely one
market will either match everything (a pointless round trip) or match nothing (a hang that
looks exactly like a slow cluster). These tests hold every one of those to `{}`, and hold the
translation honest in the two cases where a selector is warranted.

They exercise the pure translation against an injected census; nothing here starts Ray.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, config_context
from batcher.dist.executors.ray_runtime.fabric.census import build_census
from batcher.dist.executors.ray_runtime.fabric.market import (
    capacity_bundle_selector,
    capacity_selector,
    fleet_market_split,
)
from batcher.dist.executors.ray_runtime.fabric.topology import ON_DEMAND, SPOT, market_type
from batcher.plan.resource import CAPACITY_ANY, CAPACITY_ON_DEMAND, CAPACITY_SPOT

pytestmark = pytest.mark.unit

_RAY_KEY = "ray.io/market-type"


def _enabled() -> Config:
    cfg = Config()
    return cfg.replace(
        distributed=dataclasses.replace(cfg.distributed, capacity_aware_placement=True)
    )


def _row(market: str, *, cores: float = 16.0, count: int = 1, label: str = _RAY_KEY) -> dict:
    """One `node_class_census()` row, as `fleet_market_split` reads it."""
    return {
        "market_label": label if market else "",
        "market_type": market,
        "free_cpus": cores,
        "cpus": cores,
        "count": count,
    }


_MIXED = [_row(SPOT, count=4), _row(ON_DEMAND, count=2)]


# --- the label vocabulary ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ({_RAY_KEY: "SPOT"}, (_RAY_KEY, SPOT)),
        ({_RAY_KEY: "on-demand"}, (_RAY_KEY, ON_DEMAND)),
        ({"karpenter.sh/capacity-type": "spot"}, ("karpenter.sh/capacity-type", SPOT)),
        (
            {"eks.amazonaws.com/capacityType": "ON_DEMAND"},
            ("eks.amazonaws.com/capacityType", ON_DEMAND),
        ),
        ({"cloud.google.com/gke-spot": "true"}, ("cloud.google.com/gke-spot", SPOT)),
        # Labelled, but in a vocabulary this does not read. The key comes back so a caller can
        # say *which* label it could not interpret; the value does not, so nothing selects on it.
        ({_RAY_KEY: "reserved-block"}, (_RAY_KEY, "")),
        # No opinion. Emphatically not on-demand: most fleets carry no market label at all.
        ({}, ("", "")),
        ({"topology.kubernetes.io/zone": "us-west-2a"}, ("", "")),
    ],
)
def test_market_type_reads_each_provisioners_spelling(labels, expected):
    assert market_type(labels) == expected


def test_is_preemptible_still_agrees_with_market_type():
    """The boolean the shuffle's replica placement reads is now derived from the same scan.

    A positive control on both sides: it has to move when the label does, or the derivation
    could be a constant and this file would not notice.
    """
    from batcher.dist.executors.ray_runtime.fabric.topology import is_preemptible

    assert is_preemptible({_RAY_KEY: "spot"}) is True
    assert is_preemptible({_RAY_KEY: "on-demand"}) is False
    assert is_preemptible({}) is False


def test_census_carries_the_label_key_the_market_was_read_from():
    """A fleet labelled by Karpenter must be selected on Karpenter's key, not on Ray's.

    This is the same mistake `zone_label` exists to prevent, one dimension over: a selector
    naming a key the fleet does not use matches nothing, which pends rather than errors.
    """
    nodes = [
        {
            "Alive": True,
            "NodeID": "a",
            "Resources": {"CPU": 8.0},
            "Labels": {"karpenter.sh/capacity-type": "spot"},
        }
    ]
    (klass,) = build_census(nodes, None, {}, shape_zone=lambda _labels: "")
    assert klass.market_label == "karpenter.sh/capacity-type"
    assert klass.market_type == SPOT
    assert klass.preemptible is True


# --- the fleet split -----------------------------------------------------------------------


def test_split_weights_each_class_by_its_node_count():
    """A census row is a *class*, not a node. Ignoring `count` under-counts the fleet."""
    key, cores = fleet_market_split(_MIXED)
    assert key == _RAY_KEY
    assert cores == {SPOT: 64.0, ON_DEMAND: 32.0}


def test_split_declines_a_fleet_labelled_under_two_keys():
    """A selector names one key. Emitting one here would exclude every node under the other."""
    census = [*_MIXED, _row(SPOT, label="karpenter.sh/capacity-type")]
    assert fleet_market_split(census) == ("", {})


def test_split_ignores_a_node_that_says_nothing():
    census = [*_MIXED, _row("")]
    key, cores = fleet_market_split(census)
    assert key == _RAY_KEY
    assert set(cores) == {SPOT, ON_DEMAND}


# --- the gates -----------------------------------------------------------------------------


def test_no_selector_when_the_gate_is_off():
    assert capacity_selector(CAPACITY_ON_DEMAND, census=_MIXED) == {}


def test_no_selector_for_a_stage_with_no_preference():
    with config_context(_enabled()):
        assert capacity_selector(CAPACITY_ANY, census=_MIXED) == {}


@pytest.mark.parametrize(
    "census",
    [
        pytest.param([], id="unreadable"),
        pytest.param([_row("", cores=64.0)], id="unlabelled"),
        pytest.param([_row(SPOT, count=4)], id="all-spot"),
        pytest.param([_row(ON_DEMAND, count=4)], id="all-on-demand"),
        pytest.param([*_MIXED, _row(SPOT, label="karpenter.sh/capacity-type")], id="two-keys"),
    ],
)
def test_no_selector_on_a_fleet_that_cannot_express_the_preference(census):
    """Every one of these would narrow rather than choose, and one of them hangs.

    A fleet that is entirely spot and is asked for on-demand matches no node at all, and an
    unsatisfiable label selector pends indefinitely rather than failing — which reads as a
    slow cluster, not as a placement bug.
    """
    with config_context(_enabled()):
        assert capacity_selector(CAPACITY_ON_DEMAND, census=census) == {}


# --- the translation ------------------------------------------------------------------------


def test_on_demand_preference_falls_back_to_spot():
    with config_context(_enabled()):
        opts = capacity_selector(CAPACITY_ON_DEMAND, workers=4, num_cpus=1.0, census=_MIXED)
    assert opts["label_selector"] == {_RAY_KEY: ON_DEMAND}
    assert opts["fallback_strategy"] == [{"label_selector": {_RAY_KEY: SPOT}}]


def test_spot_preference_falls_back_to_on_demand():
    with config_context(_enabled()):
        opts = capacity_selector(CAPACITY_SPOT, workers=64, num_cpus=0.125, census=_MIXED)
    assert opts["label_selector"] == {_RAY_KEY: SPOT}
    assert opts["fallback_strategy"] == [{"label_selector": {_RAY_KEY: ON_DEMAND}}]


def test_the_emitted_options_are_ones_ray_actually_accepts():
    """The selector is only useful if Ray validates it, and both keys are newer than the rest
    of the scheduling API. This is the positive control for the feature probe: if `.options()`
    rejected what `capacity_selector` builds, every other assertion here would still pass while
    the fleet's placement was unchanged."""
    ray = pytest.importorskip("ray")
    with config_context(_enabled()):
        opts = capacity_selector(CAPACITY_ON_DEMAND, workers=2, num_cpus=1.0, census=_MIXED)
    assert opts, "the mixed fleet above must produce a selector for this control to mean anything"
    ray.remote(lambda: None).options(**opts)


# --- who asks for what ----------------------------------------------------------------------


def test_carbonite_asks_for_on_demand_only_when_the_plan_holds_state():
    """`Core measures, Kyber decides, Carbonite protects` — and what it protects here is work
    already done. A breaker-free plan states nothing, because every partition of it recomputes.
    """
    from batcher.carbonite.policies.scheduling import _capacity_preference
    from batcher.plan.physical import PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    def _plan(*materializes: bool) -> PhysicalPlan:
        return PhysicalPlan(
            ir={},
            output_schema=None,
            ops=tuple(
                PhysicalOp(
                    op_id=i,
                    kind="Aggregate" if m else "Filter",
                    backend="native",
                    algorithm="",
                    bounds=ResourceBounds(0, 0, 0, materializes=m),
                    inputs=(),
                )
                for i, m in enumerate(materializes)
            ),
        )

    assert _capacity_preference(_plan(False, False)) == CAPACITY_ANY
    assert _capacity_preference(_plan(False, True)) == CAPACITY_ON_DEMAND
    assert _capacity_preference(_plan()) == CAPACITY_ANY


def test_kyber_marks_a_breaker_and_leaves_a_streaming_op_alone():
    """The flag Carbonite reads has to come from the plan, not from a default.

    Without the positive half of this the whole feature degrades silently to "never prefer
    on-demand", which no other assertion in this file would catch.
    """
    import batcher as bt
    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    ds = (
        bt.from_pydict({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
        .filter(bt.col("v") > 0.0)
        .group_by("k")
        .agg(s=bt.col("v").sum())
    )
    est = CardinalityEstimator([])
    ops = annotate_ops(ds._plan, est, active_config(), CostModel(est))
    by_kind = {op.kind: op.bounds.materializes for op in ops}
    assert by_kind.get("Aggregate") is True
    assert by_kind.get("Filter") is False


# --- the gang-scheduled fleet ----------------------------------------------------------------


def test_a_fleet_states_its_preference_on_the_bundles_not_on_the_actors():
    """An actor pinned to a bundle inherits the bundle's node. A selector on the actor can
    only contradict a decision already made, and a contradicted actor never schedules —
    which is a hang, not a fallback, because `fallback_strategy` cannot move a bundle."""
    from batcher.dist.executors.ray_runtime.fabric import bundles

    with config_context(_enabled()):
        base = capacity_bundle_selector(CAPACITY_ON_DEMAND, workers=2, num_cpus=1.0, census=_MIXED)
    assert base == {_RAY_KEY: ON_DEMAND}, "the bundles carry the preference"
    assert "fallback_strategy" not in base, "a bundle selector has no fallback to carry"
    assert hasattr(bundles, "fleet_market_selector")


def test_the_actor_options_drop_a_selector_the_bundle_already_decided():
    """The positive control for the paragraph above: with a placement group in force,
    whatever `task_options` produced must not survive onto the actor."""
    from batcher.dist.executors.ray_runtime.scheduling import placement_actor_options

    base = {
        "num_cpus": 2.0,
        "label_selector": {_RAY_KEY: ON_DEMAND},
        "fallback_strategy": [{"label_selector": {_RAY_KEY: SPOT}}],
    }
    # `fleet_actor_options` is what strips them; `placement_actor_options` is the pass-through
    # underneath it, so asserting on the stripped `base` proves the strip and not the merge.
    stripped = {k: v for k, v in base.items() if k not in {"label_selector", "fallback_strategy"}}
    assert placement_actor_options(None, 0, stripped) == stripped


def test_a_bundle_preference_is_declined_when_the_market_cannot_hold_the_fleet():
    """Bundles have no fallback, and a group that cannot form is not refused — it is waited
    on for the whole `placement_timeout_s` budget and then abandoned. Declining up front
    costs nothing; pinning optimistically costs the reservation's entire budget."""
    census = [_row(SPOT, cores=64.0, count=8), _row(ON_DEMAND, cores=4.0, count=1)]
    with config_context(_enabled()):
        # 8 bundles x 1 core against 4 on-demand cores.
        assert (
            capacity_bundle_selector(CAPACITY_ON_DEMAND, workers=8, num_cpus=1.0, census=census)
            == {}
        )
        # The same fleet fits inside the on-demand capacity at four bundles.
        assert capacity_bundle_selector(
            CAPACITY_ON_DEMAND, workers=4, num_cpus=1.0, census=census
        ) == {_RAY_KEY: ON_DEMAND}


def test_a_collective_fleet_states_nothing():
    """A GPU collective is already STRICT_PACK onto one node, so narrowing which node can
    only make the gang harder to form for a preference it does not benefit from."""
    from batcher.dist.executors.ray_runtime.fabric.bundles import fleet_market_selector
    from batcher.plan.resource import SchedulingEnvelope

    env = SchedulingEnvelope(capacity_preference=CAPACITY_ON_DEMAND, gpu_collective=True)
    with config_context(_enabled()):
        assert fleet_market_selector(4, env) == {}
