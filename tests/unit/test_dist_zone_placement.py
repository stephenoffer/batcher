"""Zone-aware fleet placement: keep a shuffle's bytes out of the cross-zone toll.

A shuffle moves nearly all of its bytes worker to worker, and every cloud bills and delays
those bytes by whether the two workers share an availability zone. The bundles of a shuffle
fleet are interchangeable, so a fleet that fits inside one zone can be placed inside one — and
a fleet spread evenly over three zones was sending about two thirds of its shuffle across a
boundary it never had to cross.

These tests pin the decision against a stubbed topology, because it must be conservative in
every direction: a zone that cannot host the fleet is never chosen (that trades a cost saving
for a placement timeout), an unlabelled or single-zone cluster is left alone, and the label
*key* travels with the value so a Kubernetes-labelled fleet is not selected on Ray's key.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.capacity import Demand, preferred_fleet_zone

pytestmark = pytest.mark.unit

_K8S = "topology.kubernetes.io/zone"
_RAY = "ray.io/availability-zone"


def _node(cpus: float, zone: str, *, free: float | None = None, label: str = _RAY, gpus=0.0):
    return {
        "node_id": f"{zone}-{cpus}-{free}",
        "cpus": cpus,
        "free_cpus": cpus if free is None else free,
        "gpus": gpus,
        "memory": 0.0,
        "accelerators": 0.0,
        "accelerator_type": None,
        "zone_label": label if zone else "",
        "zone": zone,
    }


@pytest.fixture
def topology(monkeypatch):
    def _install(nodes):
        import batcher.dist.executors.ray_runtime.scaling as scaling

        monkeypatch.setattr(scaling, "node_classes", lambda: list(nodes))

    return _install


def test_a_fleet_that_fits_one_zone_is_pinned_to_it(topology):
    topology([_node(16, "us-west-2a"), _node(16, "us-west-2b")])
    assert preferred_fleet_zone(8, Demand(num_cpus=2)) in (
        {_RAY: "us-west-2a"},
        {_RAY: "us-west-2b"},
    )


def test_the_zone_with_the_most_free_capacity_wins(topology):
    """Ties are broken on free capacity, so the pin lands where the gang can actually form."""
    topology(
        [
            _node(16, "us-west-2a", free=4.0),
            _node(16, "us-west-2b", free=16.0),
        ]
    )
    assert preferred_fleet_zone(2, Demand(num_cpus=2)) == {_RAY: "us-west-2b"}


def test_a_fleet_no_single_zone_can_host_is_left_to_spread(topology):
    """Pinning a fleet into a zone too small for it converts a cost saving into a hang.

    The placement group would never form, the timeout would fire, and the stage would fall
    back to default scheduling having spent the whole budget waiting.
    """
    topology([_node(8, "us-west-2a"), _node(8, "us-west-2b")])
    assert preferred_fleet_zone(8, Demand(num_cpus=2)) == {}


def test_capacity_is_counted_free_not_nameplate(topology):
    """A zone a co-tenant has filled is not a zone this fleet can form in.

    Nameplate sizing would pick the busiest zone as readily as the emptiest, which is the one
    way this optimization can make a query slower rather than cheaper.
    """
    topology([_node(16, "us-west-2a", free=0.0), _node(16, "us-west-2b", free=0.0)])
    assert preferred_fleet_zone(4, Demand(num_cpus=2)) == {}


def test_a_single_zone_cluster_is_not_pinned(topology):
    """Nothing to choose, so nothing is constrained — the selector must stay out of the way."""
    topology([_node(16, "us-west-2a"), _node(16, "us-west-2a")])
    assert preferred_fleet_zone(4, Demand(num_cpus=1)) == {}


def test_an_unlabelled_cluster_is_not_pinned(topology):
    """A fleet whose zones Batcher cannot see gets exactly the placement it had before."""
    topology([_node(16, ""), _node(16, "")])
    assert preferred_fleet_zone(4, Demand(num_cpus=1)) == {}


def test_the_selector_uses_the_label_key_the_zone_was_read_from(topology):
    """A Kubernetes-labelled fleet must be selected on the Kubernetes key.

    `node_zone` reads `topology.kubernetes.io/zone` before Ray's own key, so a selector that
    restated Ray's key would match no node and make every such cluster unplaceable.
    """
    topology([_node(16, "eu-central-1a", label=_K8S), _node(16, "eu-central-1b", label=_K8S)])
    selector = preferred_fleet_zone(4, Demand(num_cpus=1))
    assert list(selector) == [_K8S]
    assert next(iter(selector.values())) in {"eu-central-1a", "eu-central-1b"}


def test_a_gpu_bound_fleet_counts_devices_as_well_as_cores(topology):
    """A GPU stage's zone has to hold its devices, not merely its cores."""
    topology(
        [
            _node(64, "us-west-2a", gpus=1.0),
            _node(64, "us-west-2b", gpus=8.0),
        ]
    )
    assert preferred_fleet_zone(4, Demand(num_cpus=1, num_gpus=1.0)) == {_RAY: "us-west-2b"}


def test_an_unreadable_topology_pins_nothing(topology):
    topology([])
    assert preferred_fleet_zone(4, Demand(num_cpus=1)) == {}


def test_a_gpu_collective_is_never_zone_pinned(monkeypatch):
    """A collective is already STRICT_PACK onto one node, so a zone pin can only narrow it.

    Asserted at the gate rather than at `preferred_fleet_zone`, because the exclusion is a
    scheduling decision about which fleets the optimization applies to.
    """
    from batcher.dist.executors.ray_runtime import scheduling
    from batcher.plan.resource import SchedulingEnvelope

    # Fails loudly if the gate is reached at all: the collective must be excluded before any
    # topology or config is read, so a later refactor that moves the check cannot go quiet.
    monkeypatch.setattr(
        scheduling, "active_config", lambda: pytest.fail("a collective must be gated out first")
    )
    env = SchedulingEnvelope(num_cpus=1.0, gpu_collective=True)
    assert scheduling._fleet_zone_selector(4, env) == {}


def test_the_config_switch_turns_the_pin_off(monkeypatch):
    """Zone diversity is sometimes bought deliberately, so it has to be declinable."""
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.dist.executors.ray_runtime import scheduling

    base = active_config()
    off = base.replace(
        distributed=dataclasses.replace(base.distributed, zone_aware_placement=False)
    )
    with config_context(off):
        assert scheduling._fleet_zone_selector(4, None) == {}


# --- market type -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        {"ray.io/market-type": "spot"},
        {"ray.io/market-type": "SPOT"},
        {"karpenter.sh/capacity-type": "spot"},
        {"eks.amazonaws.com/capacityType": "SPOT"},
        {"cloud.google.com/gke-spot": "true"},
    ],
)
def test_every_provisioners_spelling_of_spot_is_recognized(labels):
    """A KubeRay fleet is labelled by whoever brought the node up, not by Ray.

    Reading only Ray's own key would report every Karpenter, EKS, and GKE spot fleet as
    durable — which is the population the replica placement most needs to see.
    """
    from batcher.dist.executors.ray_runtime.fabric.topology import is_preemptible

    assert is_preemptible(labels)


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"ray.io/market-type": "on-demand"},
        {"cloud.google.com/gke-spot": "false"},
        {"ray.io/region": "us-west-2"},
    ],
)
def test_an_unlabelled_or_on_demand_node_is_not_preemptible(labels):
    """The safe direction: an absent label means "no evidence", never "assume spot".

    Distrusting every unlabelled node would change the placement on every cluster that does
    not label its capacity, which is most of them.
    """
    from batcher.dist.executors.ray_runtime.fabric.topology import is_preemptible

    assert not is_preemptible(labels)


# --- the placement decision ---------------------------------------------------------------


def test_the_placement_is_reported_as_a_decision():
    """Where the fleet was reserved is the other half of "why did my query run like that".

    A SPREAD that quietly became PACK on a one-node cluster, a fleet pinned into one zone, a
    collective that got STRICT_PACK — all invisible before, and all exactly what a reader
    asks about. Reported as a `Decision` so it lands beside Kyber's and Carbonite's rather
    than in a log nobody enabled.
    """
    from batcher._internal import events
    from batcher.dist.executors.ray_runtime import scheduling

    seen: list = []
    stop = events.subscribe(seen.append)
    try:
        scheduling._report_placement(8, "SPREAD", {_RAY: "us-west-2b"})
    finally:
        stop()
    decisions = [e for e in seen if e.kind == events.DECISION]
    assert decisions, "a successful reservation must be reported"
    fields = decisions[0].fields
    assert fields["category"] == "placement"
    assert "8 bundle(s) SPREAD in us-west-2b" in fields["summary"]
    assert fields["detail"]["zone"] == {_RAY: "us-west-2b"}


def test_an_unpinned_placement_reports_no_zone():
    """Most clusters are one zone, and the message must not imply a choice was made."""
    from batcher._internal import events
    from batcher.dist.executors.ray_runtime import scheduling

    seen: list = []
    stop = events.subscribe(seen.append)
    try:
        scheduling._report_placement(4, "PACK", {})
    finally:
        stop()
    fields = next(e for e in seen if e.kind == events.DECISION).fields
    assert fields["summary"] == "reserved 4 bundle(s) PACK"
    assert fields["detail"]["zone"] == {}


def test_reporting_a_placement_never_disturbs_it(monkeypatch):
    """This describes a reservation that already succeeded; it must not be able to undo one."""
    from batcher.dist.executors.ray_runtime import scheduling

    monkeypatch.setattr(
        "batcher._internal.events.publish",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bus is down")),
    )
    scheduling._report_placement(4, "SPREAD", {})  # must not raise
