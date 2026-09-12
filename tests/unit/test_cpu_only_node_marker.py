"""The CPU-only node marker must be read off the cluster, not assumed.

`heterogeneous_node_isolation` keeps a CPU fleet off accelerator nodes by requiring a custom
resource only CPU-only nodes advertise. The name of that resource was the configured default
`cpu_node`, which is a deploy-time convention an operator applies by hand — so on a managed
fleet that had already labelled itself under its own name the selector asked for a resource
nothing carried, and the whole mechanism was inert on the cluster shape it exists for. That
is the same defect the zone selector fixed by reading the label key off the node.

The interesting half is what the discovery must *refuse*. On a fleet with a single CPU
instance type, the instance-type label is carried by exactly the CPU-only nodes and by no
accelerator node, so a purely structural "any label they all share" rule picks it — and it
is a different fact, which stops matching the moment a second CPU instance type appears.
`test_an_instance_type_label_is_not_a_cpu_only_marker` is that case.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, config_context
from batcher.dist.executors.ray_runtime import scaling
from batcher.dist.executors.ray_runtime.node_markers import cpu_only_marker_resource


def _isolated() -> Config:
    """A config with the heterogeneous-isolation gate on; the config itself is frozen."""
    cfg = Config()
    return cfg.replace(
        distributed=dataclasses.replace(cfg.distributed, heterogeneous_node_isolation=True)
    )


def _node(cpus: float, resources: dict) -> dict:
    """One `ray.nodes()` record, in the shape `_alive_nodes` hands on."""
    return {"Resources": {"CPU": cpus, "memory": 1 << 30, **resources}}


def _anyscale_fleet() -> list[dict]:
    """Eight 16-core CPU nodes and eight 1xT4 nodes, labelled the way Anyscale labels them."""
    cpu = [
        _node(
            16.0,
            {
                "anyscale/cpu_only:true": 1.0,
                "anyscale/node-group:16cpu-32gb": 1.0,
                "anyscale/provider:aws": 1.0,
                f"node:10.0.0.{i}": 1.0,
            },
        )
        for i in range(8)
    ]
    gpu = [
        _node(
            8.0,
            {
                "GPU": 1.0,
                "accelerator_type:T4": 1.0,
                "anyscale/accelerator_shape:1xT4": 1.0,
                "anyscale/node-group:1xt4-8cpu-32gb": 1.0,
                "anyscale/provider:aws": 1.0,
                f"node:10.0.1.{i}": 1.0,
            },
        )
        for i in range(8)
    ]
    return cpu + gpu


@pytest.fixture
def fleet(monkeypatch):
    """Install a synthetic fleet for the *selector*, which does read the cluster.

    `cpu_only_marker_resource` itself takes the node list as an argument, so the tests of the
    discovery rule below need no patching at all -- which is the point of extracting it.
    """

    def use(nodes: list[dict]) -> None:
        monkeypatch.setattr(scaling, "_alive_nodes", lambda: nodes)

    return use


def test_a_managed_fleets_own_label_is_discovered():
    """The regression: the marker is the name the cluster actually advertises."""
    assert cpu_only_marker_resource(_anyscale_fleet(), "cpu_node") == "anyscale/cpu_only:true"


def test_the_configured_name_wins_when_the_operator_applied_it():
    """A deploy-time `cpu_node` label keeps precedence over anything discovered."""
    nodes = _anyscale_fleet()
    for node in nodes[:8]:
        node["Resources"]["cpu_node"] = 4.0
    assert cpu_only_marker_resource(nodes, "cpu_node") == "cpu_node"


def test_an_instance_type_label_is_not_a_cpu_only_marker():
    """A label shared by the CPU nodes but naming their *shape* must not be picked."""
    nodes = _anyscale_fleet()
    for node in nodes[:8]:
        del node["Resources"]["anyscale/cpu_only:true"]
    # `anyscale/node-group:16cpu-32gb` is on every CPU node and no GPU node, and is still
    # not the property being selected for — so the configured name stands, unchanged.
    assert cpu_only_marker_resource(nodes, "cpu_node") == "cpu_node"


def test_a_label_on_one_gpu_node_disqualifies_it():
    """A marker must be absent from *every* accelerator node, not most of them."""
    nodes = _anyscale_fleet()
    nodes[-1]["Resources"]["anyscale/cpu_only:true"] = 1.0
    assert cpu_only_marker_resource(nodes, "cpu_node") == "cpu_node"


def test_a_homogeneous_cluster_falls_back():
    """Nothing to keep a fleet off; the gate above decides, and this stays as configured."""
    assert cpu_only_marker_resource(_anyscale_fleet()[:8], "cpu_node") == "cpu_node"


def test_an_all_accelerator_cluster_falls_back():
    """Nowhere to put a CPU fleet; `cpu_only_can_host` is what refuses, not this."""
    assert cpu_only_marker_resource(_anyscale_fleet()[8:], "cpu_node") == "cpu_node"


def test_a_custom_accelerator_node_counts_as_an_accelerator():
    """A TPU/Trainium node exposes no `GPU`, and must still not host an isolated CPU fleet."""
    nodes = [
        _node(16.0, {"anyscale/cpu_only:true": 1.0, "node:10.0.0.1": 1.0}),
        _node(8.0, {"TPU": 4.0, "node:10.0.1.1": 1.0}),
    ]
    assert cpu_only_marker_resource(nodes, "cpu_node") == "anyscale/cpu_only:true"


def test_the_selector_keeps_the_configured_name_when_nothing_marks_the_cpu_nodes(
    fleet, monkeypatch
):
    """An undiscoverable marker leaves the deploy-time contract exactly as it was."""
    nodes = _anyscale_fleet()
    for node in nodes[:8]:
        del node["Resources"]["anyscale/cpu_only:true"]
    fleet(nodes)
    monkeypatch.setattr(scaling, "cpu_only_can_host", lambda _w, _c: True)
    with config_context(_isolated()):
        opts = scaling.node_class_selector(True, workers=4, num_cpus=2.0)
    assert opts == {"resources": {"cpu_node": scaling._CPU_NODE_EPS}}


def test_the_selector_requires_the_discovered_marker(fleet, monkeypatch):
    """The regression, at the seam that places actors: the fleet is pinned to the real name."""
    fleet(_anyscale_fleet())
    monkeypatch.setattr(scaling, "cpu_only_can_host", lambda _w, _c: True)
    with config_context(_isolated()):
        opts = scaling.node_class_selector(True, workers=4, num_cpus=2.0)
    assert opts == {"resources": {"anyscale/cpu_only:true": scaling._CPU_NODE_EPS}}
