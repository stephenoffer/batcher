"""Reducer placement derives node identity from the addresses the driver already holds.

Locality-aware scheduling hosts a reducer on the node its bucket concentrates on. Deciding
that needs to know which node each worker is on — which used to cost a `node_id` round-trip
per worker, paid even to discover a single-node fleet where the answer is always "no
placement to make". The advertised shuffle address carries the same node identity for free,
and it is the identity `select_mode` routes on, so placement and transport agree.

The assertions here are on which remote methods were *called*, not on the returned
placement: the probe swallows its own failures by design, so an actor that raises would
make an "it didn't probe" test pass no matter what happened.
"""

from __future__ import annotations

import pytest

from batcher.config import Config, DistributedConfig, config_context
from batcher.dist.flight_aggregate import _locality_reducer_hosts

pytestmark = pytest.mark.unit


class _Call:
    def __init__(self, log: list[str], name: str, value: object) -> None:
        self._log, self._name, self._value = log, name, value

    def remote(self) -> object:
        self._log.append(self._name)
        return self._value


class _Worker:
    """A stand-in actor that records every remote method the placement probe calls."""

    def __init__(self, log: list[str], node: str, buckets: dict[int, int]) -> None:
        self.node_id = _Call(log, "node_id", node)
        self.published_bucket_bytes = _Call(log, "published_bucket_bytes", buckets)


@pytest.fixture
def no_ray_get(monkeypatch):
    """`ray.get` over the stubs' already-resolved values — identity, no cluster."""
    ray = pytest.importorskip("ray")
    monkeypatch.setattr(ray, "get", lambda refs: list(refs))


def _locality_on(enabled: bool) -> Config:
    return Config().replace(distributed=DistributedConfig(locality_aware_scheduling=enabled))


def test_disabled_makes_no_decision_and_no_probe(no_ray_get):
    log: list[str] = []
    workers = [_Worker(log, f"10.0.0.{i}", {0: 100}) for i in (1, 2)]
    with config_context(_locality_on(False)):
        assert _locality_reducer_hosts(workers, 4, 2, ["10.0.0.1:1", "10.0.0.2:2"]) is None
    assert log == []


def test_a_single_node_fleet_short_circuits_without_probing(no_ray_get):
    """Every fetch is already same-node, so there is nothing to place — and that is
    settled from the addresses alone, without a round-trip to any worker."""
    log: list[str] = []
    addrs = ["10.0.0.1:100", "10.0.0.1:200", "10.0.0.1:300"]
    workers = [_Worker(log, "10.0.0.1", {0: 100}) for _ in addrs]
    with config_context(_locality_on(True)):
        assert _locality_reducer_hosts(workers, 6, 3, addrs) is None
    assert log == []


def test_node_identity_comes_from_the_addresses_not_from_a_probe(no_ray_get):
    """A multi-node fleet still probes for per-mapper bytes — but never for node ids."""
    log: list[str] = []
    # Bucket 0's bytes sit almost entirely on the second node; bucket 1 is spread.
    workers = [
        _Worker(log, "10.0.0.1", {0: 1, 1: 50}),
        _Worker(log, "10.0.0.2", {0: 999, 1: 50}),
    ]
    addrs = ["10.0.0.1:100", "10.0.0.2:100"]
    with config_context(_locality_on(True)):
        hosts = _locality_reducer_hosts(workers, 2, 2, addrs)
    assert log == ["published_bucket_bytes", "published_bucket_bytes"]
    assert hosts is not None
    assert hosts[0] == 1  # reducer 0 hosted where its bucket concentrates
    assert hosts[1] == 1  # spread bucket keeps the default `r % workers` round-robin


def test_missing_addresses_fall_back_to_the_node_probe(no_ray_get):
    """A caller that cannot supply addresses (or a fleet with an unbound worker) still
    gets locality, just at the cost of the round-trip."""
    log: list[str] = []
    workers = [
        _Worker(log, "nodeA", {0: 1}),
        _Worker(log, "nodeB", {0: 999}),
    ]
    with config_context(_locality_on(True)):
        hosts = _locality_reducer_hosts(workers, 2, 2, None)
    assert log[:2] == ["node_id", "node_id"]
    assert hosts is not None and hosts[0] == 1
