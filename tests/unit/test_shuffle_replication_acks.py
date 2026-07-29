"""Replica acknowledgements are collected together, and a failed one degrades alone.

Replication runs at the map barrier — the point where the reduce is already waiting — so
collecting `workers x factor` acks one blocking `ray.get` at a time put that many
sequential round trips on the critical path. Waiting for all of them at once must not cost
the per-source error isolation the degradation story depends on: a source whose replica
never acked keeps recompute, it does not fail the query.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, config_context
from batcher.dist import shuffle_replication as repl


class _FakeRef:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error


class _FakeRay:
    """Models the three `ray` calls this module makes, and records the wait shape."""

    def __init__(self, values: dict) -> None:
        self._values = values
        self.wait_calls: list[int] = []
        self.get_calls = 0

    def get(self, ref):
        if isinstance(ref, list):
            return [self.get(r) for r in ref]
        self.get_calls += 1
        if ref.error is not None:
            raise ref.error
        return ref.value

    def wait(self, refs, num_returns=1, **_):
        self.wait_calls.append(num_returns)
        return list(refs), []


class _Actor:
    def __init__(self, idx: int, node: str, addr: str, ack) -> None:
        self._idx, self._node, self._addr, self._ack = idx, node, addr, ack

    class _Remote:
        def __init__(self, value):
            self._value = value

        def remote(self, *a, **k):
            return _FakeRef(self._value)

    @property
    def node_id(self):
        return self._Remote(self._node)

    @property
    def addr(self):
        return self._Remote(self._addr)

    @property
    def replicate_buckets(self):
        ack = self._ack

        class _R:
            def remote(self, *a, **k):
                return ack

        return _R()


@pytest.fixture
def replicating():
    base = Config()
    cfg = base.replace(distributed=dataclasses.replace(base.distributed, shuffle_replication=2))
    with config_context(cfg):
        yield


def _run(monkeypatch, acks):
    """Two workers on distinct nodes, each replicating to the other."""
    addrs = ["a0:1", "a1:1"]
    actors = [
        _Actor(0, "nodeA", addrs[0], acks[0]),
        _Actor(1, "nodeB", addrs[1], acks[1]),
    ]
    fake = _FakeRay({})
    monkeypatch.setitem(__import__("sys").modules, "ray", fake)
    result = repl.replicate_shuffle_output(actors, addrs, 2, 2, set())
    return result, fake


def test_every_ack_is_waited_for_in_one_call(monkeypatch, replicating):
    acks = [_FakeRef("a1:1"), _FakeRef("a0:1")]
    result, fake = _run(monkeypatch, acks)
    assert result is not None
    # One wait covering every outstanding ack, rather than one blocking get each.
    assert len(fake.wait_calls) == 1
    assert fake.wait_calls[0] == 2


def test_a_source_whose_replica_failed_degrades_alone(monkeypatch, replicating):
    """It keeps recompute; the other source keeps its replica and the query stands.

    The ack comes from the *host* actor holding the copy, not from the source's own
    worker: with two workers, source 0's copy lives on host 1 and source 1's on host 0.
    """
    acks = [_FakeRef(error=RuntimeError("replica never acked")), _FakeRef("a0:1")]
    result, _ = _run(monkeypatch, acks)
    assert result is not None
    # host 0's ack failed, so the source it was holding (source 1) has no replica; the
    # source hosted by the healthy actor keeps one.
    assert result[1] == []
    assert result[0] == ["a0:1"]


def test_replication_off_places_nothing(monkeypatch):
    base = Config()
    cfg = base.replace(distributed=dataclasses.replace(base.distributed, shuffle_replication=1))
    with config_context(cfg):
        acks = [_FakeRef("x"), _FakeRef("y")]
        result, fake = _run(monkeypatch, acks)
    assert result is None
    assert fake.wait_calls == []


def test_a_single_worker_cluster_cannot_host_an_independent_copy(monkeypatch, replicating):
    addrs = ["a0:1"]
    actors = [_Actor(0, "nodeA", addrs[0], _FakeRef("a0:1"))]
    fake = _FakeRay({})
    monkeypatch.setitem(__import__("sys").modules, "ray", fake)
    assert repl.replicate_shuffle_output(actors, addrs, 2, 1, set()) is None


def test_a_probe_failure_is_noted_rather_than_silent(monkeypatch, replicating, caplog):
    """A probe that always fails turns replication permanently off, and every worker loss
    then pays a full map-stage recompute with nothing saying the cheaper path was gone."""

    class _AngryRay(_FakeRay):
        def get(self, ref):
            raise RuntimeError("cluster probe refused")

    addrs = ["a0:1", "a1:1"]
    actors = [
        _Actor(0, "nodeA", addrs[0], _FakeRef("x")),
        _Actor(1, "nodeB", addrs[1], _FakeRef("y")),
    ]
    monkeypatch.setitem(__import__("sys").modules, "ray", _AngryRay({}))
    from batcher._internal.logging import _FIELDS_ATTR

    with caplog.at_level("DEBUG", logger="batcher.dist"):
        assert repl.replicate_shuffle_output(actors, addrs, 2, 2, set()) is None
    steps = [getattr(r, _FIELDS_ATTR, {}).get("step") for r in caplog.records]
    assert "probe workers for replica placement" in steps, caplog.text
