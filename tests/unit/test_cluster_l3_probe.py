"""Kyber's broadcast threshold must be sized from the workers' real cache, not a default.

Ray's topology omits L3 cache, so `cluster_hardware_profile` left it 0 and every distributed
query fell back to the config broadcast threshold. The probe collects it from the workers. Two
properties matter on a heterogeneous cluster: it samples one worker per distinct node shape (not
one worker, assuming uniformity), and it takes the minimum (a broadcast table sized to the
biggest cache would spill out of the smallest node's).

The probe returns each shape's whole `HardwareProfile` rather than one cache figure — the round
trip is the cost, so every additional field is free once a task has been scheduled — and the
binding minimum is taken by `cluster_l3_cache_bytes` over those profiles.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import hardware_probe as hp

pytestmark = pytest.mark.unit


def _node(node_id, cpus, gpus=0.0, accel=None):
    res = {"CPU": cpus, "GPU": gpus}
    labels = {"ray.io/accelerator-type": accel} if accel else {}
    return {"NodeID": node_id, "Alive": True, "Resources": res, "Labels": labels}


def test_one_representative_per_distinct_node_shape():
    """Two shapes (an 8-core and a 64-core class) → two representatives, not one and not four."""
    nodes = [
        _node("a1", 8.0),
        _node("a2", 8.0),  # same shape as a1 → not a separate representative
        _node("b1", 64.0),
        _node("b2", 64.0),
    ]
    reps = hp._representative_node_ids(nodes)
    assert len(reps) == 2
    assert set(reps) <= {"a1", "a2", "b1", "b2"}
    assert reps[0] in {"a1", "a2"} and reps[1] in {"b1", "b2"}


def test_gpu_and_accelerator_type_split_shapes():
    """Same cores but different accelerators are distinct shapes — cache may differ by instance."""
    nodes = [
        _node("t4", 16.0, gpus=1.0, accel="NVIDIA_TESLA_T4"),
        _node("a100", 16.0, gpus=1.0, accel="NVIDIA_A100"),
    ]
    assert len(hp._representative_node_ids(nodes)) == 2


def test_nodes_without_cpus_or_ids_are_skipped():
    nodes = [_node("ok", 8.0), _node("", 8.0), {"NodeID": "z", "Resources": {"CPU": 0.0}}]
    assert hp._representative_node_ids(nodes) == ["ok"]


def _fake_scheduling_strategies(monkeypatch):
    """Stand in for the Ray scheduling-strategy import `_probe_representatives` performs."""
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        types.SimpleNamespace(NodeAffinitySchedulingStrategy=lambda *a, **k: None),
    )


class _FakeRay:
    """Ray's task API, resolving each ref on its own.

    One result per representative, looked up by ref rather than returned as a batch, because
    `_probe_representatives` resolves its fan-out per ref — a representative that raises must
    drop out without taking the shapes that answered with it. An `Exception` in the results
    list stands for a node whose probe failed.
    """

    def __init__(self, results):
        self._results = list(results)
        self._issued = 0

    def remote(self, **kw):
        def make(fn):
            outer = self

            class _R:
                def options(self, **o):
                    return self

                def remote(self):
                    outer._issued += 1
                    return outer._issued - 1

            return _R()

        return make

    def wait(self, refs, num_returns, timeout):
        return refs, []

    def get(self, ref):
        result = self._results[ref]
        if isinstance(result, Exception):
            raise result
        return result


def test_the_binding_minimum_is_taken_across_shapes(monkeypatch):
    """A big-cache node and a small-cache node → the threshold binds to the small one."""
    _fake_scheduling_strategies(monkeypatch)
    # 36 MiB EPYC vs 8 MiB small node, as whole profiles.
    profiles = hp._probe_representatives(
        _FakeRay(
            [
                {"fingerprint": "epyc00000000", "caches": {"l3": 36 * 1024 * 1024}},
                {"fingerprint": "small0000000", "caches": {"l3": 8 * 1024 * 1024}},
            ]
        ),
        ["a", "b"],
    )
    assert len(profiles) == 2
    monkeypatch.setattr(hp, "cluster_hardware_profiles", lambda: profiles)
    assert hp.cluster_l3_cache_bytes() == 8 * 1024 * 1024  # the smaller cache binds
    # Two shapes, two fingerprints: the fleet is mixed, and everything learned on one node is
    # kept apart from the other's rather than averaged into a model wrong for both.
    assert hp.cluster_is_heterogeneous() is True


def test_a_shape_reporting_zero_cache_does_not_drag_the_min_to_zero(monkeypatch):
    _fake_scheduling_strategies(monkeypatch)
    # One node's cache is undetectable; dropping it beats letting a `0` disable the
    # broadcast threshold for the whole cluster.
    profiles = hp._probe_representatives(
        _FakeRay(
            [
                {"fingerprint": "unknown00000", "caches": {}},
                {"fingerprint": "known0000000", "caches": {"l3": 16 * 1024 * 1024}},
            ]
        ),
        ["a", "b"],
    )
    monkeypatch.setattr(hp, "cluster_hardware_profiles", lambda: profiles)
    assert hp.cluster_l3_cache_bytes() == 16 * 1024 * 1024


def test_a_failing_shape_does_not_discard_the_shapes_that_answered(monkeypatch):
    """One representative that raises must not cost the cluster every other shape's profile.

    `ray.get(ready)` over the whole list raises on the first failed task, and
    `cluster_hardware_profiles`' `except` then returns `()` — so a single unanswerable node put
    `cluster_l3_cache_bytes()` back to `0` and every distributed query back on the config
    broadcast threshold, on a cluster that had just measured its cache. A fleet whose workers
    run a different Batcher build than the driver fails this way on every node at once, which
    is indistinguishable from having no cluster at all.
    """

    _fake_scheduling_strategies(monkeypatch)
    profiles = hp._probe_representatives(
        _FakeRay(
            [
                {"fingerprint": "good00000000", "caches": {"l3": 32 * 1024 * 1024}},
                RuntimeError("ImportError: cannot import name 'hardware_profile'"),
            ]
        ),
        ["good", "stale"],
    )
    assert len(profiles) == 1
    monkeypatch.setattr(hp, "cluster_hardware_profiles", lambda: profiles)
    assert hp.cluster_l3_cache_bytes() == 32 * 1024 * 1024


def test_a_wholly_unprobeable_fleet_says_so_once(monkeypatch, caplog):
    """No shape answering is reported; a fleet that was never asked stays quiet.

    Each node's failure is a DEBUG note, so without this a cluster whose workers cannot run the
    probe at all — the shape a mismatched worker build takes — planned on fallback defaults
    with nothing anywhere to say why.
    """
    import logging
    import sys
    import types

    fake = types.SimpleNamespace(
        is_initialized=lambda: True,
        nodes=lambda: [_node("a1", 8.0)],
    )
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setattr(hp, "_UNPROBEABLE_WARNED", False)
    monkeypatch.setattr(hp, "_PROFILES_BY_TOPOLOGY", {})
    monkeypatch.setattr(hp, "_probe_representatives", lambda ray, reps: ())
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        assert hp.cluster_hardware_profiles() == ()
    assert "no worker answered the hardware probe" in caplog.text

    # Once per process: the second query on the same dead fleet adds no second line.
    caplog.clear()
    monkeypatch.setattr(hp, "_PROFILES_BY_TOPOLOGY", {})
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        hp.cluster_hardware_profiles()
    assert "no worker answered" not in caplog.text
