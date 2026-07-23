"""Kyber's broadcast threshold must be sized from the workers' real cache, not a default.

Ray's topology omits L3 cache, so `cluster_hardware_profile` left it 0 and every distributed
query fell back to the config broadcast threshold. The probe collects it from the workers. Two
properties matter on a heterogeneous cluster: it samples one worker per distinct node shape (not
one worker, assuming uniformity), and it takes the minimum (a broadcast table sized to the
biggest cache would spill out of the smallest node's).
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


def test_the_binding_minimum_is_taken_across_shapes(monkeypatch):
    """A big-cache node and a small-cache node → the threshold binds to the small one."""

    class _FakeRay:
        def remote(self, **kw):
            def make(fn):
                class _R:
                    def options(self, **o):
                        return self

                    def remote(self):
                        return "ref"

                return _R()

            return make

        def wait(self, refs, num_returns, timeout):
            return refs, []

        def get(self, ready):
            return [36 * 1024 * 1024, 8 * 1024 * 1024]  # 36 MiB EPYC vs 8 MiB small node

    # Patch the strategy import target used inside _probe_representatives.
    import sys
    import types

    fake_mod = types.SimpleNamespace(NodeAffinitySchedulingStrategy=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", fake_mod)
    got = hp._probe_representatives(_FakeRay(), ["a", "b"])
    assert got == 8 * 1024 * 1024  # the smaller cache binds


def test_a_shape_reporting_zero_cache_does_not_drag_the_min_to_zero(monkeypatch):
    class _FakeRay:
        def remote(self, **kw):
            def make(fn):
                class _R:
                    def options(self, **o):
                        return self

                    def remote(self):
                        return "ref"

                return _R()

            return make

        def wait(self, refs, num_returns, timeout):
            return refs, []

        def get(self, ready):
            return [0, 16 * 1024 * 1024]  # one node's cache is undetectable

    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        types.SimpleNamespace(NodeAffinitySchedulingStrategy=lambda *a, **k: None),
    )
    assert hp._probe_representatives(_FakeRay(), ["a", "b"]) == 16 * 1024 * 1024
