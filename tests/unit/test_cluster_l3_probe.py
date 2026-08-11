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
    """No shape answering is reported once the retries are exhausted; a transient miss is not.

    Each node's failure is a DEBUG note, so without this a cluster whose workers cannot run the
    probe at all — the shape a mismatched worker build takes — planned on fallback defaults
    with nothing anywhere to say why.

    The warning deliberately waits for `_MAX_PROBE_ATTEMPTS`. A single miss is far more often
    an autoscaling fleet whose workers have not finished starting, and warning on that names a
    cause ("a different Batcher build") that is usually wrong — which is precisely how it
    misled a reader into hunting a build mismatch that did not exist.
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
    monkeypatch.setattr(hp, "_FAILED_ATTEMPTS", {})
    monkeypatch.setattr(hp, "_probe_representatives", lambda ray, reps: ())

    # A transient miss stays quiet: this is the cold-start case, and it is the common one.
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        assert hp.cluster_hardware_profiles() == ()
    assert "no worker answered" not in caplog.text

    # Keep missing, and the fleet is reported — once the diagnosis is actually earned.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        for _ in range(hp._MAX_PROBE_ATTEMPTS):
            assert hp.cluster_hardware_profiles() == ()
    assert "no worker answered the hardware probe" in caplog.text

    # Once per process: further queries on the same dead fleet add no second line.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        hp.cluster_hardware_profiles()
    assert "no worker answered" not in caplog.text


class _FleetRay:
    """`ray` as `cluster_hardware_profiles` uses it: `is_initialized`, `nodes`, and the task API.

    `answers` is consumed one probe round at a time, so a fleet can be made to miss on the
    first call and answer on the second — the cold-start race this module's cache rules exist
    for.
    """

    def __init__(self, nodes, answers):
        self._nodes = nodes
        self._answers = list(answers)
        self.rounds = 0
        self._round: list = []
        self._issued = 0

    def is_initialized(self):
        return True

    def nodes(self):
        return self._nodes

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
        self.rounds += 1
        self._round = self._answers.pop(0) if self._answers else []
        # An unanswered probe is one Ray never returns as ready — exactly what a worker that
        # is still starting looks like to the driver.
        return (list(refs) if self._round else []), []

    def get(self, ref):
        return self._round[ref % len(self._round)]


def _install_fleet(monkeypatch, fleet):
    import sys

    _fake_scheduling_strategies(monkeypatch)
    monkeypatch.setitem(sys.modules, "ray", fleet)
    monkeypatch.setattr(hp, "_PROFILES_BY_TOPOLOGY", {})
    monkeypatch.setattr(hp, "_FAILED_ATTEMPTS", {})
    monkeypatch.setattr(hp, "_UNPROBEABLE_WARNED", False)


def test_a_probe_that_misses_while_workers_start_is_retried_not_cached(monkeypatch):
    """The cold-start race must not blind the whole session.

    On an autoscaling fleet the first distributed query routinely beats its workers to the
    line, the probe's short wait expires, and the *old* code stored that emptiness against the
    topology — so every later query planned with default cache sizing even though the workers
    were by then up and would answer immediately. The second call must re-probe and succeed.
    """
    nodes = [_node("a1", 16.0)]
    profile = {"fingerprint": "worker000000", "caches": {"l3": 32 * 1024 * 1024}}
    fleet = _FleetRay(nodes, answers=[[], [profile]])  # miss, then answer
    _install_fleet(monkeypatch, fleet)

    assert hp.cluster_hardware_profiles() == ()  # workers not up yet
    assert fleet.rounds == 1
    again = hp.cluster_hardware_profiles()
    assert again == (profile,), "a transient miss must not be cached"
    assert fleet.rounds == 2


def test_a_successful_probe_is_cached_and_not_repeated(monkeypatch):
    """The success path keeps its memoization — the probe is a round trip per query otherwise."""
    nodes = [_node("a1", 16.0)]
    profile = {"fingerprint": "worker000000", "caches": {"l3": 32 * 1024 * 1024}}
    fleet = _FleetRay(nodes, answers=[[profile], [profile]])
    _install_fleet(monkeypatch, fleet)

    assert hp.cluster_hardware_profiles() == (profile,)
    assert hp.cluster_hardware_profiles() == (profile,)
    assert fleet.rounds == 1, "a successful probe must be memoized, not re-run per query"


def test_a_fleet_that_never_answers_stops_being_asked(monkeypatch):
    """A worker image without the engine must not cost the wait on every query, forever."""
    nodes = [_node("a1", 16.0)]
    fleet = _FleetRay(nodes, answers=[[] for _ in range(10)])
    _install_fleet(monkeypatch, fleet)

    for _ in range(6):
        assert hp.cluster_hardware_profiles() == ()
    assert fleet.rounds == hp._MAX_PROBE_ATTEMPTS, (
        "retries must be bounded, or an unprobeable fleet pays the timeout on every query"
    )


# --- The probe must describe the machines that run the work ----------------------------------

_HEAD = "node:__internal_head__"


def _head_node(node_id, cpus, memory=0.0):
    node = _node(node_id, cpus)
    node["Resources"][_HEAD] = 1.0
    node["Resources"]["memory"] = memory
    return node


def test_the_ray_head_is_not_a_worker_shape():
    """A head node is a different machine class from its workers on essentially every cluster.

    Counting it was not a harmless extra sample. `cluster_l3_cache_bytes` takes the minimum and
    `cluster_storage_class` the worst, so a modest head pinned both; and
    `cluster_worker_fingerprint` requires *agreement*, so it returned `""` — "fall back to the
    driver's key" — almost always. Every coefficient and CPU share measured on the workers was
    then filed where nothing would read it.
    """
    nodes = [_head_node("head", 4.0), _node("w1", 64.0), _node("w2", 64.0)]
    assert hp._representative_node_ids(nodes) == ["w1"]


def test_a_head_only_cluster_still_describes_itself():
    """Survivors-or-nothing: a single-node run is its head and has to be measured."""
    assert hp._representative_node_ids([_head_node("head", 8.0)]) == ["head"]


def test_memory_separates_instance_families_that_share_a_core_count():
    """At sixteen vCPUs a cloud offers 32, 64 and 128 GiB nodes. All three collapsed into one
    shape, so one was probed and the other two were *assumed* to match it — reporting a uniform
    cluster and handing the whole fleet one machine class's cache, scratch device and key."""
    nodes = []
    for name, gib in (("c5", 32), ("m5", 64), ("r5", 128)):
        node = _node(name, 16.0)
        node["Resources"]["memory"] = float(gib << 30)
        nodes.append(node)
    assert len(hp._representative_node_ids(nodes)) == 3


def test_nodes_of_one_instance_type_still_share_a_representative():
    """Memory is bucketed, so the byte-level difference between two nodes of one type — object
    store reservation, kubelet overhead — must not fan the probe back out to O(nodes)."""
    nodes = []
    for i, delta in enumerate((0, 3 << 20, -5 << 20)):
        node = _node(f"w{i}", 16.0)
        node["Resources"]["memory"] = float((64 << 30) + delta)
        nodes.append(node)
    assert len(hp._representative_node_ids(nodes)) == 1


# --- Device memory the workers measured, rather than a label looked up ------------------------


def test_measured_vram_binds_to_the_smallest_reporting_shape(monkeypatch):
    monkeypatch.setattr(
        hp,
        "cluster_hardware_profiles",
        lambda: ({"gpu_memory_bytes": 80 << 30}, {"gpu_memory_bytes": 24 << 30}),
    )
    assert hp.cluster_measured_gpu_memory_bytes() == 24 << 30


def test_measured_vram_reports_unknown_rather_than_zero(monkeypatch):
    """`0` must mean "fall back to the label lookup", never "this fleet has no VRAM"."""
    monkeypatch.setattr(hp, "cluster_hardware_profiles", lambda: ({"gpu_memory_bytes": 0},))
    assert hp.cluster_measured_gpu_memory_bytes() == 0
    monkeypatch.setattr(hp, "cluster_hardware_profiles", tuple)
    assert hp.cluster_measured_gpu_memory_bytes() == 0


def test_a_settled_unprobeable_fleet_can_be_asked_again():
    """After `_MAX_PROBE_ATTEMPTS` the emptiness is taken as settled for the life of the
    process — the right default, and the wrong terminal state for an operator who has just
    fixed the worker image the driver could not talk to."""
    hp._FAILED_ATTEMPTS[("sig",)] = hp._MAX_PROBE_ATTEMPTS
    hp._PROFILES_BY_TOPOLOGY[("other",)] = ({"fingerprint": "abc"},)
    hp.reset_hardware_probe_cache()
    assert not hp._FAILED_ATTEMPTS
    assert not hp._PROFILES_BY_TOPOLOGY
