"""A `resident_inference_pools()` scope must survive a preempted GPU node.

The query-resident pool reuses one model-loaded actor set across a query's stages. It had
no preemption recovery: a node lost after the pool was cached raised `RayError` straight
out and failed the whole query — the exact multi-hour inference job the scope exists to
keep fed. `_run_scoped_pool` now evicts the dead pool and re-runs on a recovering per-call
pool, mirroring what the session-warm path already did.
"""

from __future__ import annotations

import pytest

from _fake_ray import install_fake_ray


@pytest.mark.unit
def test_a_lost_resident_pool_heals_onto_a_recovering_pool(monkeypatch):
    from batcher.dist.executors import map as mapmod

    RayError, _ = install_fake_ray(monkeypatch)

    calls: dict[str, int] = {"resident": 0, "drive": 0, "evict": 0}

    def _resident(plan0, partitions, opts, size, registry):
        calls["resident"] += 1
        raise RayError("GPU node preempted mid-partition")

    def _drive(plan0, partitions, opts, lo, hi, policy):
        calls["drive"] += 1
        return ["recovered"], 0.5, 0.4

    def _evict(plan0, scope):
        calls["evict"] += 1

    monkeypatch.setattr(mapmod, "_run_resident_pool", _resident)
    monkeypatch.setattr(mapmod, "_drive_actor_pool", _drive)
    monkeypatch.setattr(mapmod, "_evict_scoped_pool", _evict)
    # `_run_scoped_pool` imports `recovery_policy` from ray_runtime; stub it so no real
    # config is read. The `_drive` fake ignores the policy anyway.
    import batcher.dist.executors.ray_runtime as rr

    monkeypatch.setattr(rr, "recovery_policy", lambda: None)

    results, _util, _vram = mapmod._run_scoped_pool(None, ["p0"], {}, 1, 2, {})

    assert results == ["recovered"], "a preempted resident pool must fall back, not fail the query"
    assert calls["evict"] == 1, "the dead pool must be evicted from the scope so it is rebuilt"
    assert calls["drive"] == 1


@pytest.mark.unit
def test_a_healthy_resident_pool_is_used_directly(monkeypatch):
    from batcher.dist.executors import map as mapmod

    install_fake_ray(monkeypatch)
    used = {"resident": 0, "drive": 0}

    def _resident(plan0, partitions, opts, size, registry):
        used["resident"] += 1
        return ["ok"], 0.7, 0.6

    def _drive(*a, **k):
        used["drive"] += 1
        raise AssertionError("must not fall back when the resident pool is healthy")

    monkeypatch.setattr(mapmod, "_run_resident_pool", _resident)
    monkeypatch.setattr(mapmod, "_drive_actor_pool", _drive)

    results, _, _ = mapmod._run_scoped_pool(None, ["p0"], {}, 1, 2, {})
    assert results == ["ok"]
    assert used == {"resident": 1, "drive": 0}


@pytest.mark.unit
def test_evict_scoped_pool_removes_the_signature_from_the_scope(monkeypatch):
    from batcher.dist.executors import map as mapmod

    install_fake_ray(monkeypatch)

    class _Fn:
        pass

    # A MapBatches-shaped node so `_pipeline_signature` finds a callable.
    from batcher.plan.logical import MapBatches

    node = MapBatches.__new__(MapBatches)
    object.__setattr__(node, "fn", _Fn())
    object.__setattr__(node, "input", None)
    key = mapmod._pool_key(node, {"num_gpus": 1.0})
    scope = {key: ["actor-a", "actor-b"]}

    mapmod._evict_scoped_pool(node, scope)
    assert key not in scope, "the dead pool's key must be dropped so it is rebuilt"


@pytest.mark.unit
def test_evicting_a_pipeline_drops_every_resource_configuration_it_has(monkeypatch):
    """A pipeline can hold more than one pool: the adaptive loop re-sizes `num_gpus`, and
    the key carries the request. An eviction that matched one exact key would leave the
    other configuration's actors alive, holding the devices the rebuild needs."""
    from batcher.dist.executors import map as mapmod
    from batcher.plan.logical import MapBatches

    install_fake_ray(monkeypatch)

    class _Fn:
        pass

    node = MapBatches.__new__(MapBatches)
    object.__setattr__(node, "fn", _Fn())
    object.__setattr__(node, "input", None)
    whole = mapmod._pool_key(node, {"num_gpus": 1.0})
    packed = mapmod._pool_key(node, {"num_gpus": 0.5})
    assert whole != packed, "the resource request must be part of the key"
    scope = {whole: ["a"], packed: ["b"]}

    mapmod._evict_scoped_pool(node, scope)
    assert scope == {}, f"an evicted pipeline left pools behind: {scope}"


@pytest.mark.unit
def test_a_repacked_pool_replaces_the_old_one_instead_of_growing_past_it(monkeypatch):
    """The deadlock this prevents: the adaptive loop halves `num_gpus`, and a pool keyed
    only by pipeline *grows* to the new replica count — so the previous run's whole-GPU
    actors stay alive holding every device while their replacements wait forever."""
    from batcher.dist.executors import map as mapmod
    from batcher.plan.logical import MapBatches

    install_fake_ray(monkeypatch)

    class _Fn:
        pass

    node = MapBatches.__new__(MapBatches)
    object.__setattr__(node, "fn", _Fn())
    object.__setattr__(node, "input", None)
    monkeypatch.setattr(mapmod, "_new_map_actor", lambda plan0, opts: f"actor@{opts['num_gpus']}")
    monkeypatch.setattr(mapmod, "_healthy_actors", lambda pool: list(pool))

    registry: dict = {}
    whole = mapmod._resident_pool_for(node, {"num_gpus": 1.0}, 4, registry)
    assert whole == ["actor@1.0"] * 4
    packed = mapmod._resident_pool_for(node, {"num_gpus": 0.5}, 8, registry)

    assert packed == ["actor@0.5"] * 8, "the repacked pool must be built fresh, not grown"
    assert len(registry) == 1, f"the old configuration's actors were left holding GPUs: {registry}"
