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
    sig = mapmod._pipeline_signature(node)
    scope = {sig: ["actor-a", "actor-b"]}

    mapmod._evict_scoped_pool(node, scope)
    assert sig not in scope, "the dead pool's signature must be dropped so it is rebuilt"
