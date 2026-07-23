"""GPU as a first-class scheduled, placed, and budgeted resource.

These pin four properties that decide whether a large GPU pool starts at all:

* the inference actor pool is **gang-scheduled** into a placement group, so a pool
  either gets all its bundles or degrades explicitly (it cannot half-place and stall);
* a `gpu_collective` stage therefore actually reaches `STRICT_PACK`, which was
  unreachable while the inference path built no placement group;
* placement failure is **reported**, not silently downgraded to default scheduling;
* the pool health check is **concurrent** — a 200-actor pool after a node loss costs
  one bounded wait, not 200 serial 10s timeouts.

Everything runs against a fake Ray (the `test_map_inflight.py` pattern): no GPU and no
real cluster.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from batcher.carbonite.resilience import RecoveryPolicy
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return lambda: self._fn(*args, **kwargs)


class _FakeActor:
    """A stand-in for a `_MapActor` handle, recording the options it was built with."""

    def __init__(self, opts: dict | None = None) -> None:
        self.opts = dict(opts or {})
        self.killed = False
        self.run = _Remote(lambda part, idx=0: [f"out-{part}"])
        self.gpu_stats = _Remote(lambda: 0.4)


class _FakeRay(types.ModuleType):
    pass


def _install_fake_ray(monkeypatch) -> types.ModuleType:
    """A fake `ray` sufficient for `_drive_actor_pool` / `_healthy_actors`."""
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError

    ray_mod = _FakeRay("ray")
    ray_mod.exceptions = exc
    ray_mod.wait_calls = []
    ray_mod.get_calls = []

    def _wait(refs, num_returns=1, timeout=None):
        refs = list(refs)
        ray_mod.wait_calls.append({"n": len(refs), "num_returns": num_returns, "timeout": timeout})
        k = min(num_returns, len(refs))
        return refs[:k], refs[k:]

    def _get(ref, timeout=None):
        ray_mod.get_calls.append(timeout)
        if isinstance(ref, list):
            return [r() for r in ref]
        return ref()

    ray_mod.wait = _wait
    ray_mod.get = _get
    ray_mod.kill = lambda actor: setattr(actor, "killed", True)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray_mod


class _FakePG:
    def __init__(self, bundles: int, strategy: str) -> None:
        self.bundles = bundles
        self.strategy = strategy
        self.removed = False


def _patch_actor_class(monkeypatch, mapmod, built: list):
    """Replace `_MapActor` so every spawn records the `.options(...)` it received."""

    class _Cls:
        def __init__(self, opts: dict | None = None) -> None:
            self._opts = opts

        def options(self, **kw):
            return _Cls(kw)

        def remote(self, plan0, write_spec=None):
            actor = _FakeActor(self._opts)
            actor.plan0 = plan0
            actor.write_spec = write_spec
            built.append(actor)
            return actor

    monkeypatch.setattr(mapmod, "_MapActor", _Cls())


@pytest.fixture
def mapmod():
    from batcher.dist.executors import map as _mapmod

    return _mapmod


# --------------------------------------------------------------------------------------
# Item 4: the inference actor pool must be gang-scheduled into a placement group.
# --------------------------------------------------------------------------------------


def test_actor_pool_is_placed_in_a_placement_group(monkeypatch, mapmod):
    """A multi-actor inference pool MUST reserve a placement group and bind each actor to
    a bundle. Without this the pool can half-place and stall forever."""
    _install_fake_ray(monkeypatch)
    built: list = []
    _patch_actor_class(monkeypatch, mapmod, built)

    created: list = []

    def _create(workers, env):
        pg = _FakePG(workers, "SPREAD")
        created.append(pg)
        return pg

    monkeypatch.setattr(mapmod, "create_worker_placement", _create, raising=False)
    monkeypatch.setattr(
        mapmod,
        "placement_actor_options",
        lambda pg, i, base=None: {**(base or {}), "scheduling_strategy": ("pg", i)},
        raising=False,
    )
    monkeypatch.setattr(mapmod, "release_placement", lambda pg: None, raising=False)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 1)

    parts = ["p0", "p1", "p2", "p3"]
    results, _, _ = mapmod._drive_actor_pool(
        object(), parts, {"num_gpus": 1.0}, 4, 4, RecoveryPolicy()
    )

    assert results == [["out-p0"], ["out-p1"], ["out-p2"], ["out-p3"]]
    assert created, "no placement group was created for the inference actor pool"
    assert created[0].bundles == 4
    strategies = [a.opts.get("scheduling_strategy") for a in built]
    assert all(s is not None for s in strategies), (
        f"actors were spawned without a placement-group binding: {strategies}"
    )
    # Each actor gets its OWN bundle; two actors sharing a bundle oversubscribes it.
    assert sorted(s[1] for s in strategies) == [0, 1, 2, 3]


def test_placement_group_is_released_after_the_pool_finishes(monkeypatch, mapmod):
    """The placement group is a cluster-wide reservation; leaking it strands the GPUs."""
    _install_fake_ray(monkeypatch)
    _patch_actor_class(monkeypatch, mapmod, [])

    pg = _FakePG(2, "SPREAD")
    released: list = []
    monkeypatch.setattr(mapmod, "create_worker_placement", lambda w, e: pg, raising=False)
    monkeypatch.setattr(
        mapmod, "placement_actor_options", lambda p, i, base=None: dict(base or {}), raising=False
    )
    monkeypatch.setattr(mapmod, "release_placement", released.append, raising=False)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 1)

    mapmod._drive_actor_pool(object(), ["a", "b"], {}, 2, 2, RecoveryPolicy())
    assert released == [pg], "the placement group was not released"


# --------------------------------------------------------------------------------------
# Item 5: a `gpu_collective` stage must actually reach STRICT_PACK.
# --------------------------------------------------------------------------------------


def test_gpu_collective_envelope_reaches_strict_pack(monkeypatch, mapmod):
    """A multi-GPU model needing NCCL co-location expresses it via `gpu_collective`.

    The strategy is resolved inside `create_worker_placement`, so the only way the
    inference path can honor it is by calling that function at all.
    """
    _install_fake_ray(monkeypatch)
    _patch_actor_class(monkeypatch, mapmod, [])

    from batcher.dist.executors.ray_runtime import scheduling

    seen: list = []

    def _create(workers, env):
        seen.append(scheduling._resolve_placement_strategy(env))
        return None

    monkeypatch.setattr(mapmod, "create_worker_placement", _create, raising=False)
    monkeypatch.setattr(mapmod, "release_placement", lambda pg: None, raising=False)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 1)

    env = SchedulingEnvelope(num_gpus=1.0, n_tasks=2, gpu_collective=True)
    monkeypatch.setattr(mapmod, "current_envelope", lambda: env, raising=False)

    mapmod._drive_actor_pool(object(), ["a", "b"], {}, 2, 2, RecoveryPolicy())

    assert seen == ["STRICT_PACK"], f"gpu_collective did not reach STRICT_PACK: {seen}"


# --------------------------------------------------------------------------------------
# Item 6 (caller side): placement failure must not degrade silently.
# --------------------------------------------------------------------------------------


def test_placement_failure_is_reported(monkeypatch, mapmod, caplog):
    """`create_worker_placement` returns None on timeout and the pool falls back to default
    scheduling. That fallback is a real capacity signal and MUST be logged."""
    _install_fake_ray(monkeypatch)
    _patch_actor_class(monkeypatch, mapmod, [])

    monkeypatch.setattr(mapmod, "create_worker_placement", lambda w, e: None, raising=False)
    monkeypatch.setattr(mapmod, "release_placement", lambda pg: None, raising=False)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 1)

    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        mapmod._drive_actor_pool(object(), ["a", "b", "c"], {}, 3, 3, RecoveryPolicy())

    assert any("placement" in r.message.lower() for r in caplog.records), (
        f"placement failure degraded silently; records={[r.message for r in caplog.records]}"
    )


def test_placement_error_does_not_take_down_the_stage(monkeypatch, mapmod, caplog):
    """Reserving the group is an optimization, never a correctness requirement.

    A placement API that raises (an older Ray, a cluster without the placement service)
    must degrade to default scheduling and still produce the full result — failing the
    whole inference stage on it would be strictly worse than the silent fallback it
    replaced.
    """
    _install_fake_ray(monkeypatch)
    _patch_actor_class(monkeypatch, mapmod, [])

    def _boom(workers, env):
        raise RuntimeError("no placement service")

    monkeypatch.setattr(mapmod, "create_worker_placement", _boom, raising=False)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 1)

    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        results, _, _ = mapmod._drive_actor_pool(object(), ["a", "b"], {}, 2, 2, RecoveryPolicy())

    assert results == [["out-a"], ["out-b"]]
    assert any("no placement service" in r.message for r in caplog.records)


# --------------------------------------------------------------------------------------
# Item 7: the pool health check must be concurrent.
# --------------------------------------------------------------------------------------


def test_healthy_actors_probes_concurrently(monkeypatch, mapmod):
    """A 200-actor pool must cost ONE bounded wait, not 200 serial 10s timeouts."""
    ray_mod = _install_fake_ray(monkeypatch)
    pool = [_FakeActor() for _ in range(200)]

    alive = mapmod._healthy_actors(pool)

    assert alive == pool
    assert len(ray_mod.wait_calls) == 1, (
        f"health check is serial: {len(ray_mod.wait_calls)} waits for 200 actors"
    )
    call = ray_mod.wait_calls[0]
    assert call["n"] == 200 and call["num_returns"] == 200
    assert call["timeout"] is not None, "the concurrent probe must stay bounded"
    # No per-actor blocking timeout may remain.
    assert not [t for t in ray_mod.get_calls if t], "per-actor blocking timeouts remain"


def test_healthy_actors_drops_dead_actors(monkeypatch, mapmod):
    """Concurrency must not cost the liveness semantics: unreachable actors are dropped
    and killed, survivors are kept in order."""
    ray_mod = _install_fake_ray(monkeypatch)
    good_a, bad, good_b = _FakeActor(), _FakeActor(), _FakeActor()
    bad.gpu_stats = _Remote(lambda: (_ for _ in ()).throw(RuntimeError("unreachable")))

    # The dead actor's probe never becomes ready.
    def _wait(refs, num_returns=1, timeout=None):
        refs = list(refs)
        ray_mod.wait_calls.append({"n": len(refs), "num_returns": num_returns, "timeout": timeout})
        ready = [r for r in refs if r not in _never]
        return ready, [r for r in refs if r in _never]

    probes: dict = {}
    _never: set = set()

    real_get = ray_mod.get

    def _get(ref, timeout=None):
        ray_mod.get_calls.append(timeout)
        return real_get(ref, timeout=timeout) if not isinstance(ref, list) else real_get(ref)

    ray_mod.wait = _wait
    ray_mod.get = _get

    # Mark the bad actor's ref as never-ready by identity once created.
    orig_remote = bad.gpu_stats.remote

    def _bad_remote(*a, **k):
        ref = orig_remote(*a, **k)
        _never.add(ref)
        probes[ref] = bad
        return ref

    bad.gpu_stats.remote = _bad_remote

    alive = mapmod._healthy_actors([good_a, bad, good_b])

    assert alive == [good_a, good_b]
    assert bad.killed is True


# --------------------------------------------------------------------------------------
# Item 3: Carbonite must budget GPU, not hardcode 0.
# --------------------------------------------------------------------------------------


def test_scheduling_envelope_budgets_gpu_for_a_gpu_plan():
    """Carbonite hardcoded `num_gpus=0.0`, so GPU demand reached Ray only via the
    `map_batches(num_gpus=)` tag and was never budgeted against cluster capacity."""
    from batcher.carbonite.policies import DefaultSchedulingPolicy

    assert hasattr(DefaultSchedulingPolicy, "gpu_envelope"), (
        "Carbonite has no GPU envelope; `num_gpus` is still hardcoded to 0.0"
    )


def test_gpu_envelope_clamps_to_cluster_gpu_inventory():
    """A GPU grant must be clamped to the GPUs that actually exist."""
    from batcher.carbonite.policies import DefaultSchedulingPolicy

    env = DefaultSchedulingPolicy.gpu_envelope(num_gpus=0.25, n_tasks=64, gpu_count=4)
    # Fractional requests are preserved (packing 4 actors per device is the point) ...
    assert env.num_gpus == 0.25
    # ... and 4 GPUs at 0.25 each = 16 concurrent tasks, so the fan-out clamps there.
    assert env.n_tasks == 16

    # A fan-out that already fits is left alone.
    assert DefaultSchedulingPolicy.gpu_envelope(num_gpus=1.0, n_tasks=2, gpu_count=8).n_tasks == 2

    # No GPUs visible -> no GPU grant, otherwise the task pends forever.
    none = DefaultSchedulingPolicy.gpu_envelope(num_gpus=1.0, n_tasks=8, gpu_count=0)
    assert none.num_gpus == 0.0
