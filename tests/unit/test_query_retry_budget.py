"""One retry budget per distributed query, not one per barrier.

The budget exists to make a job on a broken fleet fail on its first clear error instead of
retrying for hours. Built afresh per barrier, a query with K map and write barriers got K
floors, and the actor-pool loop drew on no budget at all — so the bound the docstrings
promised was multiplied by the number of stages. These drive the real gather loops against a
fake Ray inside a query scope and count what the scope as a whole may retry.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import RecoveryPolicy
from batcher.config import Config, config_context

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def _query(floor: int):
    """A fresh query scope with a retry floor of `floor` and no size-proportional allowance.

    The scope is entered by setting the plan id directly: minting one goes through
    `flight_worker`, which needs a genuine `ray.remote` and is not what is under test. The id
    is unique per call, as a real one is, so no test inherits another's spent budget.
    """
    import uuid

    from batcher.dist.fleet import plan_id

    base = Config()
    cfg = base.replace(
        fault_tolerance=dataclasses.replace(
            base.fault_tolerance, retry_budget_floor=floor, retry_budget_fraction=0.0
        )
    )
    token = plan_id._QUERY_PLAN_ID.set(uuid.uuid4().int >> 65)
    try:
        with config_context(cfg):
            yield
    finally:
        plan_id._QUERY_PLAN_ID.reset(token)


def _flaky(RayError, failures: int):
    """A `submit` whose partition 0 is lost to preemption `failures` times, then lands."""
    seen = {"n": 0}

    def submit(idx):
        seen["n"] += 1
        if seen["n"] <= failures:

            def lost():
                raise RayError("preempted")

            return lost
        return lambda: idx

    return submit


def test_two_barriers_in_one_query_share_one_allowance(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
    policy = RecoveryPolicy(max_attempts=10)

    with _query(floor=2):
        # The first stage spends the query's whole allowance recovering...
        assert gather_map_results(_flaky(RayError, 2), 1, policy) == [0]
        # ...so the second stage's first loss is raised rather than retried. With a budget
        # per barrier it got a fresh floor of two and quietly retried.
        with pytest.raises(RayError):
            gather_map_results(_flaky(RayError, 1), 1, policy)


def test_queries_do_not_share_a_budget(monkeypatch):
    from batcher.dist.executors.ray_runtime.policies import retry_budget
    from batcher.dist.fleet import plan_id

    install_fake_ray(monkeypatch)
    with _query(floor=2):
        first = retry_budget()
        assert retry_budget() is first
        token = plan_id._QUERY_PLAN_ID.set(-1)  # any other query
        try:
            assert retry_budget() is not first
        finally:
            plan_id._QUERY_PLAN_ID.reset(token)
    # Outside any query there is nothing to share with.
    assert retry_budget() is not retry_budget()


def test_the_actor_pool_draws_on_the_query_budget(monkeypatch):
    """The second retry loop (`_drive_actor_pool`) consulted no budget at all."""
    from batcher.dist.executors import map as mapmod

    RayError, _ = install_fake_ray(monkeypatch)
    lost: list[str] = []

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return lambda: self._fn(*args, **kwargs)

    class _DyingActor:
        def __init__(self) -> None:
            self.run = _Remote(self._run)
            self.gpu_stats = _Remote(lambda: None)

        def _run(self, part, idx=0):
            lost.append(part)
            raise RayError("actor preempted")

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0, write_spec=None):
            return _DyingActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)

    with _query(floor=2), pytest.raises(RayError):
        mapmod._drive_actor_pool(
            plan0=None,
            partitions=["p0"],
            opts={},
            min_size=1,
            max_size=1,
            policy=RecoveryPolicy(max_attempts=50),
        )
    # The first attempt plus the two retries the query's floor allows, not 51.
    assert len(lost) == 3
