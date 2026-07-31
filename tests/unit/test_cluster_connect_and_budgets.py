"""Connecting to a Ray cluster, and spending a bounded job's time on the right things.

Two failure modes on an orchestrated cluster, both silent:

The driver and the head come up concurrently, so the first attach routinely fails against
a cluster that is seconds from ready. Falling back to a local Ray there runs a distributed
job on one machine and reports success. When the user *named* the cluster, that is a wrong
answer rather than a degraded one, and it must raise.

And every bounded wait in the scheduler was written for a cluster with no horizon. Under a
lease those waits spend the budget the job needed to do its work: a Slurm allocation with
90 seconds left would spend 180 waiting for autoscaler nodes that arrive after the kill.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from batcher._internal.errors import BackendError
from batcher.config import Config, config_context
from batcher.config.deadline import remaining_budget

_DEADLINE_VARS = ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME")


@pytest.fixture
def clean_env(monkeypatch):
    """No deadline and no cluster address, whatever the host exported."""
    for var in (*_DEADLINE_VARS, "RAY_ADDRESS"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _cfg(**overrides) -> Config:
    base = Config()
    return base.replace(distributed=dataclasses.replace(base.distributed, **overrides))


class TestRemainingBudget:
    """The primitive every deadline-aware wait is built on."""

    def test_no_deadline_leaves_the_wait_alone(self, clean_env):
        assert remaining_budget(180.0) == 180.0

    def test_a_lease_shrinks_the_wait_to_what_is_left(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 60))
        assert 0 < remaining_budget(180.0) <= 60

    def test_a_long_lease_does_not_extend_the_wait(self, clean_env):
        """The budget is a ceiling, never a floor — a week of allocation must not turn a
        two-minute timeout into a week-long hang."""
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 100_000))
        assert remaining_budget(180.0) == 180.0

    def test_the_reserve_is_held_back(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 100))
        assert 0 < remaining_budget(180.0, reserve_s=60.0) <= 40

    def test_no_time_left_yields_no_wait(self, clean_env):
        """Waiting at all is wrong once the reserve exceeds what remains."""
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 30))
        assert remaining_budget(180.0, reserve_s=600.0) == 0.0

    def test_never_negative(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() - 5))
        assert remaining_budget(180.0) == 0.0


class TestPlacementTimeoutRespectsTheLease:
    def test_unbounded_job_keeps_the_configured_timeout(self, clean_env):
        from batcher.dist.executors.ray_runtime.scheduling import _placement_timeout_s

        with config_context(_cfg(placement_timeout_s=60.0)):
            assert _placement_timeout_s() == 60.0

    def test_short_lease_shrinks_the_gang_wait(self, clean_env):
        """Waiting a job's last minute for a gang that would be reclaimed as it formed
        leaves nothing to run in."""
        from batcher.dist.executors.ray_runtime.scheduling import _placement_timeout_s

        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 40))
        with config_context(_cfg(placement_timeout_s=60.0, drain_lead_s=10.0)):
            assert 0 < _placement_timeout_s() <= 30


class TestExplicitAddressIsNotSilentlyDowngraded:
    """The distinction that decides what an unreachable cluster means."""

    def test_configured_address_is_explicit(self, clean_env):
        from batcher.dist.executors.ray_runtime.readiness import _explicit_cluster_address

        with config_context(_cfg(ray_address="ray://head:10001")):
            assert _explicit_cluster_address() == "ray://head:10001"

    def test_env_address_is_explicit(self, clean_env):
        from batcher.dist.executors.ray_runtime.readiness import _explicit_cluster_address

        clean_env.setenv("RAY_ADDRESS", "ray://head:10001")
        assert _explicit_cluster_address() == "ray://head:10001"

    def test_a_merely_detected_cluster_is_not_explicit(self, clean_env):
        """A KubeRay marker is a hint. Degrading to local for it is the behavior that
        keeps a dev run working, so it must stay distinguishable from a named address."""
        from batcher.dist.executors.ray_runtime.readiness import _explicit_cluster_address

        clean_env.setenv("RAY_CLUSTER_NAME", "raycluster-sample")
        assert _explicit_cluster_address() is None

    def test_unreachable_explicit_address_raises_rather_than_going_local(self, clean_env):
        """The silent-wrong-answer case: the user named a cluster, so running single-node
        in its place must be an error, not a success on one machine."""
        from batcher.dist.executors.ray_runtime import readiness

        started_local = []

        class _FakeRay:
            def init(self, **kwargs):
                if kwargs.get("address"):
                    raise ConnectionError("head not reachable")
                started_local.append(kwargs)

            @staticmethod
            def is_initialized() -> bool:
                return False

        cfg = _cfg(ray_address="ray://head:10001", cluster_connect_timeout_s=0.0)
        with config_context(cfg), pytest.raises(BackendError, match="ray://head:10001"):
            readiness._connect_or_fall_back(_FakeRay(), workers=4)
        assert started_local == [], "must not start a local Ray for an explicit address"

    def test_unreachable_detected_cluster_falls_back_to_local(self, clean_env):
        """The degradation that must survive: a workspace whose cluster is down still runs."""
        from batcher.dist.executors.ray_runtime import readiness

        started_local = []

        class _FakeRay:
            def init(self, **kwargs):
                if kwargs.get("address"):
                    raise ConnectionError("head not reachable")
                started_local.append(kwargs)

            @staticmethod
            def is_initialized() -> bool:
                return False

        clean_env.setenv("RAY_CLUSTER_NAME", "raycluster-sample")
        with config_context(_cfg(cluster_connect_timeout_s=0.0)):
            readiness._connect_or_fall_back(_FakeRay(), workers=4)
        assert len(started_local) == 1
        assert "address" not in started_local[0]


class TestAttachRetry:
    """A head that is still coming up is 'not yet', not 'not there'."""

    def test_a_head_that_answers_on_the_second_try_attaches(self, clean_env):
        from batcher.dist.executors.ray_runtime import readiness

        attempts = []

        class _FakeRay:
            def init(self, **kwargs):
                attempts.append(kwargs)
                if len(attempts) < 3:
                    raise ConnectionError("head still starting")

            @staticmethod
            def is_initialized() -> bool:
                return False

        with config_context(_cfg(cluster_connect_timeout_s=10.0)):
            assert readiness._attach_with_retry(_FakeRay(), address="auto") is True
        assert len(attempts) == 3

    def test_a_dead_cluster_gives_up_within_the_window(self, clean_env):
        from batcher.dist.executors.ray_runtime import readiness

        class _FakeRay:
            def init(self, **kwargs):
                raise ConnectionError("no cluster")

            @staticmethod
            def is_initialized() -> bool:
                return False

        with config_context(_cfg(cluster_connect_timeout_s=1.0)):
            started = time.monotonic()
            assert readiness._attach_with_retry(_FakeRay(), address="auto") is False
            assert time.monotonic() - started < 10.0

    def test_zero_timeout_tries_exactly_once(self, clean_env):
        """The documented opt-out restores the old single-attempt behavior."""
        from batcher.dist.executors.ray_runtime import readiness

        attempts = []

        class _FakeRay:
            def init(self, **kwargs):
                attempts.append(kwargs)
                raise ConnectionError("no cluster")

            @staticmethod
            def is_initialized() -> bool:
                return False

        with config_context(_cfg(cluster_connect_timeout_s=0.0)):
            assert readiness._attach_with_retry(_FakeRay(), address="auto") is False
        assert len(attempts) == 1

    def test_a_lease_bounds_the_connect_retry(self, clean_env):
        """A job with no time left must not spend it retrying a cluster it cannot use."""
        from batcher.dist.executors.ray_runtime import readiness

        attempts = []

        class _FakeRay:
            def init(self, **kwargs):
                attempts.append(kwargs)
                raise ConnectionError("no cluster")

            @staticmethod
            def is_initialized() -> bool:
                return False

        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 5))
        with config_context(_cfg(cluster_connect_timeout_s=600.0, drain_lead_s=120.0)):
            started = time.monotonic()
            assert readiness._attach_with_retry(_FakeRay(), address="auto") is False
            assert time.monotonic() - started < 5.0
        assert len(attempts) == 1

    def test_a_concurrent_attach_is_not_treated_as_failure(self, clean_env):
        """Another thread winning the init race leaves us attached, which is success."""
        from batcher.dist.executors.ray_runtime import readiness

        class _FakeRay:
            def init(self, **kwargs):
                raise ConnectionError("lost the race")

            @staticmethod
            def is_initialized() -> bool:
                return True

        with config_context(_cfg(cluster_connect_timeout_s=10.0)):
            assert readiness._attach_with_retry(_FakeRay(), address="auto") is True
