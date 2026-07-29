"""The policy seams actually hold: every delegation goes through the policy it names.

A seam that is bypassed on one path is worse than no seam, because it is *documented* as
the way to change behaviour. A deployment that supplies its own flow control and gets it
for cold channels but not for the recurring ones it tuned for has no way to discover that
from the outside — the window it asked for is simply not the window in force.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.policies.flow_control import record_shuffle_window
from batcher.config import Config, FlowControlConfig, config_context
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds, SchedulingEnvelope

pytestmark = pytest.mark.unit


class _RecordingFlowControl:
    """A flow-control policy that pins the window and records what it was asked for."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.requests: list[int] = []

    def grant(self, requested: int, ctx: ResourceContext) -> int:
        self.requests.append(requested)
        return self.window


class _RecordingScheduling:
    def __init__(self) -> None:
        self.calls = 0

    def envelope(self, plan, ctx, *, requested_workers, available_bytes) -> SchedulingEnvelope:
        self.calls += 1
        return SchedulingEnvelope(n_tasks=3)


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _plan(credits: int = 8) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="hash",
        bounds=ResourceBounds(m_max_bytes=1024, c_max_credits=credits, n_max_parallelism=4),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def test_a_custom_flow_control_policy_governs_the_cold_path() -> None:
    policy = _RecordingFlowControl(window=11)
    rm = ResourceManager(flow_control=policy)
    assert rm.grant_credits(64) == 11
    assert policy.requests == [64]


def test_a_custom_flow_control_policy_also_governs_the_learned_path() -> None:
    """The bug: a learned window was clamped inline, bypassing the configured policy."""
    hub = _hub()
    record_shuffle_window(hub, "sig-a", 40)

    policy = _RecordingFlowControl(window=11)
    rm = ResourceManager(hub=hub, flow_control=policy)
    granted = rm.grant_credits(64, signature="sig-a")

    assert granted == 11, "the learned path bypassed the policy the manager was given"
    assert policy.requests == [40], "the policy should see the learned window as the request"


def test_the_learned_window_is_the_request_not_the_grant() -> None:
    """A learned window is still clamped: it is a starting point, not an override."""
    hub = _hub()
    record_shuffle_window(hub, "sig-huge", 1_000_000)
    with config_context(Config().replace(flow_control=FlowControlConfig(default_credits=4))):
        rm = ResourceManager(hub=hub)
        granted = rm.grant_credits(1, signature="sig-huge")
    assert 1 <= granted <= 4 * FlowControlConfig().credit_ceiling_factor


def test_a_cold_signature_falls_back_to_the_plan_request() -> None:
    policy = _RecordingFlowControl(window=7)
    rm = ResourceManager(hub=_hub(), flow_control=policy)
    assert rm.grant_credits(23, signature="never-seen") == 7
    assert policy.requests == [23]


def test_the_scheduling_grant_delegates_and_then_clamps_credits() -> None:
    scheduling = _RecordingScheduling()
    flow = _RecordingFlowControl(window=5)
    rm = ResourceManager(scheduling=scheduling, flow_control=flow)

    env = rm.scheduling_envelope(_plan(credits=8))

    assert scheduling.calls == 1
    assert env.n_tasks == 3, "the scheduling policy's own grant is preserved"
    assert env.credits == 5, "credits come from the flow-control authority, not the policy"
    assert flow.requests == [8], "the plan's own widest credit request is what is asked for"


def test_a_custom_admission_policy_is_what_validate_asks() -> None:
    from batcher.plan.resource import FeasibilityVerdict

    class _AlwaysInfeasible:
        def validate(self, plan, ctx) -> FeasibilityVerdict:
            return FeasibilityVerdict(feasible=False, binding_constraint="memory")

    verdict = ResourceManager(admission=_AlwaysInfeasible()).validate(_plan())
    assert verdict.feasible is False


def test_a_custom_memory_estimator_drives_every_sizing_decision() -> None:
    """One estimator, one peak: the spill gate and the reservation must not disagree."""

    class _Fixed:
        def envelope(self, plan, ctx) -> ResourceBounds:
            return ResourceBounds(m_max_bytes=1 << 50, c_max_credits=2, n_max_parallelism=2)

    rm = ResourceManager(memory=_Fixed())
    plan = _plan()
    assert rm.estimated_bytes(plan) == 1 << 50
    assert rm.should_spill(plan) is True
    assert "estimated peak" in (rm.spill_reason(plan) or "")


def test_the_peak_is_computed_once_per_plan() -> None:
    """Three decisions consult it; deriving it three times is how they drift apart."""

    class _Counting:
        def __init__(self) -> None:
            self.calls = 0

        def envelope(self, plan, ctx) -> ResourceBounds:
            self.calls += 1
            return ResourceBounds(m_max_bytes=4096, c_max_credits=1, n_max_parallelism=1)

    estimator = _Counting()
    rm = ResourceManager(memory=estimator)
    plan = _plan()
    rm.estimated_bytes(plan)
    rm.should_spill(plan)
    rm.recommend_spill_partitions(plan)
    rm.recommend_spill_compression(plan)
    assert estimator.calls == 1

    rm.estimated_bytes(_plan(credits=9))  # a different plan object re-derives
    assert estimator.calls == 2


# --- the control plane and the data plane must agree on the default window ----


def test_the_engine_default_credit_window_matches_carbonites_authority() -> None:
    """`DEFAULT_CREDITS` is what runs whenever Carbonite has *not* handed a window down.

    A `ShuffleSession` built without an explicit grant passes `credits=None`, and a
    producer that receives a malformed seed falls back to the same constant — so the two
    defaults drifting apart means precisely the un-granted paths run at the wrong window.
    They did: the engine held 4 while `FlowControlConfig` said 16, and the config's own
    comment records the measurement that retired 4 (2.4 MiB/s vs 7.7 MiB/s on a 50 ms-RTT
    link, because 4 batches do not fill the bandwidth-delay product).
    """
    from batcher._internal.native import engine_or_none

    native = engine_or_none()
    if native is None or not hasattr(native, "default_credits"):
        pytest.skip("engine not built, or it does not expose its default window")
    assert native.default_credits() == FlowControlConfig().default_credits
