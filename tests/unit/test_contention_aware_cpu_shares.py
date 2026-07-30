"""Low CPU utilization has two causes with opposite fixes, and the loops now tell them apart.

Both CPU-share loops size a per-task `num_cpus` down when a family's measured utilization is
low. That is right for a family that never wanted the cores and backwards for one whose cores
were taken from it — and it is backwards in a way that feeds itself: a smaller reservation
lets the scheduler pack more tasks onto the contended cores, which lowers utilization, which
shrinks the reservation again. Every step of that loop looks like a correct response to the
measurement.

`plan.feedback.oversubscribed` is the disambiguator, read from preemption and major-fault
counts Core has always recorded and nothing outside the profiler ever consumed.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.dist.adaptive_sizing.sizing import learned_cpu_weight_factor
from batcher.kyber.cpu_shares import load_cpu_utilization
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.feedback import (
    CONTENDED_PREEMPTIONS_PER_CORE_SECOND,
    PAGING_FAULTS_PER_CORE_SECOND,
    OperatorFeedback,
    oversubscribed,
)

pytestmark = pytest.mark.unit


def _row(
    *,
    util: float = 0.2,
    elapsed_ms: float = 1000.0,
    threads: int = 4,
    preemptions: int = 0,
    major_faults: int = 0,
) -> dict:
    return {
        "kind": "filter",
        "cpu_utilization": util,
        "t_op_ms": elapsed_ms,
        "threads": threads,
        "invol_ctx_switches": preemptions,
        "major_faults": major_faults,
    }


def _contended_row(**over: object) -> dict:
    # 4 threads x 1 s = 4 core-seconds, so this is well past the contention threshold.
    switches = int(CONTENDED_PREEMPTIONS_PER_CORE_SECOND * 4 * 3)
    return _row(preemptions=switches, **over)  # type: ignore[arg-type]


def _paging_row(**over: object) -> dict:
    faults = int(PAGING_FAULTS_PER_CORE_SECOND * 4 * 3)
    return _row(major_faults=faults, **over)  # type: ignore[arg-type]


class TestOversubscribed:
    def test_a_quiet_history_is_not_oversubscribed(self) -> None:
        assert oversubscribed([_row() for _ in range(5)]) is False

    def test_no_measurement_is_not_evidence(self) -> None:
        # An engine that reports no thread count says nothing either way, and "no evidence"
        # must read as False so every caller keeps its prior behavior.
        assert oversubscribed([_row(threads=0) for _ in range(5)]) is False
        assert oversubscribed([]) is False

    def test_sustained_preemption_is_oversubscription(self) -> None:
        assert oversubscribed([_contended_row() for _ in range(5)]) is True

    def test_sustained_major_faults_are_oversubscription(self) -> None:
        assert oversubscribed([_paging_row() for _ in range(5)]) is True

    def test_one_contended_run_does_not_latch_the_family(self) -> None:
        # The median, not the max: a single noisy run inside an otherwise clear history is
        # not evidence that the box is oversubscribed.
        rows = [_row() for _ in range(9)] + [_contended_row()]
        assert oversubscribed(rows) is False

    def test_a_first_touch_fault_is_not_paging(self) -> None:
        assert oversubscribed([_row(major_faults=1) for _ in range(5)]) is False


class TestDistributedCpuWeight:
    def _hub(self, rows: list[dict]) -> MetadataHub:
        hub = MetadataHub(InProcessBackend())
        for i, row in enumerate(rows):
            hub.record(
                OperatorFeedback(
                    op_id=i,
                    kind="filter",
                    n_actual=100,
                    t_op_ms=row["t_op_ms"],
                    m_peak_bytes=0,
                    selectivity=1.0,
                    batch_size=1024,
                    cpu_utilization=row["cpu_utilization"],
                    threads=row["threads"],
                    invol_ctx_switches=row["invol_ctx_switches"],
                    major_faults=row["major_faults"],
                )
            )
        return hub

    def test_an_idle_family_still_shrinks_its_reservation(self) -> None:
        hub = self._hub([_row(util=0.2) for _ in range(5)])
        factor = learned_cpu_weight_factor(hub, "filter")
        assert factor is not None and factor < 1.0

    def test_a_contended_family_keeps_its_planned_reservation(self) -> None:
        hub = self._hub([_contended_row(util=0.2) for _ in range(5)])
        # Shrinking here packs more of this family's tasks onto the cores they are already
        # fighting over, which lowers utilization further and shrinks it again.
        assert learned_cpu_weight_factor(hub, "filter") is None

    def test_a_paging_family_keeps_its_planned_reservation(self) -> None:
        hub = self._hub([_paging_row(util=0.2) for _ in range(5)])
        assert learned_cpu_weight_factor(hub, "filter") is None

    def test_contention_does_not_suppress_a_busy_family(self) -> None:
        # Suppression is about which *direction* a low reading justifies. A family measured
        # near the target keeps its full weight either way, so this is not a regression path
        # — but it must still not crash or invert.
        hub = self._hub([_row(util=0.9) for _ in range(5)])
        assert learned_cpu_weight_factor(hub, "filter") == 1.0


class TestKyberCpuShares:
    def _hub(self, rows: list[dict]) -> MetadataHub:
        hub = MetadataHub(InProcessBackend())
        for i, row in enumerate(rows):
            hub.record(
                OperatorFeedback(
                    op_id=i,
                    kind="filter",
                    n_actual=100,
                    t_op_ms=row["t_op_ms"],
                    m_peak_bytes=0,
                    selectivity=1.0,
                    batch_size=1024,
                    cpu_utilization=row["cpu_utilization"],
                    threads=row["threads"],
                    invol_ctx_switches=row["invol_ctx_switches"],
                    major_faults=row["major_faults"],
                )
            )
        return hub

    def _samples(self) -> int:
        return max(1, active_config().optimizer.cost_calibration_min_samples)

    def test_an_idle_family_learns_a_share(self) -> None:
        hub = self._hub([_row(util=0.3) for _ in range(self._samples() + 2)])
        assert load_cpu_utilization(hub).get("filter") == pytest.approx(0.3)

    def test_a_contended_family_keeps_its_static_prior(self) -> None:
        hub = self._hub([_contended_row(util=0.3) for _ in range(self._samples() + 2)])
        assert "filter" not in load_cpu_utilization(hub)

    def test_a_paging_family_keeps_its_static_prior(self) -> None:
        hub = self._hub([_paging_row(util=0.3) for _ in range(self._samples() + 2)])
        assert "filter" not in load_cpu_utilization(hub)

    def test_a_cold_hub_learns_nothing(self) -> None:
        assert load_cpu_utilization(MetadataHub(InProcessBackend())) == {}
        assert load_cpu_utilization(None) == {}
