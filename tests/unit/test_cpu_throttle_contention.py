"""A quota-throttled container is a contended box, and nothing else could tell.

`oversubscribed` decides whether a family's low CPU utilization was the family's doing or the
machine's, and getting it backwards starts a loop that feeds itself: contention lowers
utilization, low utilization shrinks the per-task reservation, a smaller reservation lets the
scheduler pack more tasks onto the same cores, and that raises contention again.

Its two original signals both miss the clamp that is most common in practice. CFS quota
throttling dequeues a thread at the *end of a period*, so at the default 100 ms period it
yields on the order of ten involuntary switches per core-second against a threshold of two
hundred, and it faults not at all. A container clamped to a third of its quota therefore read
as a perfectly quiet box while every core it was owed sat idle by decree.

These tests pin the third signal, and — just as importantly — pin that it stays quiet on the
cases the median is there to absorb.
"""

from __future__ import annotations

import pytest

from batcher.plan.feedback import (
    CONTENDED_PREEMPTIONS_PER_CORE_SECOND,
    THROTTLED_PERIOD_SHARE,
    oversubscribed,
    preemption_rate,
)

pytestmark = pytest.mark.unit

_MS = 1000.0
_THREADS = 4
#: Involuntary switches a *throttled* (not contended) cgroup produces over the row above:
#: one dequeue per 100 ms period, per thread.
_THROTTLE_SWITCHES = int(10.0 * (_MS / 1000.0) * _THREADS)


def _row(
    throttled: float = 0.0,
    *,
    switches: int = _THROTTLE_SWITCHES,
    faults: int = 0,
    thermal: int = 0,
) -> dict:
    """One feedback row for a family that ran quietly except for the clamp."""
    return {
        "t_op_ms": _MS,
        "threads": _THREADS,
        "invol_ctx_switches": switches,
        "major_faults": faults,
        "cpu_throttled_ratio": throttled,
        "cpu_thermal_events": thermal,
    }


# --- the gap this closes --------------------------------------------------------------------


def test_quota_throttling_is_invisible_to_the_preemption_signal():
    """The premise. If this ever stops holding, the new signal is redundant, not wrong."""
    rate = preemption_rate(_THROTTLE_SWITCHES, _MS, _THREADS)
    assert rate < CONTENDED_PREEMPTIONS_PER_CORE_SECOND / 10


def test_a_clamped_container_reads_as_contended():
    assert oversubscribed([_row(0.45)] * 10) is True


def test_the_same_history_without_the_throttle_field_reads_as_quiet():
    """The before-picture: identical rows, minus the one measurement, take the wrong branch."""
    assert oversubscribed([_row(0.0)] * 10) is False


# --- and stays quiet where it should --------------------------------------------------------


def test_a_family_that_simply_did_not_want_the_cores_is_not_latched():
    assert oversubscribed([_row(0.0)] * 10) is False


def test_brushing_the_quota_occasionally_is_not_a_regime():
    assert oversubscribed([_row(THROTTLED_PERIOD_SHARE / 3)] * 10) is False


def test_one_throttled_run_in_a_clear_history_does_not_latch_the_family():
    """A zero is a real reading, so it counts toward the median. Dropping zeros would take
    the median over the throttled runs alone and latch on a single one."""
    assert oversubscribed([_row(0.9), *([_row(0.0)] * 9)]) is False


def test_a_sustained_majority_does_latch():
    assert oversubscribed([*([_row(0.5)] * 6), *([_row(0.0)] * 4)]) is True


def test_a_history_from_an_engine_that_never_reported_the_field_is_no_evidence():
    rows = [{"t_op_ms": _MS, "threads": _THREADS, "invol_ctx_switches": 40, "major_faults": 0}]
    assert oversubscribed(rows * 10) is False


def test_exactly_at_the_threshold_is_not_over_it():
    assert oversubscribed([_row(THROTTLED_PERIOD_SHARE)] * 10) is False


# --- the original two signals still stand ---------------------------------------------------


def test_preemption_alone_still_detects_contention():
    assert oversubscribed([_row(0.0, switches=4000)] * 5) is True


def test_paging_alone_still_detects_contention():
    assert oversubscribed([_row(0.0, faults=100)] * 5) is True


def test_throttling_is_read_even_when_the_thread_count_is_missing():
    """A throttled cgroup is a fact about the box, so it is evidence even on a row the
    preemption and paging signals must skip for want of a thread count."""
    rows = [{"t_op_ms": _MS, "threads": 0, "cpu_throttled_ratio": 0.45}] * 10
    assert oversubscribed(rows) is True


# --- the field reaches the record ------------------------------------------------------------


def test_the_field_defaults_to_no_evidence():
    from batcher.plan.feedback import OperatorFeedback

    fb = OperatorFeedback(
        op_id=1,
        kind="filter",
        n_actual=1,
        t_op_ms=1.0,
        m_peak_bytes=1,
        selectivity=1.0,
        batch_size=1,
    )
    assert fb.cpu_throttled_ratio == 0.0


def test_the_field_is_persisted_with_the_row():
    """It has to survive into `op_stats`, or the consumer above never sees it."""
    from batcher.metadata.hub import _row_of
    from batcher.plan.feedback import OperatorFeedback

    fb = OperatorFeedback(
        op_id=1,
        kind="filter",
        n_actual=1,
        t_op_ms=1.0,
        m_peak_bytes=1,
        selectivity=1.0,
        batch_size=1,
        cpu_throttled_ratio=0.42,
    )
    assert _row_of(fb)["cpu_throttled_ratio"] == pytest.approx(0.42)


# --- the silicon clamping itself ------------------------------------------------------------
#
# A thermally throttled core is *running*, just slowly. It never exhausts a period it was not
# given, so the quota signal cannot see it, and nothing is preempting it, so the preemption
# signal cannot either. Bare metal only: a virtualized guest is not shown the counters.


def test_a_thermally_throttled_cpu_reads_as_contended():
    assert oversubscribed([_row(thermal=3)] * 10) is True


def test_one_hot_run_in_a_clear_history_does_not_latch_the_family():
    assert oversubscribed([_row(thermal=50), *([_row()] * 9)]) is False


def test_a_cool_box_is_not_latched_by_the_thermal_signal():
    assert oversubscribed([_row()] * 10) is False


def test_thermal_and_quota_are_independent_signals():
    """Either alone is sufficient; neither masks the other."""
    assert oversubscribed([_row(0.45, thermal=0)] * 10) is True
    assert oversubscribed([_row(0.0, thermal=2)] * 10) is True


# --- the probe itself -------------------------------------------------------------------------


def test_the_thermal_counters_report_none_when_unreadable(monkeypatch):
    """A virtualized host does not expose them. `None` is not `0`: one says the reading is
    unavailable, the other says the silicon has never clamped itself."""
    from batcher._internal.hardware import cpu

    monkeypatch.setattr(cpu, "_THERMAL_THROTTLE_GLOB", "/nonexistent/cpu[0-9]*/thermal_throttle")
    assert cpu.cpu_thermal_throttle_count() is None
    assert cpu.cpu_thermal_events() == 0


def _fake_sysfs(root, per_cpu):
    for index, (core, package) in enumerate(per_cpu):
        directory = root / f"cpu{index}" / "thermal_throttle"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "core_throttle_count").write_text(str(core))
        (directory / "package_throttle_count").write_text(str(package))


def test_the_thermal_counters_sum_across_cpus(tmp_path, monkeypatch):
    from batcher._internal.hardware import cpu

    monkeypatch.setattr(cpu, "_THERMAL_THROTTLE_GLOB", f"{tmp_path}/cpu[0-9]*/thermal_throttle")
    _fake_sysfs(tmp_path, [(10, 5), (3, 5)])
    assert cpu.cpu_thermal_throttle_count() == 23


def test_the_first_reading_establishes_a_baseline_rather_than_reporting_since_boot(
    tmp_path, monkeypatch
):
    """A box that throttled during last month's heatwave is not throttling now."""
    from batcher._internal.hardware import cpu

    monkeypatch.setattr(cpu, "_THERMAL_THROTTLE_GLOB", f"{tmp_path}/cpu[0-9]*/thermal_throttle")
    monkeypatch.setattr(cpu, "_THERMAL_BASELINE", None)
    _fake_sysfs(tmp_path, [(1000, 0)])
    assert cpu.cpu_thermal_events() == 0  # baseline, not 1000
    _fake_sysfs(tmp_path, [(1004, 0)])
    assert cpu.cpu_thermal_events() == 4


def test_counters_going_backwards_re_baseline_rather_than_report_negative(tmp_path, monkeypatch):
    from batcher._internal.hardware import cpu

    monkeypatch.setattr(cpu, "_THERMAL_THROTTLE_GLOB", f"{tmp_path}/cpu[0-9]*/thermal_throttle")
    monkeypatch.setattr(cpu, "_THERMAL_BASELINE", None)
    _fake_sysfs(tmp_path, [(100, 0)])
    cpu.cpu_thermal_events()
    _fake_sysfs(tmp_path, [(1, 0)])
    assert cpu.cpu_thermal_events() == 0
    _fake_sysfs(tmp_path, [(3, 0)])
    assert cpu.cpu_thermal_events() == 2
