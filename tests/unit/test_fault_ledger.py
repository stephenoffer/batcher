"""Learning which nodes are bad from outcomes, and the four guards that make it safe.

The ledger is the only health signal that catches a node whose telemetry reads perfectly and
whose tasks all die — a mismatched driver, a half-deployed image, a disk returning `EIO`. It is
also the signal that can do the most damage if it is naive, so each property below is one of
the guards, and each guard exists because the version without it is worse than nothing:

* without **decay**, a node is punished forever for one bad minute and the fleet only shrinks;
* without **hysteresis**, one failure — usually the workload's fault — removes a good node;
* without **probation**, a node quarantined during a blip never comes back;
* without a **blast-radius cap**, a systemic failure condemns the entire cluster in a minute
  and turns a degraded job into a dead one.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import FaultLedger, QuarantinePolicy

pytestmark = pytest.mark.unit


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances by hand, so nothing sleeps."""

    class Clock:
        now = 1000.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

    fake = Clock()
    monkeypatch.setattr("batcher.carbonite.resilience.blocklist.time.monotonic", lambda: fake.now)
    return fake


def _ledger(**overrides) -> FaultLedger:
    policy = QuarantinePolicy(
        failure_threshold=overrides.pop("failure_threshold", 3.0),
        half_life_s=overrides.pop("half_life_s", 300.0),
        cooldown_s=overrides.pop("cooldown_s", 60.0),
        max_cooldown_s=overrides.pop("max_cooldown_s", 900.0),
        max_blocked_fraction=overrides.pop("max_blocked_fraction", 0.34),
        min_targets=overrides.pop("min_targets", 4),
    )
    ledger = FaultLedger(policy, label="node")
    ledger.observe([f"node-{i}" for i in range(overrides.pop("fleet", 10))])
    return ledger


def test_one_failure_never_removes_a_node(clock):
    ledger = _ledger()
    ledger.record_failure("node-0", "worker_lost")
    assert ledger.is_blocked("node-0") is False
    assert ledger.blocked_keys() == ()


def test_repeated_placement_blaming_failures_quarantine_it(clock):
    ledger = _ledger()
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    assert ledger.is_blocked("node-0") is True
    assert ledger.blocked_keys() == ("node-0",)


def test_the_workloads_own_bugs_never_blame_a_node(clock):
    # The failure this guard prevents is specific: quarantining a healthy node over a
    # deterministic bug takes out the next node the retry lands on too, and so on.
    ledger = _ledger()
    for _ in range(20):
        ledger.record_failure("node-0", "application")
        ledger.record_failure("node-0", "device_oom")
    assert ledger.is_blocked("node-0") is False
    # The counters still record them, because an operator asking "what went wrong here" needs
    # to see that this node saw twenty failures even though none of them were its fault.
    assert ledger.health("node-0").failures == 40


def test_failure_weight_decays_so_a_bad_minute_is_not_permanent(clock):
    ledger = _ledger(half_life_s=60.0)
    ledger.record_failure("node-0", "worker_lost")
    ledger.record_failure("node-0", "worker_lost")
    assert ledger.health("node-0").weight == pytest.approx(2.0)
    clock.advance(60.0)
    assert ledger.health("node-0").weight == pytest.approx(1.0)
    # And it reaches exactly zero rather than an ever-smaller float, so a long-lived process
    # does not accumulate a record per node that never quite clears.
    clock.advance(1800.0)
    assert ledger.health("node-0").weight == 0.0


def test_a_quarantine_expires_into_probation_not_into_forgiveness(clock):
    ledger = _ledger(cooldown_s=60.0)
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    assert ledger.is_blocked("node-0") is True
    clock.advance(61.0)
    assert ledger.is_blocked("node-0") is False  # schedulable again
    assert ledger.health("node-0").probing is True


def test_a_failure_on_probation_re_blocks_at_double_the_cooldown(clock):
    ledger = _ledger(cooldown_s=60.0)
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    clock.advance(61.0)
    assert ledger.is_blocked("node-0") is False
    ledger.record_failure("node-0", "storage")
    assert ledger.is_blocked("node-0") is True
    assert ledger.health("node-0").cooldown_s == pytest.approx(120.0)
    assert ledger.health("node-0").offenses == 2


def test_the_doubling_is_capped_so_a_bad_node_is_retried_eventually(clock):
    ledger = _ledger(cooldown_s=60.0, max_cooldown_s=100.0)
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    for _ in range(5):
        clock.advance(1000.0)
        assert ledger.is_blocked("node-0") is False
        ledger.record_failure("node-0", "storage")
    assert ledger.health("node-0").cooldown_s == pytest.approx(100.0)


def test_a_success_on_probation_releases_the_node(clock):
    ledger = _ledger(cooldown_s=60.0)
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    clock.advance(61.0)
    ledger.is_blocked("node-0")
    ledger.record_success("node-0")
    assert ledger.is_blocked("node-0") is False
    assert ledger.health("node-0").weight == 0.0
    assert ledger.health("node-0").blocked_until_s == 0.0


def test_a_success_repays_one_failure_rather_than_wiping_the_slate(clock):
    # A node failing one task in three is a real problem, and a reset-on-success rule would
    # hide it forever behind the two that worked.
    ledger = _ledger()
    ledger.record_failure("node-0", "worker_lost")
    ledger.record_failure("node-0", "worker_lost")
    ledger.record_success("node-0")
    assert ledger.health("node-0").weight == pytest.approx(1.0)


def test_a_systemic_failure_cannot_condemn_the_whole_fleet(clock):
    # Every node failing every task means the cause is the job, not the fleet. Quarantining
    # them all removes the capacity a retry would need and never releases it, because nothing
    # succeeds anywhere to clear the ledger.
    ledger = _ledger(fleet=6, max_blocked_fraction=0.34)
    for node in range(6):
        for _ in range(3):
            ledger.record_failure(f"node-{node}", "storage")
    assert len(ledger.blocked_keys()) <= 2
    assert len(ledger.blocked_keys()) >= 1  # the first offenders are still taken out


def test_hitting_the_cap_is_announced_once_not_once_per_failure(clock):
    from batcher._internal import events

    seen: list[str] = []

    def _sink(event) -> None:
        if event.kind == events.RECOVERY:
            seen.append(event.fields.get("event", ""))

    ledger = _ledger(fleet=6, max_blocked_fraction=0.34)
    unsubscribe = events.subscribe(_sink)
    try:
        # A systemically broken fleet keeps failing, so the capped branch is reached again by
        # every subsequent failure. One copy of the signal is the point of the signal.
        for node in range(6):
            for _ in range(10):
                ledger.record_failure(f"node-{node}", "storage")
    finally:
        unsubscribe()
    capped = seen.count("quarantine_capped")
    # Sixty failures, six nodes. At most one announcement per node, not one per failure.
    assert 1 <= capped <= 6


def test_the_cap_is_not_applied_to_a_fleet_too_small_to_take_a_fraction_of(clock):
    ledger = FaultLedger(QuarantinePolicy(min_targets=4, max_blocked_fraction=0.34))
    ledger.observe(["only-node"])
    for _ in range(3):
        ledger.record_failure("only-node", "storage")
    assert ledger.is_blocked("only-node") is True


def test_the_report_puts_the_worst_offender_first(clock):
    ledger = _ledger()
    ledger.record_failure("node-1", "worker_lost")
    for _ in range(3):
        ledger.record_failure("node-2", "device_fault")
    report = ledger.report()
    assert report[0].key == "node-2"
    assert report[0].reasons == ("device_fault",)


def test_an_unknown_target_reads_as_healthy(clock):
    ledger = _ledger()
    assert ledger.is_blocked("never-seen") is False
    assert ledger.health("never-seen").failures == 0


def test_reset_forgets_everything_for_a_reused_process(clock):
    ledger = _ledger()
    for _ in range(3):
        ledger.record_failure("node-0", "storage")
    ledger.reset()
    assert ledger.blocked_keys() == ()
    assert ledger.is_blocked("node-0") is False
