"""Wall-clock termination deadlines: the batch-scheduler half of preemption.

A Slurm allocation is not reclaimed with a notice — it simply ends at a time fixed when
the job was submitted, and every process in it is killed. These tests pin the three
things that makes safe: reading the deadline out of the environment without being fooled
by the sentinel an unlimited job exports, draining a configurable lead time before it,
and trapping the early-warning signal an HPC job is submitted with.

They also pin the CPU-allocation cap, which is the same class of bug in a different
dimension: on a Slurm node without cgroup task confinement the affinity mask reports
every core on the box, so sizing to it fans a job out far past what it was granted.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import time

import pytest

from batcher._internal.hardware.cpu import available_cpu_count
from batcher.carbonite.resilience import PreemptionMonitor, termination_probe
from batcher.config import Config, config_context, deadline
from batcher.config.deadline import (
    DEADLINE_HORIZON_S,
    DEADLINE_PAST_GRACE_S,
    deadline_epoch_s,
    deadline_probe,
    seconds_remaining,
)
from batcher.config.profiles import detect_leased_allocation, detect_spot_environment

# Every deadline env var the reader consults, so a test can guarantee a clean slate rather
# than depend on whichever of them the host scheduler happens to have exported.
_DEADLINE_VARS = ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME")

# The relative form, for the schedulers that publish a wall-clock *limit* and never the moment
# it expires — which is every one of them except Slurm.
_RELATIVE_VAR = "BATCHER_DEADLINE_SECONDS"


@pytest.fixture
def clean_env(monkeypatch):
    """No deadline in the environment, whatever the host scheduler set."""
    for var in (*_DEADLINE_VARS, _RELATIVE_VAR):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def just_started(clean_env):
    """Pin the relative lease's base to now, so the assertion does not depend on suite age.

    The base is the process's real start time, which in a fifteen-minute test run is fifteen
    minutes ago — so a 600-second lease has genuinely expired and the reading is correct while
    the assertion about it would not be. Pinning it is what makes the test about the *reader*.
    """
    clean_env.setattr(deadline, "_PROCESS_START_S", time.time())
    return clean_env


def test_a_relative_lease_reaches_the_same_drain_path(just_started):
    # PBS, LSF, Grid Engine and HTCondor all know a job's wall-clock limit and none of them
    # export the moment it ends, so without a relative form none of them can drain at all.
    just_started.setenv(_RELATIVE_VAR, "600")
    remaining = seconds_remaining()
    assert remaining is not None
    assert 500 < remaining <= 600
    assert detect_leased_allocation() is True


def test_a_relative_lease_is_measured_from_when_the_process_started(clean_env):
    # Not from when this module was imported. A driver that imports Batcher some way into its
    # life would otherwise be told it has the whole lease left, and would drain late or not at
    # all -- the failure the deadline path exists to prevent.
    clean_env.setattr(deadline, "_PROCESS_START_S", time.time() - 500)
    clean_env.setenv(_RELATIVE_VAR, "600")
    remaining = seconds_remaining()
    assert remaining is not None
    assert 90 < remaining <= 100


def test_an_expired_relative_lease_reads_as_no_time_left(just_started):
    # Inside the kill grace period, which is precisely when draining matters most.
    just_started.setattr(deadline, "_PROCESS_START_S", time.time() - 900)
    just_started.setenv(_RELATIVE_VAR, "600")
    assert seconds_remaining() == 0.0


def test_an_absolute_deadline_outranks_a_relative_one(just_started):
    # The absolute form is exact; the relative one is measured from process start, which is
    # only ever at or after the moment the launcher meant. Where both are present the exact
    # one is the answer.
    just_started.setenv(_RELATIVE_VAR, "10000")
    just_started.setenv("SLURM_JOB_END_TIME", str(time.time() + 100))
    assert seconds_remaining() <= 100


def test_no_deadline_reads_as_unbounded(clean_env):
    assert deadline_epoch_s() is None
    assert seconds_remaining() is None
    assert detect_leased_allocation() is False


@pytest.mark.parametrize("var", _DEADLINE_VARS)
def test_deadline_read_from_either_source(clean_env, var):
    """Both the explicit override and Slurm's own export are honored."""
    clean_env.setenv(var, str(time.time() + 600))
    assert deadline_epoch_s() is not None
    remaining = seconds_remaining()
    assert remaining is not None
    assert 500 < remaining <= 600


def test_explicit_override_wins_over_slurm(clean_env):
    """A launcher that sets the deadline itself overrides what Slurm advertised."""
    clean_env.setenv("SLURM_JOB_END_TIME", str(time.time() + 10_000))
    clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 100))
    assert seconds_remaining() <= 100


def test_unlimited_slurm_sentinel_is_not_a_deadline(clean_env):
    """Slurm exports a saturated end time for an unlimited job rather than omitting it.

    Reading that as a real deadline would report every unlimited HPC job as leased and hand
    it the hardened spot profile it does not need.
    """
    clean_env.setenv("SLURM_JOB_END_TIME", "4294967294")
    assert deadline_epoch_s() is None
    assert detect_leased_allocation() is False


def test_relative_seconds_mistake_is_rejected(clean_env):
    """Exporting a *relative* duration where an epoch was meant must not pin a drain on.

    ``3600`` as an absolute epoch is 1970, which would otherwise read as "expired" forever.
    """
    clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", "3600")
    assert deadline_epoch_s() is None


def test_malformed_deadline_degrades_to_unbounded(clean_env):
    clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", "not-a-number")
    assert deadline_epoch_s() is None


def test_deadline_inside_the_kill_grace_still_counts(clean_env):
    """A deadline that just passed is the kill grace window, not noise.

    This is the moment draining matters most, so it must report zero time left rather than
    fall back to "unbounded" and let the fleet keep taking work.
    """
    clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() - 5))
    assert deadline_epoch_s() is not None
    assert seconds_remaining() == 0.0


def test_grace_and_horizon_bounds_are_ordered():
    """The accepted window must be a real window — a past grace far shorter than the
    forward horizon, so a stale export is rejected long before a genuine lease is."""
    assert 0 < DEADLINE_PAST_GRACE_S < DEADLINE_HORIZON_S


class TestDrainLeadWindow:
    """The probe fires once the deadline is within the lead time, and not before."""

    def test_outside_the_lead_window_is_not_draining(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 600))
        assert deadline_probe(120.0)() is False

    def test_inside_the_lead_window_is_draining(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 60))
        assert deadline_probe(120.0)() is True

    def test_no_deadline_never_drains(self, clean_env):
        """A cluster with no lease must behave exactly as it did before."""
        assert deadline_probe(120.0)() is False
        assert deadline_probe(10**9)() is False

    def test_zero_lead_drains_only_once_expired(self, clean_env):
        clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 30))
        assert deadline_probe(0.0)() is False


def test_termination_probe_honors_the_configured_lead(clean_env):
    """`termination_probe` reads `drain_lead_s`, so the same deadline drains or does not
    depending only on the configured budget."""
    clean_env.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + 300))

    def _lead(seconds: float) -> Config:
        return Config().replace(
            distributed=dataclasses.replace(Config().distributed, drain_lead_s=seconds)
        )

    with config_context(_lead(60.0)):
        assert termination_probe() is False
    with config_context(_lead(600.0)):
        assert termination_probe() is True


def test_leased_allocation_selects_the_spot_profile(clean_env):
    """A time-limited allocation is preemptible in every way that matters to the engine, so
    it must reach the same hardened budgets a spot node gets."""
    assert detect_spot_environment() is False
    clean_env.setenv("SLURM_JOB_END_TIME", str(time.time() + 3600))
    assert detect_leased_allocation() is True
    assert detect_spot_environment() is True


class TestDrainSignals:
    """Slurm's early warning is `SIGUSR1`; Kubernetes eviction and Slurm's time limit are
    `SIGTERM`. Missing either means the drain never runs."""

    @pytest.mark.parametrize("signame", ["SIGTERM", "SIGUSR1"])
    def test_signal_triggers_the_drain(self, signame):
        signum = getattr(signal, signame, None)
        if signum is None:  # pragma: no cover - platform without the signal
            pytest.skip(f"{signame} not available on this platform")
        drained: list[str] = []
        # Install a benign prior handler before starting the monitor. The monitor chains to
        # whatever it displaced, and once Ray has been initialized in this process that is
        # Ray's own SIGTERM handler, which calls `sys.exit` — correct in production (drain,
        # then let Ray shut the worker down) and fatal to a test that raises the signal for
        # real. Pinning the prior handler keeps this about the monitor.
        previous = signal.getsignal(signum)
        signal.signal(signum, lambda *_: None)
        monitor = PreemptionMonitor(probe=lambda: False, poll_interval_s=3600)
        monitor.on_drain(lambda: drained.append(signame))
        monitor.start()
        try:
            assert monitor.is_draining() is False
            signal.raise_signal(signum)
            assert monitor.is_draining() is True
            assert drained == [signame]
        finally:
            monitor.stop()
            signal.signal(signum, previous)

    def test_prior_handler_still_runs(self):
        """Trapping must chain, not replace — a job that installed its own checkpoint
        handler would otherwise silently stop checkpointing."""
        seen: list[int] = []
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
        monitor = PreemptionMonitor(probe=lambda: False, poll_interval_s=3600)
        monitor.start()
        try:
            signal.raise_signal(signal.SIGTERM)
            assert monitor.is_draining() is True
            assert seen == [int(signal.SIGTERM)]
        finally:
            monitor.stop()
            signal.signal(signal.SIGTERM, previous)

    def test_stop_restores_the_prior_handlers(self):
        """A monitor that leaked its trap would keep draining a process that stopped it."""
        before = {
            name: signal.getsignal(getattr(signal, name))
            for name in ("SIGTERM", "SIGUSR1")
            if getattr(signal, name, None) is not None
        }
        monitor = PreemptionMonitor(probe=lambda: False, poll_interval_s=3600)
        monitor.start()
        monitor.stop()
        for name, handler in before.items():
            assert signal.getsignal(getattr(signal, name)) is handler


class TestSlurmCpuAllocation:
    """`available_cpu_count` must never exceed what Slurm granted on this node."""

    @pytest.fixture(autouse=True)
    def _no_slurm(self, monkeypatch):
        for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            monkeypatch.delenv(var, raising=False)
        self.monkeypatch = monkeypatch

    def test_allocation_caps_the_count(self):
        self.monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
        assert available_cpu_count() == 1

    def test_cpus_per_task_wins_over_the_node_share(self):
        """The per-task grant is the tighter, more specific bound of the two."""
        self.monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
        self.monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        assert available_cpu_count() <= 2

    def test_allocation_never_raises_the_count(self):
        """Slurm is an upper bound, not a grant of cores the cgroup or cpuset denies."""
        self.monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        unbounded = available_cpu_count()
        self.monkeypatch.setenv("SLURM_CPUS_PER_TASK", "100000")
        assert available_cpu_count() == unbounded

    @pytest.mark.parametrize("bad", ["", "not-a-number", "0", "-1"])
    def test_unparseable_allocation_is_ignored(self, bad):
        """A junk value must degrade to the old behavior, never to a silent single-core run."""
        self.monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        unbounded = available_cpu_count()
        self.monkeypatch.setenv("SLURM_CPUS_PER_TASK", bad)
        assert available_cpu_count() == unbounded

    @pytest.mark.parametrize(("expansion", "granted"), [("4(x2)", 4), ("8,4(x2)", 4), ("2(x8)", 2)])
    def test_a_heterogeneous_expansion_binds_rather_than_being_ignored(self, expansion, granted):
        """A run-length expansion used to be classified unparseable and dropped, leaving the job
        with **no** Slurm bound at all.

        That was the wrong half of the trade. The stated worry was a half-parse producing a
        silent single-core run — but the failure it chose instead is the one this bound exists
        to prevent: with no bound the job sizes to the affinity mask, which on an HPC node
        without `task/cgroup` confinement is every core on a shared machine. The smallest grant
        in the expansion is taken, since which entry describes *this* node is not derivable from
        the variable, and under-parallelizing costs throughput where over-parallelizing on the
        node that got the small grant is what gets a job killed.
        """
        self.monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
        self.monkeypatch.setenv("SLURM_CPUS_PER_TASK", expansion)
        assert available_cpu_count() == min(granted, os.cpu_count() or granted)
