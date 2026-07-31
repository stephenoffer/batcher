"""The wall-clock deadline this process will be killed at, so it drains before that.

A cloud spot instance announces its own reclamation and a Kubernetes pod is sent a
``SIGTERM``, so `carbonite.resilience.preemption` can watch for both. A batch scheduler
gives neither. A Slurm allocation simply *ends* at a time fixed when the job was
submitted, and at that instant every process in it is killed — no notice, no metadata
endpoint, and on many sites no warning signal either unless the submitter asked for one.
The same shape appears wherever a launcher hands out a fixed lease: an HPC queue, a CI
job timeout, a batch VM with a scheduled teardown.

Without reading that deadline the engine treats the final second of an allocation
exactly like the first. It starts a shuffle stage it cannot possibly finish, and the
kill lands mid-write — costing the whole stage rather than the tail of it.

The deadline is a *local clock comparison*, which is what makes it useful where the
preemption monitor is blind: it needs no metadata service, no signal, and no cooperation
from the scheduler. Reported here as a plain number of seconds so the existing drain
machinery can consume it. Crossing into the lead window reads as "draining", which is
already wired to migrate this worker's shuffle output to a survivor before it dies.

This sits in `config` — layer 0 — with the other environment detection
(`profiles.detect_spot_environment`), and for the same layering reason: both the profile
resolution *below* Carbonite and the preemption monitor *inside* it need the answer, and
only a layer-0 home lets them share one rather than paste two.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

__all__ = [
    "DEADLINE_HORIZON_S",
    "DEADLINE_PAST_GRACE_S",
    "deadline_epoch_s",
    "deadline_probe",
    "remaining_budget",
    "seconds_remaining",
]

# Env vars naming an absolute termination time, in Unix epoch seconds, in priority order.
#
#   * `BATCHER_DEADLINE_EPOCH_S` is the explicit, scheduler-agnostic form. A launcher that
#     knows when its lease expires — an HPC queue we do not name, a CI runner, a VM with a
#     scheduled teardown — exports this and gets the whole drain path with no other change.
#   * `SLURM_JOB_END_TIME` is set by Slurm on every job that has a time limit, which is the
#     overwhelming majority: sites almost always set a partition `MaxTime`. This is the one
#     signal that makes a plain `sbatch` allocation safe, and it costs an env read.
#
# Deliberately env-var only, matching `config.profiles`: no network call and no scheduler
# client library on a path that runs inside every worker's poll loop.
_DEADLINE_VARS = ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME")

# A deadline further out than this is treated as no deadline at all. Slurm does not omit
# `SLURM_JOB_END_TIME` for an unlimited job — it exports a saturated sentinel (`4294967294`,
# early 2106) — so a naive read would report every unlimited job as "deadlined" and, through
# `profiles.detect_leased_allocation`, hand it the spot resilience profile it does not need.
# A year is far past any real allocation and far short of the sentinel, so it separates the
# two cleanly while never rejecting a genuine lease.
DEADLINE_HORIZON_S = 365 * 24 * 3600.0

# How far *behind* now a deadline may be and still count. A deadline that just passed is not
# noise — it is the kill grace period (Slurm's `KillWait` between the `SIGTERM` and the
# `SIGKILL`), which is precisely when draining matters most; reading it as "unbounded" would
# un-drain the fleet at the worst possible moment. The bound exists to catch the one plausible
# misconfiguration instead: exporting a *relative* number of seconds ("3600") where an
# absolute epoch was meant, which lands decades in the past and would otherwise pin the fleet
# into a permanent drain with nothing to say why.
DEADLINE_PAST_GRACE_S = 3600.0


def deadline_epoch_s() -> float | None:
    """The Unix time this process expects to be killed at, or `None` if unbounded.

    Reads the first of `BATCHER_DEADLINE_EPOCH_S` / `SLURM_JOB_END_TIME` that parses to a
    time within `DEADLINE_PAST_GRACE_S` behind now and `DEADLINE_HORIZON_S` ahead of it. A
    value that is unparseable, beyond the horizon (Slurm's unlimited-job sentinel), or far
    enough in the past to be a relative-seconds mistake reads as no deadline — so a
    malformed export degrades to today's behavior rather than draining the fleet forever.

    Returns:
        The absolute deadline in epoch seconds, or `None` when none is known.
    """
    now = time.time()
    for var in _DEADLINE_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        try:
            epoch = float(raw)
        except ValueError:
            continue  # a malformed export must not be read as "expires now"
        if now - DEADLINE_PAST_GRACE_S <= epoch <= now + DEADLINE_HORIZON_S:
            return epoch
    return None


def seconds_remaining() -> float | None:
    """Seconds left in this allocation, or `None` when no deadline is known.

    Never negative: a deadline already passed reports `0.0`, because the useful question
    at that point is "how much time do I have" and the honest answer is none.

    Examples:
        .. doctest::

            >>> import os, time
            >>> os.environ["BATCHER_DEADLINE_EPOCH_S"] = str(time.time() + 600)
            >>> 590 < seconds_remaining() <= 600
            True
            >>> del os.environ["BATCHER_DEADLINE_EPOCH_S"]
            >>> seconds_remaining() is None
            True

    Returns:
        Seconds until termination, or `None` when unbounded.
    """
    epoch = deadline_epoch_s()
    if epoch is None:
        return None
    return max(0.0, epoch - time.time())


def remaining_budget(requested_s: float, *, reserve_s: float = 0.0) -> float:
    """Shrink a wait to the time this process will actually still be alive for.

    Every bounded wait in the scheduler — for the autoscaler to bring nodes up, for a
    placement group to become satisfiable — was written against a cluster with no horizon,
    where waiting two minutes for capacity is free if the alternative is running
    under-provisioned. Under a lease it is not free, it is the *whole* remaining budget: a
    Slurm job with 90 seconds left will spend 180 waiting for nodes that would arrive after
    it is killed, and die having computed nothing. Worse, the failure is silent and looks
    like a slow cluster.

    So a wait is capped at what is left, minus `reserve_s` for the work the wait exists to
    enable. Waiting past the point where the result cannot be used is never right, and
    returning a smaller budget only ever makes the caller give up *sooner* and run on the
    capacity it already has — which is the correct behavior, and the one every one of these
    call sites already implements for a stalled autoscaler.

    Returns `requested_s` unchanged when no deadline is known, so a cluster with no lease
    behaves exactly as it did before.

    Args:
        requested_s: The wait the caller would use with no deadline.
        reserve_s: Time to hold back for the work following the wait, so the budget is not
            spent entirely on waiting for the ability to start.

    Returns:
        The wait to actually use, never negative and never above `requested_s`.

    Examples:
        .. doctest::

            >>> import os, time
            >>> remaining_budget(180.0)  # no deadline: unchanged
            180.0
            >>> os.environ["BATCHER_DEADLINE_EPOCH_S"] = str(time.time() + 60)
            >>> 0 < remaining_budget(180.0) <= 60
            True
            >>> remaining_budget(180.0, reserve_s=600.0)  # nothing left to spend
            0.0
            >>> del os.environ["BATCHER_DEADLINE_EPOCH_S"]
    """
    requested = max(0.0, float(requested_s))
    remaining = seconds_remaining()
    if remaining is None:
        return requested
    return max(0.0, min(requested, remaining - max(0.0, float(reserve_s))))


def deadline_probe(lead_s: float) -> Callable[[], bool]:
    """A drain probe that fires once fewer than `lead_s` seconds remain.

    The returned callable has the same shape as `preemption.cloud_preemption_probe` — no
    arguments, True means "this node is going away" — so it composes into the same
    `PreemptionMonitor` and reuses the migration path already built for spot reclamation.

    `lead_s` is how long the drain itself needs: enough to migrate this worker's published
    shuffle output to a survivor. Too short and the kill lands mid-migration, which costs
    the same recompute as not draining at all; too long and the fleet stops accepting work
    while it still had useful time. `DistributedConfig.drain_lead_s` carries the default.

    Always False when no deadline is known, so a cluster with no lease pays one env read
    per poll and behaves exactly as it did before.

    Args:
        lead_s: Seconds before the deadline at which to begin draining.

    Returns:
        A zero-argument probe reporting whether the deadline is within `lead_s`.
    """
    lead = max(0.0, float(lead_s))

    def probe() -> bool:
        remaining = seconds_remaining()
        return remaining is not None and remaining <= lead

    return probe
