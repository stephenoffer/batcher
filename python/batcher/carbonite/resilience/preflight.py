"""The node-level readiness checks the device probe does not cover.

A distributed job discovers a bad node the expensive way: it schedules onto it, the task
fails, the retry lands somewhere else, and — because the scheduler still sees a free slot —
the next task lands right back on it. On a long stage that costs the whole stage.

The *device* half of finding that out early already exists: `carbonite.accel.health` reads
every GPU's telemetry, fault counters, and Xid history. This is the other half, the two node
conditions that fail every task placed there while every device on the node reads perfectly
healthy:

* **A scratch directory that cannot be written.** Spill, checkpoints, and the object store all
  need it. A filesystem remounted read-only, or one with no space, fails every task on the
  node while the node itself stays up and stays scheduled.
* **Node faults in the kernel log.** The OOM killer having already fired here, a filesystem
  error, a PCIe link retraining under a device. A worker the kernel killed never raises, never
  unwinds, and never logs, so from the driver it is indistinguishable from a preemption.

**A failed check is a report, never an exception.** This runs on a worker, on every node — a
check that raises would turn "one node is degraded" into "the fleet did not start", which is
the failure this exists to prevent, made worse. Severity is reported and the caller decides.
And a check that cannot *run* — no readable kernel log, a container that shares neither it nor
a configured scratch directory — is `"unknown"`, never `"failed"`: a fleet must not refuse to
start because a base image changed.

Carbonite owns it (a protection concern). The per-node report is a plain dataclass, so `dist`
collects one from each worker and reduces them on the driver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "CheckResult",
    "PreflightReport",
    "preflight_check",
]

#: Bytes of free scratch space below which spill is considered unlikely to survive a stage.
#: Deliberately modest: this is a floor that catches a *full* disk, not a sizing estimate, and
#: a stage's real spill footprint is Carbonite's memory envelope to reason about.
_MIN_SCRATCH_BYTES = 1 << 30


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One preflight check's outcome.

    Attributes:
        name: What was checked (`"devices"`, `"device_health"`, `"scratch"`, `"kernel"`).
        status: `"ok"`, `"warn"`, `"failed"`, or `"unknown"`. `"unknown"` means the check
            could not run at all, which is not evidence of anything and must never be counted
            as a failure.
        detail: One line for an operator, naming what was found.
    """

    name: str
    status: str = "unknown"
    detail: str = ""

    @property
    def blocking(self) -> bool:
        """Whether this result alone should keep work off the node."""
        return self.status == "failed"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Every check's outcome for one node.

    Attributes:
        node: The node's identifier, as the caller knows it.
        checks: One `CheckResult` per check, in the order they ran.
    """

    node: str = ""
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the node is fit for work — no check outright failed."""
        return not any(c.blocking for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """The checks that failed, in order."""
        return tuple(c for c in self.checks if c.blocking)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        """The checks that passed with something worth saying, in order."""
        return tuple(c for c in self.checks if c.status == "warn")

    def summary(self) -> str:
        """One line naming what is wrong, or that nothing is.

        Returns:
            A short description suitable for a log line or an incident report.
        """
        bad = self.failures or self.warnings
        if not bad:
            return f"{self.node or 'node'}: ok"
        return f"{self.node or 'node'}: " + "; ".join(f"{c.name}: {c.detail}" for c in bad)


def _check_scratch(path: str) -> CheckResult:
    """Whether the spill directory can actually be written, and has room."""
    if not path:
        return CheckResult("scratch", "unknown", "no scratch directory configured")
    probe = os.path.join(path, f".batcher-preflight-{os.getpid()}")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "wb") as handle:
            handle.write(b"\0")
        os.unlink(probe)
    except OSError as exc:
        # A read-only remount and a full disk both land here, and both fail every task on the
        # node from this moment while the node stays up and keeps being scheduled.
        return CheckResult("scratch", "failed", f"{path} is not writable: {exc.strerror or exc}")
    try:
        usage = os.statvfs(path)
        free = usage.f_bavail * usage.f_frsize
    except (OSError, AttributeError):
        return CheckResult("scratch", "ok", f"{path} is writable")
    if free < _MIN_SCRATCH_BYTES:
        return CheckResult("scratch", "warn", f"{path} has {free // (1 << 20)} MiB free")
    return CheckResult("scratch", "ok", f"{path} has {free // (1 << 30)} GiB free")


def _check_kernel_log() -> CheckResult:
    """Whether the kernel has already reported a node fault here."""
    from batcher._internal.hardware.faults.kernel import (
        node_fault_counts,
        node_faults,
        node_faults_readable,
        worst_severity,
    )

    if not node_faults_readable():
        return CheckResult("kernel", "unknown", "the kernel log is not readable here")
    faults = node_faults()
    if not faults:
        return CheckResult("kernel", "ok", "no node faults reported")
    counts = node_fault_counts(faults)
    detail = ", ".join(f"{kind} x{count}" for kind, count in sorted(counts.items()))
    severity = worst_severity(faults)
    # A fatal kernel fault is reported as a *warning*, not a failure. These are node
    # conditions in the recent past, and the node may well have recovered — an OOM kill an
    # hour ago is history, where a scratch directory that will not accept a write right now
    # is a fact. Refusing a node on history is how a fleet loses capacity it still has.
    return CheckResult("kernel", "warn" if severity != "note" else "ok", detail)


def preflight_check(node: str = "", *, scratch_path: str = "") -> PreflightReport:
    """Run the node-level checks on this host and report what was found.

    Never raises. A check that cannot run reports `"unknown"`, which is not a failure: a fleet
    must not refuse to start because a container does not share the host's kernel log.

    Args:
        node: This node's identifier, carried into the report.
        scratch_path: The spill directory to test for writability and space. Empty resolves to
            the system temporary directory, which is where a per-query scratch directory is
            actually created — checking nothing instead would report `"unknown"` on every
            deployment that has not configured spill, which is precisely the set this catches.

    Returns:
        A `PreflightReport` whose `ok` says whether work should be placed here.
    """
    import tempfile

    from batcher._internal.logging import note_suppressed

    checks: list[CheckResult] = []
    path = scratch_path or tempfile.gettempdir()
    probes = (
        ("scratch", lambda: _check_scratch(path)),
        ("kernel", _check_kernel_log),
    )
    for name, probe in probes:
        try:
            checks.append(probe())
        except Exception as exc:
            # A probe that raises has told us nothing about the node, so the result is
            # "unknown" and the node keeps its slot. The alternative — a broken probe
            # draining a healthy fleet — is the exact failure this module exists to avoid.
            note_suppressed("carbonite", f"run the {name} preflight check", exc)
            checks.append(CheckResult(name, "unknown", f"the check itself failed: {exc}"))
    return PreflightReport(node=node, checks=tuple(checks))
