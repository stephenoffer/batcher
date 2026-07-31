"""Checking a node before trusting it, and the two ways that check must not backfire.

The checks themselves are cheap. What makes them safe to run on every worker at startup is
the discipline around their verdicts:

* **A check that cannot run reports `"unknown"`, never `"failed"`.** No NVML, no readable
  kernel log, a container that shares neither — a fleet must not refuse to start because a
  base image stopped shipping `pynvml`.
* **A check never raises.** This runs on every node at once; a probe that threw would turn
  "one node is degraded" into "the fleet did not start", which is the failure it exists to
  prevent, made worse.

And at the fleet level, *some* nodes failing and *all* of them failing have opposite remedies:
the first is capacity the scheduler routes around, the second is a configuration error, and
draining the fleet over it produces an outage in place of an error message.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import preflight as pf

pytestmark = pytest.mark.unit


def _report(*checks: tuple[str, str]) -> pf.PreflightReport:
    return pf.PreflightReport(node="n", checks=tuple(pf.CheckResult(n, s, "") for n, s in checks))


# --- individual checks --------------------------------------------------------------------


def test_a_writable_scratch_directory_passes(tmp_path):
    result = pf._check_scratch(str(tmp_path))
    assert result.status in ("ok", "warn")


def test_an_unwritable_scratch_directory_fails(tmp_path):
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        result = pf._check_scratch(str(blocked))
    finally:
        blocked.chmod(0o700)
    assert result.status == "failed"
    assert "not writable" in result.detail


def test_no_configured_scratch_is_unknown_not_a_failure():
    assert pf._check_scratch("").status == "unknown"


# --- the report ---------------------------------------------------------------------------


def test_a_probe_that_raises_does_not_take_the_node_out(monkeypatch):
    monkeypatch.setattr(pf, "_check_kernel_log", lambda: 1 / 0)
    report = pf.preflight_check("node-1")
    assert report.ok is True
    kernel = next(c for c in report.checks if c.name == "kernel")
    assert kernel.status == "unknown"


def test_the_report_never_raises_whatever_the_host_is():
    report = pf.preflight_check("node-1")
    assert {c.name for c in report.checks} == {"scratch", "kernel"}
    assert report.summary()


def test_a_failure_is_named_in_the_summary():
    report = _report(("scratch", "failed"), ("devices", "ok"))
    assert report.ok is False
    assert [c.name for c in report.failures] == ["scratch"]
    assert "scratch" in report.summary()


def test_warnings_do_not_take_a_node_out_of_rotation():
    report = _report(("device_health", "warn"), ("kernel", "warn"))
    assert report.ok is True
    assert len(report.warnings) == 2
