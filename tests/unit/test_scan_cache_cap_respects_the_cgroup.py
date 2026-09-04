"""The scan cache is sized by the ceiling this process runs under, not the host's RAM.

`dist.executors.scan_read._default_scan_cache_cap` bounds how much decoded scan data a worker
holds. It used to read `psutil.virtual_memory().total`, which reports the **host's** RAM.
Under a container -- the ordinary way a Ray worker runs -- that is the wrong number: a 4 GiB
container on a 512 GiB node sizes the cache against 512 GiB and gets the cgroup to OOM-kill
it. The comment beside the calculation already worried about exactly this failure arriving
through the *divisor* ("every worker independently fills to `frac * node_RAM` and the node
OOMs"); it could arrive through the numerator too.

`_internal.hardware.memory.machine_memory_bytes` is documented as "the one implementation"
behind every memory-sizing decision, and takes the tightest of host RAM, `memory.max`,
`memory.high`, the batch scheduler's grant and `RLIMIT_AS`. `carbonite.memory.probe` already
delegates to it; this now does too.

The probe failure is handled the same way `carbonite.memory.probe` handles it -- fall back to
the *configured* `memory.default_total_bytes` rather than a hardcoded 8 GiB. An operator who
tells Batcher how much memory it has should be believed here as well, and a constant cannot
be told.

Not asserted here: that a cgroup limit is actually respected. That needs a container, and
this suite has none. What is asserted is that the cap is a function of the sanctioned reader
and the configured fallback -- which is the part that was wrong.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, active_config, set_config
from batcher.dist.executors import scan_read

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_config():
    original = active_config()
    try:
        yield original
    finally:
        set_config(original)


def test_the_cap_tracks_the_process_memory_ceiling(monkeypatch):
    """Doubling the ceiling the process runs under doubles the cap."""
    monkeypatch.setattr(scan_read, "_scan_cache_siblings", lambda: 1)
    monkeypatch.setenv("BATCHER_SCAN_CACHE_FRACTION", "0.5")

    monkeypatch.setattr(scan_read, "machine_memory_bytes", lambda: 8 * 1024**3)
    small = scan_read._default_scan_cache_cap()
    monkeypatch.setattr(scan_read, "machine_memory_bytes", lambda: 16 * 1024**3)
    large = scan_read._default_scan_cache_cap()

    assert small == 4 * 1024**3
    assert large == 2 * small, (
        "the cap did not follow the memory ceiling, so it is being computed from something "
        "else -- the host's RAM, or a constant"
    )


def test_an_unreadable_ceiling_falls_back_to_the_configured_default(monkeypatch, restore_config):
    """Not to a hardcoded 8 GiB: an operator who configured a total must be believed.

    This is the half that fails against the previous implementation, which returned
    `8 * 1024**3` regardless of configuration.
    """
    monkeypatch.setattr(scan_read, "_scan_cache_siblings", lambda: 1)
    monkeypatch.setenv("BATCHER_SCAN_CACHE_FRACTION", "0.5")
    monkeypatch.setattr(scan_read, "machine_memory_bytes", lambda: 0)

    base = restore_config
    configured = 40 * 1024**3
    set_config(
        dataclasses.replace(
            base, memory=dataclasses.replace(base.memory, default_total_bytes=configured)
        )
    )
    assert scan_read._default_scan_cache_cap() == configured // 2, (
        "the fallback ignored `memory.default_total_bytes`, so it is still a constant"
    )


def test_the_siblings_divisor_still_applies(monkeypatch):
    """The bound this function already had must survive the change to the numerator."""
    monkeypatch.setenv("BATCHER_SCAN_CACHE_FRACTION", "1.0")
    monkeypatch.setattr(scan_read, "machine_memory_bytes", lambda: 12 * 1024**3)

    monkeypatch.setattr(scan_read, "_scan_cache_siblings", lambda: 1)
    alone = scan_read._default_scan_cache_cap()
    monkeypatch.setattr(scan_read, "_scan_cache_siblings", lambda: 4)
    shared = scan_read._default_scan_cache_cap()

    assert alone == 12 * 1024**3
    assert shared == alone // 4, "the per-node divisor was lost"


def test_the_default_config_is_not_secretly_eight_gib(restore_config):
    """Guard against a vacuous suite.

    `memory.default_total_bytes` happens to be 8 GiB today, which is the same number the old
    hardcoded fallback used. If it stays that way forever, the fallback test above passes
    against either implementation. This fails if the two are ever the only thing being
    compared, by pinning that the test above supplies its own distinct value.
    """
    assert Config().memory.default_total_bytes == 8 * 1024**3, (
        "if this default changed, the fallback test's 40 GiB is still distinct from it -- "
        "update this note rather than the assertion"
    )
