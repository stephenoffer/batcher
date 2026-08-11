"""Carbonite reports the disk it would spill to, and resolves it in exactly one place.

Two related gaps.

Carbonite reported memory in detail -- both pools, the kernel's verdict, the pressure level,
the admission queue -- and disk not at all. So a query that spilled slowly, or died of
`ENOSPC`, carried nothing in its profile about the volume it spilled to. The `DiskPressure`
ladder already existed and only the spill store consulted it, which made the one component
that could act on a filling volume also the only one that could see it.

And "which volume?" was spelled out separately wherever it was asked: configured
`memory.spill_dir`, else the measured local scratch, else the system tempdir. When two
copies disagree the failure is quiet -- the hardware fingerprint that keys every learned
spill threshold described a container's overlay while the spill landed on the node's NVMe,
merging two machine classes that behave nothing alike.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher._internal.site import local_scratch_root, spill_scratch_dir
from batcher.carbonite.manager import ResourceManager
from batcher.carbonite.spill.disk import reset_disk_sampling, scratch_disk_stats
from batcher.config import Config, config_context

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_disk_readings():
    """The volume readings are TTL-cached; a patched `disk_usage` must be seen at once."""
    reset_disk_sampling()
    yield
    reset_disk_sampling()


# --- one resolution ------------------------------------------------------------


def test_a_configured_spill_dir_is_the_answer(tmp_path) -> None:
    """An operator who named a directory has already made the decision."""
    base = Config()
    pinned = base.replace(memory=dataclasses.replace(base.memory, spill_dir=str(tmp_path)))
    with config_context(pinned):
        assert spill_scratch_dir() == str(tmp_path)


def test_without_one_it_falls_back_through_the_measured_volume() -> None:
    """Configured, else measured local scratch, else the tempdir -- and always a path.

    `local_scratch_root` may legitimately answer `None` (no fast local storage mounted);
    this must not, because its caller is about to write to whatever it returns.
    """
    import tempfile

    with config_context(Config()):
        resolved = spill_scratch_dir()
    assert resolved == (local_scratch_root() or tempfile.gettempdir())
    assert isinstance(resolved, str) and resolved


def test_the_cost_model_prices_the_disk_the_spill_lands_on(tmp_path, monkeypatch) -> None:
    """The spill advisor reads the same resolution, so policy and write agree on the disk.

    Deriving it separately is how the cost model came to price a container's overlay while
    the spill landed on the node's NVMe.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "batcher._internal.hardware.storage.device_cost_factor",
        lambda path: seen.append(path) or 1.0,
        raising=True,
    )
    base = Config()
    pinned = base.replace(memory=dataclasses.replace(base.memory, spill_dir=str(tmp_path)))
    with config_context(pinned):
        ResourceManager(pinned).recommend_spill_compression(_unsized_plan())
    assert seen == [str(tmp_path)], (
        "the spill cost model priced a different volume than the one a spill would use"
    )


# --- the reading ---------------------------------------------------------------


def test_the_scratch_reading_has_the_documented_shape() -> None:
    stats = scratch_disk_stats()
    assert set(stats) == {"path", "pressure", "free_bytes", "total_bytes"}
    assert stats["pressure"] in {"NORMAL", "ELEVATED", "FULL", "UNKNOWN"}
    assert stats["free_bytes"] >= -1
    assert stats["total_bytes"] >= -1


def test_an_unreadable_volume_says_so_rather_than_reporting_zero(monkeypatch) -> None:
    """`-1` and `UNKNOWN`, never `0` and `NORMAL`.

    A measurement that could not be taken is not the same claim as a volume that is empty,
    and the whole value of the reading is telling an operator which one they have.
    """

    def _explode():
        raise OSError("no such volume")

    monkeypatch.setattr("batcher._internal.site.spill_scratch_dir", _explode, raising=True)
    stats = scratch_disk_stats()
    assert stats["pressure"] == "UNKNOWN"
    assert stats["free_bytes"] == -1
    assert stats["total_bytes"] == -1


def test_a_probe_failure_never_breaks_the_snapshot(monkeypatch) -> None:
    """`stats()` feeds `explain(analyze=True)`; a diagnostic must not fail the query."""

    def _explode():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("batcher._internal.site.spill_scratch_dir", _explode, raising=True)
    assert ResourceManager().stats()["scratch_disk"]["pressure"] == "UNKNOWN"


# --- it reaches the snapshot ----------------------------------------------------


def test_the_manager_reports_the_scratch_volume() -> None:
    """The gap this closes: a spilled query's profile said nothing about its disk."""
    stats = ResourceManager().stats()
    assert "scratch_disk" in stats
    assert stats["scratch_disk"]["path"], "the snapshot named no scratch volume"


def _unsized_plan():
    """A plan with no byte estimate -- enough to reach the device probe."""
    from batcher.plan.physical import PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    op = PhysicalOp(
        op_id=0,
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=1 << 30, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


# --- the shortfall is reported, not merely computable ---------------------------


def test_a_saturated_bucket_count_says_so(caplog) -> None:
    """`partitions_for_envelope` clamps at 4,096 buckets, so past that its promise stops
    holding. `envelope_shortfall` has always been able to say by how much and nothing
    consulted it, which made an extra write-and-re-read of the whole spilled state
    indistinguishable from the spill just being large.
    """
    import logging as _logging

    from batcher.plan.resource import ResourceBounds

    rm = ResourceManager()
    plan = _huge_plan()
    with caplog.at_level(_logging.INFO, logger="batcher.carbonite.spill"):
        parts = rm.partitions_for_bounds(
            plan, ResourceBounds(m_max_bytes=1 << 20, c_max_credits=0, n_max_parallelism=0)
        )
    assert parts == 4096, "the bucket count did not saturate, so nothing should be reported"
    assert any("exceed the offered envelope" in r.message for r in caplog.records), (
        "the buckets will not fit the envelope and nothing said so"
    )


def test_buckets_that_fit_report_nothing(caplog) -> None:
    """The negative control: one line per query that needs it, none for a query that does not."""
    import logging as _logging

    from batcher.plan.resource import ResourceBounds

    rm = ResourceManager()
    with caplog.at_level(_logging.INFO, logger="batcher.carbonite.spill"):
        rm.partitions_for_bounds(
            _unsized_plan(),
            ResourceBounds(m_max_bytes=1 << 30, c_max_credits=0, n_max_parallelism=0),
        )
    assert not [r for r in caplog.records if "exceed the offered envelope" in r.message]


def _huge_plan():
    """A plan whose state is far past `MAX_SPILL_PARTITIONS x envelope`."""
    from batcher.plan.physical import PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    op = PhysicalOp(
        op_id=0,
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=1 << 50, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))
