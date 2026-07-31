"""The kernel memory signals nothing was reading: `memory.high`, `memory.events`, and PSI.

Each test here corresponds to a way a container dies (or silently crawls) that the byte
accounting in `probe`/`pressure` could not see. The cgroup hierarchy is faked on disk rather
than mocked, because the thing under test *is* the file-format archaeology.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.memory import kernel, probe
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config, MemoryConfig, config_context

_GIB = 1 << 30


@pytest.fixture(autouse=True)
def _fresh_samples():
    """Both modules memoize; a stale snapshot would make every test observe the first one."""
    kernel.reset_kernel_sampling()
    probe.reset_memory_sampling()
    yield
    kernel.reset_kernel_sampling()
    probe.reset_memory_sampling()


def _fake_cgroup(monkeypatch, tmp_path, files: dict[str, str], *, leaf: str = "leaf") -> None:
    """Point the cgroup readers at a fake hierarchy: a mount root plus one delegated leaf.

    Two levels, because the attribution rules under test are all about the difference — PSI is
    only trusted from our own slice, usage is read nearest-first, and a limit binds from
    anywhere in the ancestry.
    """
    root = tmp_path / "root"
    own = root / leaf
    own.mkdir(parents=True)
    for name, content in files.items():
        (own / name).write_text(content)
    monkeypatch.setattr(kernel, "cgroup_v2_dirs", lambda: (str(root), str(own)))


def test_memory_high_is_the_ceiling_that_actually_binds(monkeypatch, tmp_path):
    """K8s memory QoS sets `high` from the pod request and `max` from its limit. A query
    planned against `max` spends its life asleep in direct reclaim between the two."""
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        {"memory.max": str(8 * _GIB), "memory.high": str(4 * _GIB)},
    )
    state = kernel.kernel_memory_state()
    assert state.limit_bytes == 8 * _GIB
    assert state.high_bytes == 4 * _GIB
    assert state.effective_limit_bytes == 4 * _GIB


def test_a_missing_memory_high_leaves_the_ceiling_at_the_hard_limit(monkeypatch, tmp_path):
    """The overwhelmingly common case (bare metal, most containers) must be unchanged."""
    _fake_cgroup(monkeypatch, tmp_path, {"memory.max": str(8 * _GIB)})
    assert kernel.kernel_memory_state().effective_limit_bytes == 8 * _GIB


def test_unreclaimable_charge_excludes_page_cache(monkeypatch, tmp_path):
    """`memory.current` counts clean file pages the kernel drops before killing anything, so
    charging them as pressure reads a box that has merely read files as one about to die."""
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        {
            "memory.max": str(20 * _GIB),
            "memory.current": str(15 * _GIB),
            "memory.stat": f"anon {4 * _GIB}\nfile {10 * _GIB}\nslab {_GIB}\n",
        },
    )
    state = kernel.kernel_memory_state()
    assert state.current_bytes == 15 * _GIB
    assert state.unreclaimable_bytes == 5 * _GIB
    assert state.headroom_bytes == 15 * _GIB


def test_an_oom_kill_is_reported_as_history_not_a_forecast(monkeypatch, tmp_path):
    """A non-zero `oom_kill` is proof this container has already died at this size."""
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        {"memory.events": "low 0\nhigh 12\nmax 3\noom 2\noom_kill 1\n"},
    )
    state = kernel.kernel_memory_state()
    assert state.oom_kills == 1
    assert state.throttle_events == 12
    assert state.was_oom_killed is True


def test_no_cgroup_reports_nothing_rather_than_zero(monkeypatch, tmp_path):
    """`None` and `0` must stay distinct: zero swap in use is not the same fact as a host
    with no cgroups, and a caller that conflates them concludes things about a machine it
    never measured."""
    _fake_cgroup(monkeypatch, tmp_path, {})
    state = kernel.kernel_memory_state()
    assert state.oom_kills is None
    assert state.swap_current_bytes is None
    assert state.effective_limit_bytes is None
    assert state.was_oom_killed is False


def test_swap_headroom_is_the_soft_landing_under_an_over_committed_query(monkeypatch, tmp_path):
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        {"memory.swap.max": str(4 * _GIB), "memory.swap.current": str(_GIB)},
    )
    assert kernel.kernel_memory_state().swap_headroom_bytes == 3 * _GIB


class TestPsiAttribution:
    """PSI is only actionable when it describes *this* workload's stalls.

    Acting on a host-wide reading means one container spills and halves its morsels because a
    different container is thrashing — which relieves nothing, since the memory is not ours to
    give back, and makes the pressure level depend on what else the box happens to be running.
    """

    def test_our_own_slice_is_trusted(self, monkeypatch, tmp_path):
        _fake_cgroup(
            monkeypatch,
            tmp_path,
            {"memory.pressure": "some avg10=60.00 avg60=1.00 total=1\nfull avg10=45.00 total=1\n"},
        )
        assert kernel.memory_stall_full() == pytest.approx(0.45)

    def test_a_delegated_cgroup_without_psi_reports_nothing(self, monkeypatch, tmp_path):
        """Not the host's figure. We are in someone's slice and nothing measured *our* stalls."""
        _fake_cgroup(monkeypatch, tmp_path, {"memory.max": str(_GIB)})
        assert kernel.memory_stall_full() is None

    def test_the_mount_root_alone_is_not_our_slice(self, monkeypatch, tmp_path):
        """The root's PSI is the whole machine. A container must not act on it."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "memory.pressure").write_text(
            "some avg10=90.00 total=1\nfull avg10=80.00 total=1\n"
        )
        own = root / "leaf"
        own.mkdir()
        monkeypatch.setattr(kernel, "cgroup_v2_dirs", lambda: (str(root), str(own)))
        assert kernel.memory_stall_full() is None


class TestStallFloor:
    """PSI raises the pressure level and can never lower it."""

    def _monitor(
        self, monkeypatch, stall: float | None, *, enabled: bool = True
    ) -> PressureMonitor:
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.memory_stall_full", lambda: stall, raising=False
        )
        cfg = Config(memory=MemoryConfig(stall_aware_pressure=enabled))
        return PressureMonitor(cfg)

    def test_a_thrashing_cgroup_is_spilled_even_with_bytes_to_spare(self, monkeypatch):
        """The failure the byte accounting cannot see: reclaim is defending a limit, so the
        charge sits pinned below it while the container makes no progress."""
        monitor = self._monitor(monkeypatch, 0.5)
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.1))
        assert monitor.classify() is PressureLevel.SPILL

    def test_a_moderate_stall_only_warns(self, monkeypatch):
        monitor = self._monitor(monkeypatch, 0.15)
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.1))
        assert monitor.classify() is PressureLevel.ELEVATED

    def test_it_never_lowers_a_level_the_bytes_earned(self, monkeypatch):
        """A quiet kernel with a full pool is still a full pool."""
        monitor = self._monitor(monkeypatch, 0.0)
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.99))
        assert monitor.classify() is PressureLevel.CRITICAL

    def test_it_is_capped_at_spill_so_a_reclaim_burst_cannot_stop_the_query(self, monkeypatch):
        """CRITICAL pauses producers outright. A stall share is a rate, not a headroom figure,
        so it must never be the thing that halts a query with gigabytes free."""
        monitor = self._monitor(monkeypatch, 1.0)
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.0))
        assert monitor.classify() is PressureLevel.SPILL

    def test_the_config_switch_makes_it_inert(self, monkeypatch):
        monitor = self._monitor(monkeypatch, 1.0, enabled=False)
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.0))
        assert monitor.classify() is PressureLevel.NORMAL


class TestOomKillBackoff:
    """A restarted worker must not re-derive the envelope that already got it killed."""

    def test_a_prior_kill_shrinks_the_auto_sensed_envelope(self, monkeypatch):
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.oom_kill_count", lambda: 1, raising=False
        )
        cfg = Config(memory=MemoryConfig(oom_kill_backoff=0.5))
        monitor = PressureMonitor(cfg)
        monkeypatch.setattr(monitor, "available_bytes", lambda: 8 * _GIB)
        assert monitor.envelope_bytes() == 4 * _GIB

    def test_a_clean_history_leaves_the_envelope_alone(self, monkeypatch):
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.oom_kill_count", lambda: 0, raising=False
        )
        monitor = PressureMonitor(Config(memory=MemoryConfig(oom_kill_backoff=0.5)))
        monkeypatch.setattr(monitor, "available_bytes", lambda: 8 * _GIB)
        assert monitor.envelope_bytes() == 8 * _GIB

    def test_an_explicit_cap_is_an_instruction_and_is_never_scaled(self, monkeypatch):
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.oom_kill_count", lambda: 3, raising=False
        )
        cfg = Config(memory=MemoryConfig(max_memory_bytes=6 * _GIB, oom_kill_backoff=0.5))
        with config_context(cfg):
            assert PressureMonitor(cfg).envelope_bytes() == 6 * _GIB


class TestEffectiveCeilingDrivesPressure:
    """The pressure fraction must measure the ceiling that binds, not the one that kills."""

    def test_a_throttled_container_reads_as_pressured(self, monkeypatch):
        """With `memory.high` at half of `memory.max`, a footprint at 87% of `high` is a
        container the kernel is already sleeping in reclaim. Dividing by `max` reports it as
        43% used, so the level stays NORMAL through the entire throttled band."""
        monkeypatch.setattr(probe, "effective_limit_bytes", lambda: 4 * _GIB)
        monkeypatch.setattr(probe, "total_memory_bytes", lambda: 8 * _GIB)
        monkeypatch.setattr(probe, "cgroup_current_bytes", lambda: int(3.5 * _GIB))
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.memory_stall_full", lambda: None, raising=False
        )
        assert PressureMonitor(Config()).classify() >= PressureLevel.SPILL

    def test_without_a_throttle_threshold_nothing_changes(self, monkeypatch):
        """The same footprint against a plain `memory.max` is what it always was."""
        monkeypatch.setattr(probe, "effective_limit_bytes", lambda: 8 * _GIB)
        monkeypatch.setattr(probe, "total_memory_bytes", lambda: 8 * _GIB)
        monkeypatch.setattr(probe, "cgroup_current_bytes", lambda: int(3.5 * _GIB))
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.memory_stall_full", lambda: None, raising=False
        )
        assert PressureMonitor(Config()).classify() is PressureLevel.NORMAL


class TestOomHistoryForcesSpill:
    """The third spill signal, and the only one that is evidence rather than inference.

    The other two go quiet in exactly this situation: Kyber emits `0` for an operator whose
    cardinality it cannot estimate, and live pressure is measured *before* the query that will
    cause the problem has allocated anything. A worker that was killed, restarted, and handed
    the same un-sized plan therefore takes the "fits" fast path straight back into the kill.
    """

    def _unsized_plan(self):
        from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
        from batcher.plan.resource import ResourceBounds

        op = PhysicalOp(
            op_id=OpId(1),
            kind="Aggregate",
            backend="native",
            algorithm="hash",
            # `0` is what Kyber emits when it cannot estimate the cardinality.
            bounds=ResourceBounds(m_max_bytes=0, c_max_credits=4, n_max_parallelism=4),
            inputs=(),
        )
        return PhysicalPlan(ir={}, output_schema=None, ops=(op,))

    def _sized_plan(self):
        from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
        from batcher.plan.resource import ResourceBounds

        op = PhysicalOp(
            op_id=OpId(1),
            kind="Aggregate",
            backend="native",
            algorithm="hash",
            bounds=ResourceBounds(m_max_bytes=1024, c_max_credits=4, n_max_parallelism=4),
            inputs=(),
        )
        return PhysicalPlan(ir={}, output_schema=None, ops=(op,))

    def _killed(self, monkeypatch, times: int):
        from batcher.carbonite.memory.kernel import KernelMemoryState

        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.kernel_memory_state",
            lambda: KernelMemoryState(oom_kills=times),
            raising=False,
        )

    def test_an_unsized_plan_in_a_killed_cgroup_goes_out_of_core(self, monkeypatch):
        from batcher.carbonite import ResourceManager

        self._killed(monkeypatch, 1)
        cfg = Config(memory=MemoryConfig(max_memory_bytes=1 << 40))
        with config_context(cfg):
            reason = ResourceManager(cfg).spill_reason(self._unsized_plan())
        assert reason is not None
        assert "OOM-killed" in reason

    def test_a_sized_plan_that_fits_is_left_alone(self, monkeypatch):
        """Narrow on purpose: a plan Kyber *did* size has already been compared against the
        budget, and overruling that would spill queries that measurably fit."""
        from batcher.carbonite import ResourceManager

        self._killed(monkeypatch, 1)
        cfg = Config(memory=MemoryConfig(max_memory_bytes=1 << 40))
        with config_context(cfg):
            assert ResourceManager(cfg).spill_reason(self._sized_plan()) is None

    def test_a_clean_cgroup_leaves_an_unsized_plan_in_memory(self, monkeypatch):
        from batcher.carbonite import ResourceManager

        self._killed(monkeypatch, 0)
        cfg = Config(memory=MemoryConfig(max_memory_bytes=1 << 40))
        with config_context(cfg):
            assert ResourceManager(cfg).spill_reason(self._unsized_plan()) is None
