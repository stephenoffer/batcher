"""The engine-side hardware probes, including the degraded paths that must not raise.

The reason these are worth testing is not the happy path. It is that an installed `.so`
routinely lags the source tree — every session that has not rebuilt is running one — so the
probes have to answer "I don't know" for an entry point the extension does not have, rather
than raising an `AttributeError` out of what is supposed to be an advisory reading.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import engine as engine_hw
from batcher._internal.hardware.engine import allocator, detected

pytestmark = pytest.mark.unit


class _Engine:
    """A stand-in engine exposing exactly the entry points a test names."""

    def __init__(self, **entry_points: object) -> None:
        for name, value in entry_points.items():
            setattr(self, name, value)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Each test gets its own reading, since the probes are memoized for the process."""
    from batcher._internal.hardware import reset_hardware_probes

    reset_hardware_probes()
    yield
    reset_hardware_probes()


def test_an_engine_that_is_not_built_reports_unknown_rather_than_raising(monkeypatch):
    monkeypatch.setattr(detected, "engine_or_none", lambda: None)
    assert detected.engine_hardware() == {}
    assert detected.engine_pinning_order() == ()
    assert detected.engine_numa_map() == {}
    assert allocator.allocator_stats() == {}
    assert allocator.release_retained_memory() == 0


def test_an_older_extension_missing_the_entry_points_reports_unknown(monkeypatch):
    # The case that actually happens: the engine imports fine, but it was built before these
    # functions existed. Looking the attribute up rather than calling it is what makes this a
    # shrug instead of an AttributeError in the middle of a memory-envelope decision.
    monkeypatch.setattr(detected, "engine_or_none", _Engine)
    assert detected.engine_hardware() == {}
    assert detected.engine_pinning_order() == ()
    assert allocator.allocator_stats() == {}
    assert allocator.release_retained_memory(force=True) == 0


def test_an_entry_point_that_raises_is_swallowed(monkeypatch):
    def boom() -> dict[str, int]:
        raise RuntimeError("cpuid unavailable")

    monkeypatch.setattr(detected, "engine_or_none", lambda: _Engine(engine_hardware=boom))
    assert detected.engine_hardware() == {}


def test_reported_facts_are_passed_through_and_shaped(monkeypatch):
    monkeypatch.setattr(
        detected,
        "engine_or_none",
        lambda: _Engine(
            engine_hardware=lambda: {
                "logical_cores": 92,
                "l2_bytes": 1 << 20,
                "isa_tier": "avx512",
            },
            engine_pinning_order=lambda: [0, 24, 1, 25],
            engine_numa_map=lambda: [(0, [0, 1]), (1, [24, 25])],
        ),
    )
    hw = detected.engine_hardware()
    assert hw["logical_cores"] == 92
    assert hw["isa_tier"] == "avx512"
    # A tuple, not a list: these are memoized readings handed to many callers, and a mutable
    # one would let a caller edit every other caller's copy.
    assert detected.engine_pinning_order() == (0, 24, 1, 25)
    assert detected.engine_numa_map() == {0: (0, 1), 1: (24, 25)}


def test_release_returns_the_bytes_the_allocator_reports(monkeypatch):
    calls: list[bool] = []

    def collect(force: bool) -> int:
        calls.append(force)
        return 4096

    monkeypatch.setattr(detected, "engine_or_none", lambda: _Engine(allocator_collect=collect))
    assert allocator.release_retained_memory() == 4096
    assert allocator.release_retained_memory(force=True) == 4096
    assert calls == [False, True], "the force flag must reach the allocator"


def test_the_package_facade_exports_what_it_documents():
    for name in (
        "allocator_stats",
        "engine_hardware",
        "engine_numa_map",
        "engine_pinning_order",
        "release_retained_memory",
    ):
        assert hasattr(engine_hw, name), name
        assert name in engine_hw.__all__


def test_resetting_probes_re_reads_the_engine(monkeypatch):
    from batcher._internal.hardware import reset_hardware_probes

    readings = iter([{"logical_cores": 4}, {"logical_cores": 8}])
    monkeypatch.setattr(
        detected, "engine_or_none", lambda: _Engine(engine_hardware=lambda: next(readings))
    )
    assert detected.engine_hardware()["logical_cores"] == 4
    assert detected.engine_hardware()["logical_cores"] == 4, "memoized within a process"
    reset_hardware_probes()
    assert detected.engine_hardware()["logical_cores"] == 8, "the reset hook must cover it"


@pytest.mark.skipif(
    not detected.engine_hardware(), reason="engine not built, or predates the entry points"
)
def test_the_real_engine_reports_a_coherent_machine():
    hw = detected.engine_hardware()
    assert hw["logical_cores"] >= 1
    assert 1 <= hw["physical_cores"] <= hw["logical_cores"]
    assert hw["smt_width"] >= 1
    assert hw["numa_nodes"] >= 1
    assert hw["cache_line"] in (32, 64, 128)
    assert hw["l1d_bytes"] <= hw["l2_bytes"] <= hw["l3_bytes"]
    assert hw["compute_threads"] == hw["physical_cores"]
    # A quota can cap the logical count below `physical x SMT`; it must never make SMT
    # disappear, which a `logical / physical` ratio would.
    if hw["logical_cores"] > hw["physical_cores"]:
        assert hw["has_smt"], "more logical CPUs than cores means SMT, whatever the quota says"
    order = detected.engine_pinning_order()
    assert len(set(order)) == len(order), "a repeated CPU would oversubscribe one core"


def test_the_two_planes_describe_the_same_machine():
    """The control plane's CPU facts must equal the data plane's.

    Kyber sizes thresholds from `_internal.hardware`; the engine sizes its thread pools and
    shard counts from `bc_arrow::CpuTopology`. They are separate implementations of the same
    questions in two languages, so nothing but a test keeps them from drifting apart, and a
    drift is silent: the plan is sized for one machine and executed on another.

    The NUMA count must agree exactly. The two core counts are allowed to differ by one, and
    only one, because the engine reaches the affinity mask through `available_parallelism`
    while the control plane reads `sched_getaffinity`, and on a *fractional* CFS quota the two
    can round the same `cpu.max` differently. One core is the whole width of that effect; the
    failure this test exists to catch is the other order of magnitude entirely, a plane
    reporting the host's 48 cores for a process budgeted 4.

    The cache sizes are deliberately not compared. The two planes apply different domain
    policies today -- the control plane takes the binding domain across usable CPUs, the engine
    reads `cpu0` -- so an equality here would either fail on a hybrid or stacked-cache part or
    freeze that divergence in place as expected behavior.
    """
    hw = detected.engine_hardware()
    if not hw:
        pytest.skip("the engine extension is not built in this environment")

    from batcher._internal.hardware import available_cpu_count
    from batcher._internal.hardware.topology import numa_node_count, physical_core_count

    assert abs(hw["logical_cores"] - available_cpu_count()) <= 1, (
        f"the engine says {hw['logical_cores']} usable cores and the control plane says "
        f"{available_cpu_count()}; a plan sized against one will be executed with the other"
    )
    assert abs(hw["physical_cores"] - physical_core_count()) <= 1, (
        f"the engine says {hw['physical_cores']} physical cores and the control plane says "
        f"{physical_core_count()}; that is the denominator both use for compute-bound fan-out"
    )
    assert hw["numa_nodes"] == numa_node_count()


def test_both_planes_hold_physical_cores_under_the_cpu_budget():
    """Neither plane may report more physical cores than the process is allowed CPUs.

    `/sys` is host-wide, so a sibling walk sees every core on the box. A cpuset pin narrows it;
    a CFS *bandwidth* quota does not appear in the mask at all, so the walk has to be capped
    explicitly. The engine has always clamped; the control plane did not, and reported the
    host's physical core count for a process throttled to a fraction of it.
    """
    from batcher._internal.hardware import available_cpu_count
    from batcher._internal.hardware.topology import physical_core_count

    budget = available_cpu_count()
    assert 1 <= physical_core_count() <= budget

    hw = detected.engine_hardware()
    if hw:
        assert 1 <= hw["physical_cores"] <= hw["logical_cores"] <= budget
