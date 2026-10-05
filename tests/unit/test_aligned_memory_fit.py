"""The aligned executor runs as many units a node at once as its memory holds, not only its cores.

`dist.executors.aligned.memory_fit.fit_units` decides it from each node's CPU count and Ray
memory. The case it exists for is TPC-H q9 at SF1000 on 64 GB nodes: two 8-core tasks per node
over ~6 GB units held 28 GB each and were OOM-killed in a loop. Each test names the constraint
that binds, so a fit that ignored one of them would fail the test for it.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.aligned.memory_fit import UNIT_FOOTPRINT, fit_units

pytestmark = pytest.mark.unit

_GB = 1 << 30


def test_small_units_keep_the_core_sizing():
    fit = fit_units([(16, 42 * _GB)] * 4, unit_cpus=8, slots=8, largest_unit=256 << 20)
    assert (fit.unit_cpus, fit.slots, fit.per_node) == (8, 8, 2)
    assert fit.memory_bytes == 21 * _GB  # each task's share of the node is its engine budget


def test_a_large_unit_runs_one_task_per_node_on_every_core():
    # The q9 shape: 6 GB units need ~27 GB each, so a 42 GB node holds one, not two.
    fit = fit_units([(16, 42 * _GB)] * 4, unit_cpus=8, slots=8, largest_unit=6 * _GB)
    assert (fit.unit_cpus, fit.slots, fit.per_node) == (16, 4, 1)
    assert fit.memory_bytes == 42 * _GB
    assert fit.memory_bytes >= 6 * _GB * UNIT_FOOTPRINT


def test_a_unit_too_large_for_any_node_still_runs_one_per_node():
    # Fewer than one cannot run at all; the engine budget makes the overrun spill instead.
    fit = fit_units([(16, 20 * _GB)] * 2, unit_cpus=8, slots=4, largest_unit=10 * _GB)
    assert (fit.unit_cpus, fit.slots, fit.per_node) == (16, 2, 1)
    assert fit.memory_bytes == 20 * _GB


def test_the_tightest_node_decides():
    nodes = [(16, 100 * _GB), (16, 30 * _GB)]
    fit = fit_units(nodes, unit_cpus=8, slots=4, largest_unit=4 * _GB)  # needs 18 GB a task
    assert fit.per_node == 1 and fit.memory_bytes == 30 * _GB


def test_an_unmeasured_cluster_keeps_the_core_sizing_and_sets_no_budget():
    fit = fit_units([(16, 0)] * 4, unit_cpus=8, slots=8, largest_unit=6 * _GB)
    assert (fit.unit_cpus, fit.slots, fit.memory_bytes) == (8, 8, None)
    fit = fit_units([], unit_cpus=8, slots=8, largest_unit=6 * _GB)
    assert (fit.unit_cpus, fit.slots, fit.memory_bytes) == (8, 8, None)


def test_nodes_too_small_for_a_task_are_not_counted():
    # The 0-CPU head node holds no unit task, so its memory must not tighten the fit.
    nodes = [(0, 4 * _GB), (16, 42 * _GB), (16, 42 * _GB)]
    fit = fit_units(nodes, unit_cpus=8, slots=4, largest_unit=256 << 20)
    assert (fit.slots, fit.per_node, fit.memory_bytes) == (4, 2, 21 * _GB)


def test_a_node_with_more_cores_but_no_more_memory_runs_what_its_memory_holds():
    # 64 cores beside 16 on the same 64 GB: cores alone would put 8 tasks on the large node,
    # each handed the 32 GB budget, four times what the node has.
    # A 4 GB unit, so two tasks' footprint (`UNIT_FOOTPRINT`) fits the 64 GB node.
    nodes = [(64, 64 * _GB), (16, 64 * _GB)]
    fit = fit_units(nodes, unit_cpus=8, slots=10, largest_unit=4 * _GB)
    assert fit.per_node == 2 and fit.memory_bytes == 32 * _GB
    assert fit.slots == 4


def test_a_node_with_more_memory_keeps_its_cores_busy():
    # Memory is not the constraint on the large node, so it keeps a task per 8 cores.
    nodes = [(64, 512 * _GB), (16, 64 * _GB)]
    fit = fit_units(nodes, unit_cpus=8, slots=10, largest_unit=256 << 20)
    assert fit.memory_bytes == 32 * _GB
    assert fit.slots == 8 + 2


def test_a_unit_task_prefetches_only_while_the_node_keeps_its_floor(monkeypatch):
    """The next unit is read early only if holding it leaves the engine's headroom floor intact.

    Read through the engine's own guard (`memory_headroom`), so the figure and the floor are
    the ones the executors trip on. Unreadable (no reading) keeps the old behaviour: prefetch.
    """
    from batcher.dist.executors.aligned import run

    class _Engine:
        reading: tuple[int, int] | None = (40 * _GB, 4 * _GB)

        def memory_headroom(self):
            return self.reading

    fake = _Engine()
    monkeypatch.setattr(run, "engine", lambda: fake)
    assert run._room_to_prefetch(6 * _GB)  # 40 - 2x6 = 28 GB left, well above 4
    fake.reading = (14 * _GB, 4 * _GB)
    assert not run._room_to_prefetch(6 * _GB)  # 14 - 12 = 2 GB: under the floor
    fake.reading = None
    assert run._room_to_prefetch(6 * _GB)
