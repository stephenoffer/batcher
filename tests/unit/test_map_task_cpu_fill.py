"""A UDF stage's CPU ask must scale with the cluster, not only with the data.

`_adaptive_task_cpus` sizes each map task at `rows_i x weight / rows_per_cpu`, so the
*stage* asks for `total_rows x weight / rows_per_cpu` — a number that does not mention the
fleet. Cutting the source into more partitions does not change it either, because the count
and the per-task share move inversely. On a cluster wider than the data implies, the
difference is cores left idle for the whole query.

Measured on the 64 x 16-core cluster (1,024 cores), a 300-pass NumPy UDF over TPC-H sf100
`orders` (150M rows), which asks for `150M x 4 / 2M` = 300 cores:

| stage CPU ask | wall | cluster busy |
|---|---:|---:|
| 300 (the data-derived ask) | 3,375 ms | 16.2% |
| **~1,024 (filled)**        | **2,382 ms** | **48.6%** |
| ~2,048 (twice the fleet)   | 2,963 ms | 57.2% |

Every node was active in all three, so this is not a placement problem — it is how much of
each node the stage was allowed to occupy. The last row is why the fill is capped at the
cluster's own cores: past that it buys CPU busy-ness and loses wall time.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.map import _filled_to_the_fleet

pytestmark = pytest.mark.unit


def _cores(monkeypatch, n: float) -> None:
    monkeypatch.setattr("batcher.dist.executors.map._cluster_cores", lambda: n)


def test_a_stage_asking_for_less_than_the_fleet_is_scaled_up_to_it(monkeypatch):
    _cores(monkeypatch, 1024.0)
    wants = [4.0] * 75  # the measured shape: 300 of 1,024 cores
    filled = _filled_to_the_fleet(wants, node_cores=16.0, learned=1.0)
    assert sum(filled) == pytest.approx(1024.0)
    assert all(f == pytest.approx(filled[0]) for f in filled)


def test_the_skew_between_partitions_survives_the_fill(monkeypatch):
    """Proportional, so a partition with twice the rows keeps twice the share."""
    _cores(monkeypatch, 1024.0)
    filled = _filled_to_the_fleet([1.0, 2.0, 4.0, 8.0], node_cores=1000.0, learned=1.0)
    assert filled[1] == pytest.approx(2 * filled[0])
    assert filled[3] == pytest.approx(8 * filled[0])
    assert sum(filled) == pytest.approx(1024.0)


def test_a_stage_that_already_meets_the_fleet_is_untouched(monkeypatch):
    """Scales up only. A stage sized past the cluster keeps its own sizing."""
    _cores(monkeypatch, 64.0)
    wants = [8.0] * 16  # 128 against a 64-core fleet
    assert _filled_to_the_fleet(wants, node_cores=16.0, learned=1.0) == wants


def test_the_fill_never_asks_for_a_bundle_no_node_can_host(monkeypatch):
    """A share past `node_cores` is unplaceable, so the scale is capped by it.

    The shape that forces it: a huge fleet and a handful of partitions. Filling
    proportionally would put 256 cores on each of four tasks; no node has that.
    """
    _cores(monkeypatch, 1024.0)
    filled = _filled_to_the_fleet([1.0] * 4, node_cores=16.0, learned=1.0)
    assert max(filled) <= 16.0
    assert sum(filled) == pytest.approx(64.0)


@pytest.mark.parametrize("wants", [[], [0.0, 0.0]])
def test_nothing_to_scale_is_returned_unchanged(monkeypatch, wants):
    _cores(monkeypatch, 1024.0)
    assert _filled_to_the_fleet(list(wants), node_cores=16.0, learned=1.0) == wants


def test_an_unreadable_cluster_leaves_the_sizing_alone(monkeypatch):
    """`_cluster_cores` falls back to the local count; a zero must not erase the shares."""
    _cores(monkeypatch, 0.0)
    wants = [2.0, 3.0]
    assert _filled_to_the_fleet(wants, node_cores=16.0, learned=1.0) == wants


def test_a_family_measured_as_cpu_idle_is_not_filled_back_up(monkeypatch):
    """The fill must not undo `_learned_weight_factor`, which is the measured term.

    A family recorded as using a quarter of its reserved cores has its reservation cut to a
    quarter; filling it to the whole fleet afterwards would hand an IO- or GPU-bound stage
    every core in the cluster to leave idle. The fill target carries the same factor.
    """
    _cores(monkeypatch, 1024.0)
    full = _filled_to_the_fleet([4.0] * 64, node_cores=16.0, learned=1.0)
    idle = _filled_to_the_fleet([1.0] * 64, node_cores=16.0, learned=0.25)
    assert sum(full) == pytest.approx(1024.0)
    assert sum(idle) == pytest.approx(256.0)
    assert sum(idle) < sum(full)


def test_only_a_udf_stage_is_filled(monkeypatch):
    """A scan task cannot use a second core for its own partition, so it is not given one.

    This is the guard that keeps the fill from reserving idle cores on a shared cluster: it
    is applied in `_adaptive_task_cpus` under `has_map_batches`, not inside the scaler.
    """
    import batcher as bt
    from batcher.dist.executors.map import _adaptive_task_cpus

    _cores(monkeypatch, 1024.0)
    monkeypatch.setattr("batcher.dist.executors.map._placeable_node_cores", lambda: 16.0)
    monkeypatch.setattr("batcher.dist.executors.map._learned_weight_factor", lambda p, h=None: 1.0)
    monkeypatch.setattr("batcher.dist.executors.map.descriptor_rows", lambda p: 2_000_000)

    scan = bt.from_pydict({"a": [1]}).filter(bt.col("a") > 0)._plan
    udf = bt.from_pydict({"a": [1]}).map_batches(lambda b: b)._plan
    # 128 partitions so the per-node cap (16 x 128) cannot be what binds the fill.
    parts = [object()] * 128
    # 2M rows against a 2M `rows_per_cpu` is one core a task at weight 1, four at the UDF
    # weight — 128 and 512 cores respectively, both under this fleet's 1,024.
    assert sum(_adaptive_task_cpus(parts, scan)) == pytest.approx(128.0)
    assert sum(_adaptive_task_cpus(parts, udf)) == pytest.approx(1024.0)
