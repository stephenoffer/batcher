"""The GPU fan-out's *wiring*, run for real on Ray with the device backend swapped for pandas.

Every other GPU case in the suite checks what a device computes. None of them checked that the
fan-out around it is connected: that a source can be described to a worker, that the descriptor
reaches a task, that the barrier collects the shards, that a failed shard takes the right rung
of the recovery ladder, and that the merge reassembles them.

That gap was not hypothetical. `_scan_splits` stopped being re-exported from `partition_io`,
and because the fan-out's import of it sat inside an `except Exception: return None`, every
multi-device path answered "this source cannot be fanned out" and fell back to a single device.
Correct results, no error, no test failure, and the entire feature disabled. These cases exist
so that cannot happen quietly again.

The device backend is pandas here and the tasks ask for no GPU, so this runs anywhere. What it
exercises is the scheduling, which is the part that was broken; what a device computes is
covered by the translator's own differential suite.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray", reason="the fan-out is Ray scheduling")


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    """An isolated local Ray cluster, so this never joins one the environment already has."""
    import os

    os.environ.pop("RAY_ADDRESS", None)
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=str(tmp_path_factory.mktemp("ray")),
    )
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture(scope="module")
def source_dir(tmp_path_factory):
    """A multi-file, multi-row-group parquet directory — a genuinely splittable source.

    Splittability is the whole point: an in-memory source has no splits to describe, so it can
    never exercise the descriptor path that broke.
    """
    out = tmp_path_factory.mktemp("pq")
    rng = np.random.default_rng(5)
    for i in range(4):
        table = pa.table(
            {
                "g": rng.integers(0, 6, 2000).astype("int64"),
                "v": rng.random(2000) * 100,
            }
        )
        pq.write_table(table, out / f"part{i}.parquet", row_group_size=500)
    return str(out)


def _host_shard(descriptor, ops):
    """What the GPU shard task does, on the host backend instead of cuDF."""
    import pandas as pd

    from batcher.core.gpu_plan import DfBackend
    from batcher.dist.gpu.tasks import run_shard_chain

    return run_shard_chain(descriptor, ops, DfBackend(pd))


@pytest.fixture
def host_tasks(monkeypatch):
    """Point the fan-out's device task at the host backend, and ask for no GPU."""
    import batcher.dist.gpu.tasks as tasks

    monkeypatch.setattr(tasks, "gpu_shard_partial", _host_shard)
    monkeypatch.setattr(tasks, "gpu_task_options", lambda: {"num_gpus": 0, "max_retries": 0})


def _rows(table: pa.Table, *, ordered: bool) -> list[tuple]:
    def canon(v):
        if isinstance(v, float):
            return "__nan__" if v != v else float(f"{v:.12e}")
        return v

    out = [tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)]
    return out if ordered else sorted(out, key=repr)


CASES = {
    "aggregate": (lambda ds: ds.group_by("g").agg(s=col("v").sum(), n=bt.count()), False),
    "row_local": (lambda ds: ds.filter(col("v") > 50.0), True),
    "top_n": (lambda ds: ds.sort("v", descending=True).limit(7), True),
    "distinct": (lambda ds: ds.select("g").distinct(), False),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("sharded", [False, True])
def test_the_fan_out_reproduces_the_single_node_answer(
    case, sharded, cluster, source_dir, host_tasks
):
    """The wiring end to end: descriptors, tasks, the barrier, and the merge."""
    from batcher.core.gpu_plan import gpu_plan_ops
    from batcher.dist.gpu.aggregate import sharded_gpu_aggregate

    build, ordered = CASES[case]
    ds = build(bt.read.parquet(source_dir))
    scan, ops = gpu_plan_ops(ds._plan)
    got = sharded_gpu_aggregate(ds._sources[scan.source_id], ops, gpu_count=2, sharded=sharded)
    assert got is not None, "the fan-out declined a splittable source"
    expected = ds.collect()
    assert _rows(got.select(expected.column_names), ordered=ordered) == _rows(
        expected, ordered=ordered
    )


def test_a_splittable_source_can_be_described_to_a_worker(cluster, source_dir):
    """The specific thing that broke: a source that *is* splittable must describe itself.

    Asserted separately from the fan-out because the failure mode is a `None` that reads as a
    legitimate "not splittable" — indistinguishable, from the outside, from an import that
    moved.
    """
    from batcher.dist.gpu.aggregate import shard_descriptors
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    source = bt.read.parquet(source_dir)._sources[0]
    assert whole_source_descriptor(source) is not None
    fanned = shard_descriptors(source, 2, sharded=True, preserve_order=False)
    assert fanned is not None and len(fanned) > 1


def test_an_in_memory_source_declines(cluster):
    """...and one that genuinely cannot be split still says so, which is the other half."""
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    assert whole_source_descriptor(bt.from_pydict({"a": [1, 2, 3]})._sources[0]) is None


@pytest.mark.parametrize(
    ("failure", "expected_field"),
    [
        # a shard that does not fit is subdivided and rerun on the device
        ("rmm::bad_alloc: out_of_memory", "subdivided"),
        # anything else is recomputed by the native CPU engine
        ("worker died unexpectedly", "recovered_on_cpu"),
    ],
)
def test_a_failing_shard_still_produces_the_right_answer(
    failure, expected_field, cluster, source_dir, monkeypatch
):
    """The recovery ladder, with the failure injected into the task itself.

    Both rungs must reach the same answer as a clean run — the whole claim of the fan-out — and
    both must say they were taken, because a degraded run is otherwise indistinguishable from a
    healthy one.
    """
    import batcher.dist.gpu.tasks as tasks
    from batcher._internal import events
    from batcher.core.gpu_plan import gpu_plan_ops
    from batcher.dist.gpu.aggregate import sharded_gpu_aggregate

    def _failing(descriptor, ops):
        pieces = descriptor.get("splits", descriptor.get("batches", []))
        # A memory failure clears once the shard is small enough; anything else never does.
        if "out_of_memory" not in failure or len(pieces) > 1:
            raise RuntimeError(failure)
        return _host_shard(descriptor, ops)

    monkeypatch.setattr(tasks, "gpu_shard_partial", _failing)
    monkeypatch.setattr(tasks, "gpu_task_options", lambda: {"num_gpus": 0, "max_retries": 0})

    ds = bt.read.parquet(source_dir).group_by("g").agg(s=col("v").sum(), n=bt.count())
    scan, ops = gpu_plan_ops(ds._plan)
    seen: list = []
    unsubscribe = events.subscribe(seen.append)
    try:
        got = sharded_gpu_aggregate(ds._sources[scan.source_id], ops, gpu_count=2, sharded=True)
    finally:
        if callable(unsubscribe):
            unsubscribe()

    assert got is not None
    expected = ds.collect()
    assert _rows(got.select(expected.column_names), ordered=False) == _rows(expected, ordered=False)
    degraded = [e.fields for e in seen if e.fields.get("event") == "shard_degraded"]
    assert degraded, "a degraded run must say so"
    assert degraded[0][expected_field] > 0
