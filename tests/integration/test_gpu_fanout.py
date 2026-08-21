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

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray", reason="the fan-out is Ray scheduling")
cloudpickle = pytest.importorskip("ray.cloudpickle", reason="the fan-out is Ray scheduling")


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    """An isolated local Ray cluster, so this never joins one the environment already has."""
    import os

    # Dropped so `ray.init(address="local")` really is local, and **restored afterwards**:
    # this is a process-wide environment variable, and a module that leaves it unset makes
    # every later module start its own single-node Ray instead of attaching to the session's
    # cluster. That is invisible here and shows up three files later as a preemption or
    # placement test failing on a cluster too small to schedule what it asks for.
    prior_address = os.environ.pop("RAY_ADDRESS", None)
    # Serialize this module's helpers **by value**, not by reference.
    #
    # The shard stubs below (`_host_shard`, and the injected `_failing`) are module-level
    # functions of a pytest test module, and cloudpickle pickles a module-level function by
    # reference when its module looks importable. The driver can import `test_gpu_fanout`
    # because pytest put `tests/integration` on *its* `sys.path`; a Ray worker cannot, and
    # every shard task died with `ModuleNotFoundError: No module named 'test_gpu_fanout'`.
    #
    # That failure was invisible, and in exactly the way this module exists to prevent: the
    # recovery ladder caught it, recomputed the shard on the CPU engine, and every assertion
    # about the *answer* still passed — so the whole file was green while the fan-out it
    # tests never ran once. Registering by value makes the worker deserialize the function
    # body instead of importing it.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
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
        cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
        ray.shutdown()
        if prior_address is not None:
            os.environ["RAY_ADDRESS"] = prior_address


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
    """Point the fan-out's device task at the host backend, and ask for no GPU.

    The stub takes `num_gpus` because the real `gpu_task_options` does: a packed fan-out asks
    for a fraction of a device rather than a whole one. A stub that accepts only the old
    no-argument call fails every case here with a `TypeError` from inside the fixture, which
    reads as the fan-out being broken rather than the double being stale.
    """
    import batcher.dist.gpu.resources as resources
    import batcher.dist.gpu.tasks as tasks
    from batcher.carbonite.accel.fractional import whole_device_packing

    host_opts = {"num_gpus": 0, "max_retries": 0}
    monkeypatch.setattr(tasks, "gpu_shard_partial", _host_shard)
    monkeypatch.setattr(tasks, "gpu_task_options", lambda num_gpus=0.0, **_: dict(host_opts))
    # `gpu_shard_options` is where the fan-out gets the options it launches *shards* with;
    # `gpu_task_options` only covers the un-packed retry handle. Patching one and not the other
    # leaves every shard asking Ray for a real device, and this suite runs on a local cluster
    # that has none — so the tasks pend forever and the case hangs instead of failing.
    monkeypatch.setattr(
        resources,
        "gpu_shard_options",
        lambda descriptors, schema=None, **_: (dict(host_opts), whole_device_packing(1, "test")),
    )


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


@pytest.mark.timeout(90)
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("sharded", [False, True])
def test_the_fan_out_reproduces_the_single_node_answer(
    case, sharded, cluster, source_dir, host_tasks
):
    """The wiring end to end: descriptors, tasks, the barrier, and the merge.

    Bounded explicitly because the failure mode here is a *hang*, not an error: the fan-out's
    barrier waits on shard tasks with no deadline, so when they cannot be placed the case stops
    rather than fails. That was hidden until recently — the `host_tasks` double had drifted out
    of step with `gpu_task_options`, so every case died on a `TypeError` before reaching the
    barrier and the wait was never entered. Fixing the double is what exposed it. The timeout
    turns the hang back into a reported failure; the barrier itself wanting a deadline is a
    separate finding and is not fixed here.
    """
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
    import dataclasses

    from batcher.config import active_config, config_context
    from batcher.dist.gpu.aggregate import shard_descriptors
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    source = bt.read.parquet(source_dir)._sources[0]
    assert whole_source_descriptor(source) is not None
    # This fixture's source is a few hundred kilobytes, and `plan_shard_count` will not cut a
    # shard below `gpu_min_shard_bytes` (128 MB by default) because the Ray dispatch would cost
    # more than the shard's own compute. That floor is right and separately tested; it just
    # means a test-sized relation is *correctly* one shard. Lower it so this case can still
    # exercise the thing it is about, which is the descriptor wiring, not the sizing policy.
    # Only the one field: a fresh `DistributedConfig()` would also reset the autoscale wait and
    # the placement timeout this suite's local cluster depends on, and the case would hang
    # rather than fail.
    cfg = active_config()
    tiny_floor = cfg.replace(
        distributed=dataclasses.replace(cfg.distributed, gpu_min_shard_bytes=1)
    )
    with config_context(tiny_floor):
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
    # Takes `num_gpus` like the real `gpu_task_options` does — the fan-out sizes the device
    # share from its packing decision, so a zero-arg stub is not a stand-in for it.
    monkeypatch.setattr(
        tasks,
        "gpu_task_options",
        lambda num_gpus=1.0, **_: {"num_gpus": 0, "max_retries": 0},
    )

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
