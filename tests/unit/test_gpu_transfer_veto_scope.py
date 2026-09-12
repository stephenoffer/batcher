"""The transfer model must not charge a host copy this tier no longer makes.

`kyber.gpu.policy`'s veto asks whether a stage is transfer-bound: it charges the decoded working
set as a host-to-device copy and refuses the device when the copy costs more than the kernels
save. That is the right model for a frame the driver hands a device.

It is the wrong model for a scan the device reads off storage, and the difference is not
marginal. On six T4s the veto refused **every** TPC-H query at sf10 — "T4 would run this at
0.53x the CPU once the host copy is charged (97% of device time is transfer)" — while the same
queries measured up to 2.35x on the device. `dist/gpu/device_read.py` exists precisely to remove
that copy: what crosses PCIe is the compressed Parquet, and on a frame-cache hit nothing crosses.

So the veto is applied only where the host really is in the path. The direction of the
conservatism matters: skipping it wrongly costs a GPU attempt that falls back to the CPU engine,
and applying it wrongly costs the accelerator entirely.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io import InMemorySource
from batcher.io.splits.device import reads_on_device
from batcher.kyber.gpu.policy import _device_reads_its_own_input

pytestmark = pytest.mark.unit


def _parquet(tmp_path, name="t.parquet"):
    import batcher as bt

    path = tmp_path / name
    pq.write_table(pa.table({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]}), path)
    return bt.read.parquet(str(path))


# --- the io predicate --------------------------------------------------------


def test_a_parquet_source_reads_on_the_device(tmp_path):
    assert reads_on_device(_parquet(tmp_path)._sources[0]) is True


def test_an_in_memory_source_does_not():
    """Its rows are already decoded on the driver; a device gets them across the link."""
    assert reads_on_device(InMemorySource([pa.record_batch({"x": [1]})])) is False


def test_a_source_that_will_not_split_does_not():
    class _Opaque:
        def splits(self, target_size=None):
            raise RuntimeError("no")

    assert reads_on_device(_Opaque()) is False


def test_a_source_with_no_splits_does_not():
    class _Empty:
        def splits(self, target_size=None):
            return []

    assert reads_on_device(_Empty()) is False


# --- the plan-level question -------------------------------------------------


def test_a_plan_over_one_parquet_scan(tmp_path):
    ds = _parquet(tmp_path).filter(__import__("batcher").col("v") > 1.0)
    assert _device_reads_its_own_input(ds._plan, ds._sources) is True


def test_a_join_of_two_parquet_scans(tmp_path):
    left, right = _parquet(tmp_path, "l.parquet"), _parquet(tmp_path, "r.parquet")
    joined = left.join(right, on="k")
    assert _device_reads_its_own_input(joined._plan, joined._sources) is True


def test_one_in_memory_side_is_enough_to_keep_the_veto(tmp_path):
    """`all`, not `any`: the copy is charged if *any* input crosses the link decoded."""
    import batcher as bt

    joined = _parquet(tmp_path).join(bt.from_pydict({"k": [1], "w": [9.0]}), on="k")
    assert _device_reads_its_own_input(joined._plan, joined._sources) is False


def test_a_plan_with_no_scan_keeps_the_veto():
    """Nothing to establish, so nothing is claimed."""
    assert _device_reads_its_own_input(None, []) is False


def test_an_unreadable_plan_keeps_the_veto():
    assert _device_reads_its_own_input(object(), []) is False
