"""Stacked map stages fuse into one actor, so a CPU stage's width must not size a GPU pool.

`map_batches(Decode, concurrency=32).map_batches(Model, num_gpus=1, concurrency=8)` is the
two-stage batch-inference pipeline every guide recommends, and it asked for **32 actors holding
a GPU apiece** on an 8-GPU cluster: `_map_resources` merges stacked stages by taking the widest
concurrency, which is right while they share a resource class and wrong the moment they do not.

It does not fail. The gang is unsatisfiable, so the pool thrashes -- measured on a live 8-GPU
fleet: 10 actors `PENDING_CREATION`, 1 alive, 136 dead from spawn-and-kill churn -- and the
query hangs with no error. The accelerator stage's own concurrency is the one that can be
honoured, so it is the one that must win.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.executors.map import _map_resources

pytestmark = pytest.mark.unit


class Decode:
    """A CPU stage: wide, because decoding wants the cluster."""

    def __call__(self, batch):
        return {"px": batch["v"]}


class Model:
    """A device stage: narrow, because it wants one actor per device."""

    def __call__(self, batch):
        return {"pred": batch["px"]}


def _chain(cpu_concurrency, gpu_concurrency, num_gpus=1.0, resources=None):
    ds = bt.from_arrow(pa.table({"v": pa.array([1.0, 2.0])}))
    ds = ds.map_batches(
        Decode, output_columns=["px"], batch_format="numpy", concurrency=cpu_concurrency
    )
    kwargs = {"concurrency": gpu_concurrency, "num_gpus": num_gpus}
    if resources:
        kwargs["resources"] = resources
    return ds.map_batches(Model, output_columns=["pred"], batch_format="numpy", **kwargs)._plan


def test_the_device_stage_sizes_the_fused_pool():
    num_gpus, wants_pool, concurrency, _accel, _res = _map_resources(_chain(32, 8))
    assert num_gpus == 1
    assert wants_pool is True
    assert concurrency == 8, (
        f"the fused pool asked for {concurrency} actors each holding a device; the CPU stage's "
        "width is not a device count"
    )


def test_a_custom_accelerator_stage_sizes_it_too():
    # `num_gpus` covers only what Ray calls `GPU`; a TPU / Trainium / Gaudi stage names a
    # custom resource instead and is just as un-widenable by a CPU neighbour.
    _gpus, _pool, concurrency, _accel, resources = _map_resources(
        _chain(48, 4, num_gpus=0.0, resources={"TPU": 4.0})
    )
    assert resources == {"TPU": 4.0}
    assert concurrency == 4


def test_a_cpu_only_stack_still_takes_the_widest():
    # The original rule, and it is right where it applies: two host stages in one actor can
    # both use the whole node, so the wider request is the one to honour.
    _gpus, _pool, concurrency, _accel, _res = _map_resources(_chain(32, 8, num_gpus=0.0))
    assert concurrency == (32, 32)


def test_a_device_stage_with_no_concurrency_leaves_the_merge_alone():
    # Nothing to prefer: the accelerator stage named no size, so the CPU stage's stands and
    # the pool sizing falls to `gpu_aware_pool_default` as before.
    _gpus, _pool, concurrency, _accel, _res = _map_resources(_chain(12, None))
    assert concurrency == 12
