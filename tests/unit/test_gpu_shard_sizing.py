"""How many shards a GPU fan-out cuts, and why cutting more is not free.

Shard count was `#devices x oversubscribe` whatever the input. On a sixteen-device fleet that
is sixty-four shards of everything — including a six-million-row scan, where each shard is a
Ray dispatch, a cuDF first touch and a device-allocator setup delivering about a hundred
thousand rows to a kernel. Measured on this cluster, TPC-H q6 at sf1 took **196 seconds** that
way against 0.12 s on the CPU engine, and essentially all of it was per-task fixed cost.

The count now has three bounds — memory, parallelism and granularity — and these assert each of
them separately, because a single "reasonable number" assertion would pass for the wrong
reason. The granularity bound is the one that was missing, and it is the one that has to
override the parallelism bound rather than be averaged with it: a relation too small to fill
every device should run on one, not on all of them badly.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pytest

from batcher.config import active_config, set_config
from batcher.dist.gpu.shards import plan_shard_count

pytestmark = pytest.mark.unit

MIB = 1 << 20
GIB = 1 << 30
#: One T4, which is the fleet these numbers were measured on.
DEVICE = 15e9


@contextlib.contextmanager
def config_scope(**overrides):
    """The distributed config with `overrides` applied, restored on exit."""
    saved = active_config()
    try:
        set_config(saved.replace(distributed=dataclasses.replace(saved.distributed, **overrides)))
        yield
    finally:
        set_config(saved)


def test_a_small_relation_runs_on_one_shard():
    """The regression. Sixteen devices, six million rows: one shard, not sixty-four."""
    assert plan_shard_count(6 * MIB, gpu_count=16, device_bytes=DEVICE) == 1


def test_shards_never_fall_below_the_granularity_floor():
    """Four floors' worth of data is four shards, however many devices are waiting."""
    with config_scope(gpu_min_shard_bytes=128 * MIB):
        assert plan_shard_count(4 * 128 * MIB, gpu_count=16, device_bytes=DEVICE) == 4


def test_a_relation_that_fills_the_fleet_uses_every_device():
    """Past the floor, parallelism takes over: enough data for sixteen shards gets sixteen."""
    with config_scope(gpu_min_shard_bytes=128 * MIB):
        assert plan_shard_count(16 * GIB, gpu_count=16, device_bytes=DEVICE) == 16


def test_a_relation_larger_than_the_fleets_memory_is_cut_by_memory():
    """Each shard must fit a device with room for what the operators build on it, so the
    count is driven past the device count once the data is."""
    with config_scope(gpu_shard_expansion=2.0, gpu_shard_oversubscribe=8):
        # 15 GB device / 2x expansion = 7.5 GB per shard; 240 GB needs 32 of them.
        assert plan_shard_count(240e9, gpu_count=16, device_bytes=DEVICE) == 32


def test_the_oversubscribe_ceiling_still_caps_the_count():
    """However large the relation, a fan-out does not cut more than the fleet may pipeline."""
    with config_scope(gpu_shard_oversubscribe=4):
        assert plan_shard_count(100e12, gpu_count=16, device_bytes=DEVICE) == 64


def test_an_unknown_size_keeps_the_device_driven_count():
    """A source that cannot report its rows gets exactly the sizing this path used before
    anything was measured — never a fabricated figure."""
    with config_scope(gpu_shard_oversubscribe=4):
        assert plan_shard_count(0, gpu_count=16, device_bytes=DEVICE) == 64


def test_an_unreadable_device_still_sizes_on_granularity():
    """No device memory figure is not a reason to over-shard: the granularity bound needs
    none of it."""
    assert plan_shard_count(6 * MIB, gpu_count=16, device_bytes=0) == 1


def test_the_count_is_always_at_least_one():
    """An empty relation still has to be read by somebody."""
    assert plan_shard_count(1, gpu_count=16, device_bytes=DEVICE) == 1


class TestSourceBytes:
    """Pricing a relation. The estimate feeds both the shard count and the replication bound,
    so it has to narrow with the projection or a pruned read looks as expensive as a full one."""

    @staticmethod
    def _source(rows: int):
        import pyarrow as pa

        import batcher as bt

        table = pa.table(
            {
                "a": pa.array(range(rows), type=pa.int64()),
                "b": pa.array([float(i) for i in range(rows)], type=pa.float64()),
                "c": pa.array([str(i) for i in range(rows)], type=pa.string()),
            }
        )
        return bt.from_arrow(table)._sources[0]

    def test_a_projection_narrows_the_estimate(self):
        from batcher.dist.gpu.shards import source_bytes

        source = self._source(1000)
        assert source_bytes(source, ["a"]) < source_bytes(source, None)

    def test_two_fixed_width_columns_are_priced_by_their_widths(self):
        from batcher.dist.gpu.shards import source_bytes

        # int64 + float64 = 16 bytes a row, and nothing about that is an estimate.
        assert source_bytes(self._source(1000), ["a", "b"]) == pytest.approx(16_000)

    def test_an_unknown_column_does_not_raise(self):
        """Pricing is best-effort; a name the schema does not have must not fail a query."""
        from batcher.dist.gpu.shards import source_bytes

        assert source_bytes(self._source(10), ["nope"]) > 0
