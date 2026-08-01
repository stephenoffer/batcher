"""Whether a plan that *fits* one GPU is still spread across the fleet.

Fitting one device is a floor, not a target, and the routing used to treat it as the whole
question: a working set smaller than one card's memory was dispatched to one card, whatever the
cluster had. Measured on a four-T4 cluster, a 32M-row group-by estimated at ~0.8 GB took one
device and left three idle, and the sharded fan-out built for exactly that work was never
entered — it is only reached when the data is too *large* for a single device.

The stage is mergeable, so spreading it is a scheduling decision with no semantic content. The
only reason to decline is granularity: a piece too small to be worth its own Ray dispatch. That
bound is `gpu_min_shard_bytes`, which is the same floor `dist.gpu.shards.plan_shard_count` cuts
by — the two must not disagree about what a worthwhile piece is, so both read the one setting.
"""

from __future__ import annotations

import pytest

from batcher.kyber.gpu.spread import devices_worth_using

pytestmark = pytest.mark.unit

#: The granularity floor the decision is taken against, in gigabytes.
FLOOR_GB = (128 << 20) / 1e9


def test_a_working_set_worth_several_pieces_uses_several_devices():
    """Six floors' worth on four devices is four devices, not one."""
    assert devices_worth_using(6 * FLOOR_GB, 4) == 4


def test_a_working_set_smaller_than_one_piece_stays_on_one_device():
    """Below the floor the Ray dispatch costs more than the shard's own compute, so splitting
    it is how a small query pays four dispatches to do one dispatch's work."""
    assert devices_worth_using(FLOOR_GB / 2, 4) == 1


def test_a_working_set_worth_one_piece_is_not_cut_in_two():
    """Floor division, not ceiling: a relation must be worth two *whole* pieces before it is
    split, or one barely over the floor becomes a full shard and a sliver."""
    assert devices_worth_using(FLOOR_GB * 1.5, 4) == 1


def test_the_answer_never_exceeds_the_fleet():
    assert devices_worth_using(1000.0, 4) == 4
    assert devices_worth_using(1000.0, 1) == 1


def test_an_unknown_size_stays_on_one_device():
    """Spreading on a guess is the failure the granularity bound exists to prevent."""
    assert devices_worth_using(0.0, 4) == 1
    assert devices_worth_using(-1.0, 4) == 1


def test_the_spread_grows_with_the_working_set():
    """Monotonic, so a larger relation never lands on fewer devices than a smaller one."""
    counts = [devices_worth_using(n * FLOOR_GB, 8) for n in range(1, 12)]
    assert counts == sorted(counts)


# --- the routing decision that consults it ----------------------------------------------------


def _decision(rows: int, width: int, gpus: int):
    """`decide_gpu_backend` for a shardable group-by over `rows` rows of `width` bytes."""
    import pyarrow as pa

    import batcher as bt
    from batcher import col
    from batcher.kyber.gpu.policy import decide_gpu_backend

    table = pa.table({"k": pa.array([1, 2], pa.int64()), "v": pa.array([1.0, 2.0], pa.float64())})
    ds = bt.from_arrow(table).group_by("k").agg(s=col("v").sum())
    plan, sources = ds._plan, ds._sources

    class _Sized:
        """A source that reports the row count the decision should be taken against."""

        def __getattr__(self, name):
            return getattr(sources[0], name)

        def row_count(self):
            return rows

    return decide_gpu_backend(
        plan, [_Sized()], None, gpu_count=gpus, force=True, gpu_memory_gb=16.0
    )


def test_a_fitting_plan_on_a_multi_gpu_cluster_shards():
    """The decision this file exists for: `distributed` is now True for a plan that fits one
    device, because the fleet has more than one and the work is worth dividing."""
    decision = _decision(rows=40_000_000, width=16, gpus=4)
    assert decision.use_gpu
    assert decision.distributed
    assert decision.desired_gpus > 1
    assert "spread across" in decision.reason


def test_a_fitting_plan_on_a_single_gpu_cluster_does_not_shard():
    decision = _decision(rows=40_000_000, width=16, gpus=1)
    assert decision.use_gpu
    assert not decision.distributed
    assert decision.desired_gpus == 1


def test_a_tiny_plan_still_takes_one_device():
    """The granularity floor has to survive the routing, or every small forced request fans
    out across the fleet to reduce a handful of rows."""
    decision = _decision(rows=11_000_000, width=1, gpus=4)
    assert decision.use_gpu
    assert not decision.distributed
    assert decision.desired_gpus == 1


# --- an explicit request that cannot be honored says so ----------------------------------------


def test_an_explicit_gpu_request_with_no_visible_device_is_audible(caplog, monkeypatch):
    """`backend="gpu"` is always safe, and it was also always silent: on a CPU-only driver every
    explicit request fell back to the CPU engine with no signal but the running time.

    The reason is almost never "there are no GPUs" — the count is read from the live Ray
    topology and only when Ray is initialized in this process, which a plain `collect()` never
    does. That distinction is what the message carries, so it is what is asserted.
    """
    import logging

    import batcher.api.terminal.gpu_backend.route as route

    monkeypatch.setattr(route, "_cluster_gpu_count", lambda: 0)
    monkeypatch.setattr(route, "_NOTED_NO_DEVICE", False)
    with caplog.at_level(logging.WARNING, logger="batcher.api"):
        assert route.try_gpu_collect(object(), [], None, force=True) is None
    assert any('backend="gpu"' in r.message for r in caplog.records)


def test_the_same_request_is_not_repeated_for_every_terminal_op(caplog, monkeypatch):
    """The condition is a property of the cluster, not of the query, so it is said once."""
    import logging

    import batcher.api.terminal.gpu_backend.route as route

    monkeypatch.setattr(route, "_cluster_gpu_count", lambda: 0)
    monkeypatch.setattr(route, "_NOTED_NO_DEVICE", False)
    with caplog.at_level(logging.WARNING, logger="batcher.api"):
        for _ in range(3):
            route.try_gpu_collect(object(), [], None, force=True)
    assert sum('backend="gpu"' in r.message for r in caplog.records) == 1


def test_an_automatic_request_stays_silent(caplog, monkeypatch):
    """`backend="auto"` did not ask for a device, so a cluster without one is not news."""
    import logging

    import batcher.api.terminal.gpu_backend.route as route

    monkeypatch.setattr(route, "_cluster_gpu_count", lambda: 0)
    monkeypatch.setattr(route, "_NOTED_NO_DEVICE", False)
    with caplog.at_level(logging.WARNING, logger="batcher.api"):
        assert route.try_gpu_collect(object(), [], None, force=False) is None
    assert not [r for r in caplog.records if 'backend="gpu"' in r.message]
