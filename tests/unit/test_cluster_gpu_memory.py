"""A distributed run must size GPU work from the *cluster's* devices, not the driver's.

Ray reports an accelerator's count and model name but never its memory, so `HardwareProfile`
carried a `gpu_memory_bytes` field that the cluster path never populated. Every GPU sizing
decision therefore fell through to `DistributedConfig.resolved_gpu_memory_gb()`, which probes
the **local process**. On the usual topology — a CPU-only head node scheduling GPU workers —
that probe finds no device and returns a hardcoded 12 GB T4.

The failure is silent in both directions and that is what makes it worth pinning: an A100
fleet planned as a T4 shards (or refuses as "exceeds all GPUs") a working set one device would
have held and seeds inference batches ~6x too small, while a fat GPU driver next to small
workers over-estimates and OOMs them. Nothing errors either way.
"""

from __future__ import annotations

import pytest

from batcher._internal.accelerators import accelerator_memory_bytes

pytestmark = pytest.mark.unit

_GIB = 1 << 30


@pytest.mark.parametrize(
    ("name", "expected_gib"),
    [
        ("NVIDIA_TESLA_T4", 16),
        ("NVIDIA_A100", 40),  # the smallest variant sold under this name
        ("NVIDIA_A100_80G", 80),
        ("NVIDIA_H100", 80),
        ("AMD_INSTINCT_MI300X", 192),
        # Ray's casing is inconsistent across vendors, so lookup normalizes.
        ("AMD_Instinct_MI300X", 192),
    ],
)
def test_known_accelerator_models_resolve_to_their_memory(name, expected_gib):
    assert accelerator_memory_bytes(name) == expected_gib * _GIB


@pytest.mark.parametrize("name", [None, "", "NVIDIA_SOMETHING_UNRELEASED"])
def test_unknown_models_report_unknown_rather_than_guessing(name):
    """`0` is `HardwareProfile`'s "unknown" sentinel, so an unrecognized device leaves the
    caller on its existing default instead of a fabricated figure."""
    assert accelerator_memory_bytes(name) == 0


def _profile(monkeypatch, classes):
    """`cluster_hardware_profile()` over a synthetic topology."""
    from batcher.dist.executors.ray_runtime import scaling

    monkeypatch.setattr(scaling, "node_classes", lambda: classes)
    monkeypatch.setattr(scaling, "worker_node_memory_bytes", lambda: 64 * _GIB)
    return scaling.cluster_hardware_profile()


def test_vram_binds_to_the_smallest_device_in_a_mixed_fleet(monkeypatch):
    """Sizing to the largest device would OOM every smaller one it lands on."""
    hw = _profile(
        monkeypatch,
        [
            {"cpus": 32.0, "gpus": 8.0, "accelerator_type": "NVIDIA_A100"},
            {"cpus": 16.0, "gpus": 4.0, "accelerator_type": "NVIDIA_TESLA_T4"},
        ],
    )
    assert hw is not None
    assert hw.gpu_memory_bytes == 16 * _GIB  # the T4, not the A100


def test_gpu_count_is_devices_not_gpu_bearing_nodes(monkeypatch):
    """The regression: the cluster path reported the number of GPU *nodes*, but the figure is
    consumed as a device count (`one_gpu_gb * gpu_count` is the whole-fleet VRAM budget), so
    two 8-GPU boxes were planned as two GPUs and refused work the cluster could hold."""
    hw = _profile(
        monkeypatch,
        [
            {"cpus": 32.0, "gpus": 8.0, "accelerator_type": "NVIDIA_A100"},
            {"cpus": 32.0, "gpus": 8.0, "accelerator_type": "NVIDIA_A100"},
        ],
    )
    assert hw is not None
    assert hw.gpu_count == 16


def test_one_unknown_device_makes_the_whole_fleet_vram_unknown(monkeypatch):
    """A partial minimum is not a bound: the unlabelled device could be smaller than every
    device that *was* recognized, so reporting the known minimum would be a figure that only
    looks authoritative. Unknown falls back to the caller's default."""
    hw = _profile(
        monkeypatch,
        [
            {"cpus": 32.0, "gpus": 8.0, "accelerator_type": "NVIDIA_A100"},
            {"cpus": 16.0, "gpus": 1.0, "accelerator_type": None},
        ],
    )
    assert hw is not None
    assert hw.gpu_memory_bytes == 0


def test_a_cpu_only_cluster_reports_no_gpu_memory(monkeypatch):
    hw = _profile(monkeypatch, [{"cpus": 32.0, "gpus": 0.0, "accelerator_type": None}])
    assert hw is not None
    assert (hw.gpu_count, hw.gpu_memory_bytes) == (0, 0)


def test_inference_packing_uses_the_cluster_device_not_the_driver():
    """A 20 GB model against the 12 GB T4 fallback reserves a whole GPU; against the A100 the
    cluster actually has, it packs — the difference between ~85% of every device wasted and
    not."""
    from batcher.kyber.gpu.policy import decide_gpu_map_params

    on_a100 = decide_gpu_map_params(20.0, 0.0, None, gpu_memory_gb=80.0)
    fallback = decide_gpu_map_params(20.0, 0.0, None)
    assert on_a100.num_gpus < 1.0
    assert on_a100.num_gpus < fallback.num_gpus
    # And the VRAM left after the model seeds a correspondingly larger batch.
    assert (on_a100.batch_size or 0) > (fallback.batch_size or 0)


def test_a_user_pinned_fraction_still_wins_over_the_cluster_figure():
    """The cluster number fills what the user left unset; it never overrides an explicit ask."""
    from batcher.kyber.gpu.policy import decide_gpu_map_params

    assert decide_gpu_map_params(20.0, 0.5, 128, gpu_memory_gb=80.0).num_gpus == 0.5


def test_actor_packing_uses_the_cluster_device_not_the_drivers(monkeypatch):
    """The OOM, not a slowdown: one GPU fraction is applied to every actor in the fleet, so a
    0.25 derived from the driver's 80 GB A100 packs four actors onto a 16 GB T4 worker. The
    binding (smallest) cluster device is the only figure valid on every node."""
    import batcher.api.executors as ex

    # The driver sees a big device; the cluster's binding device is small.
    import batcher.dist.executors.ray_runtime.accelerators as accel
    import batcher.ml.gpu as mlgpu

    calls: list[float] = []

    monkeypatch.setattr(mlgpu, "gpu_vram_gb", lambda: 80.0)
    monkeypatch.setattr(accel, "cluster_gpu_memory_gb", lambda: 16.0)
    monkeypatch.setattr(
        mlgpu, "recommend_gpu_fraction", lambda model_gb, vram: calls.append(vram) or 1.0
    )

    import batcher as bt

    plan = (
        bt.from_pydict({"x": [1]})
        .ml.map_batches(lambda b: b, num_gpus=1, model_memory_gb=10.0)
        ._plan
    )
    ex._map_scheduling_envelope(plan, 1, None)
    assert calls == [16.0]  # the T4 worker, not the A100 driver


def test_packing_falls_back_to_the_local_device_off_cluster(monkeypatch):
    """A single-node GPU run has no cluster topology to read, and must keep packing against
    the device it can actually see rather than skipping packing entirely."""
    import batcher.api.executors as ex
    import batcher.dist.executors.ray_runtime.accelerators as accel
    import batcher.ml.gpu as mlgpu

    calls: list[float] = []
    monkeypatch.setattr(accel, "cluster_gpu_memory_gb", lambda: None)
    monkeypatch.setattr(mlgpu, "gpu_vram_gb", lambda: 24.0)
    monkeypatch.setattr(
        mlgpu, "recommend_gpu_fraction", lambda model_gb, vram: calls.append(vram) or 1.0
    )

    import batcher as bt

    plan = (
        bt.from_pydict({"x": [1]})
        .ml.map_batches(lambda b: b, num_gpus=1, model_memory_gb=10.0)
        ._plan
    )
    ex._map_scheduling_envelope(plan, 1, None)
    assert calls == [24.0]


def test_routing_fits_on_one_real_gpu_where_the_t4_fallback_would_shard():
    """The end-to-end consequence: the same working set is a single dispatch on the cluster's
    real device and a sharded query against the driver's assumed one."""
    import batcher as bt
    from batcher.kyber.gpu.policy import decide_gpu_backend

    rows = 200_000  # ~3.2 MB of int64 pairs — small in absolute terms, but the point is the
    q = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})  # *ratio* to the budget
    # A group-by, so the plan HAS a mergeable reducer to shard on. Sharding is not a property
    # of the data alone: a plan with nothing to fold cannot be split across devices however
    # small the budget is, so a bare scan would test the budget against the wrong ladder rung.
    q = q.group_by("k").agg(s=bt.col("v").sum())
    plan, sources = q._plan, q._sources
    # The device the cluster really has vs a budget below the working set. Real VRAM figures
    # would need a multi-GB fixture to separate; the routing ladder is the same either way.
    big = decide_gpu_backend(plan, sources, gpu_count=8, force=True, gpu_memory_gb=80.0)
    small = decide_gpu_backend(plan, sources, gpu_count=8, force=True, gpu_memory_gb=0.001)
    assert big.use_gpu and not big.distributed  # fits one device
    assert small.distributed  # the same data must shard against a tiny device


def test_a_plan_with_nothing_to_divide_is_not_sharded_across_devices():
    """Sharding needs a plan that divides, not just data too big for one device.

    A chain reducing with `median` needs a group's whole value set, so no shard can produce a
    partial anyone can fold. Routing it to a fan-out would promise a scale-out that does not
    exist, and the single dispatch underneath would OOM and fall back anyway; the spillable CPU
    engine is the honest destination, and the reason says so.
    """
    import batcher as bt
    from batcher.kyber.gpu.policy import decide_gpu_backend

    rows = 200_000
    q = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    q = q.group_by("k").agg(m=bt.col("v").median())
    tiny = decide_gpu_backend(q._plan, q._sources, gpu_count=8, force=True, gpu_memory_gb=0.001)
    assert not tiny.use_gpu and not tiny.distributed
    assert "nothing to shard on" in tiny.reason
    # ...and with room on one device it is a perfectly ordinary single dispatch.
    roomy = decide_gpu_backend(q._plan, q._sources, gpu_count=8, force=True, gpu_memory_gb=80.0)
    assert roomy.use_gpu and not roomy.distributed


def test_a_row_local_chain_shards_because_its_slices_reassemble():
    """A filter is the most divisible shape there is, and used to be refused the fan-out.

    Each shard's output is its slice of the answer, in order, so concatenating the slices is
    the answer. Excluding it for having "nothing to fold" left the largest scans — the ones a
    filter is written for — bounded by one device's memory.
    """
    import batcher as bt
    from batcher.kyber.gpu.policy import decide_gpu_backend

    rows = 200_000
    q = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    q = q.filter(bt.col("v") > 10)
    tiny = decide_gpu_backend(q._plan, q._sources, gpu_count=8, force=True, gpu_memory_gb=0.001)
    assert tiny.use_gpu and tiny.distributed


def test_a_reducing_chain_is_sized_by_what_it_processes_not_what_it_returns():
    """`group_by().agg().sort().limit(10)` processes the scan, and returns ten rows.

    Estimating the OUTPUT put every such query below the small-input threshold and refused it
    the GPU on the grounds that ten rows do not amortize a kernel launch — with a scan of any
    size underneath it. The estimate has to descend past the whole run of reducing operators,
    not just the top one.
    """
    import batcher as bt
    from batcher.kyber.gpu.policy import _estimate

    rows = 200_000
    q = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    deep = q.group_by("k").agg(s=bt.col("v").sum()).sort("s").limit(10)
    assert _estimate(deep._plan, deep._sources, None)[0] == rows
