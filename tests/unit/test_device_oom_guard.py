"""Device-memory scope, allocator configuration, and the GPU out-of-memory ladder.

The theme is that "the GPU" is not a well-defined thing on a node with eight of them, and
every place the engine assumed device 0 was a place it could size against a board it does not
own. These run without a GPU: the driver's telemetry is the boundary and is faked here.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.devices import OomKind, classify_oom, is_device_oom
from batcher._internal.hardware.devices import scope as scope_mod
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel.device import plan_torch_allocator
from batcher.config import AcceleratorConfig, DeviceMemoryConfig
from batcher.ml.autobatch import ThroughputController

_GIB = 1 << 30


def _telemetry(*specs: tuple[int, str, int, int]) -> tuple[DeviceTelemetry, ...]:
    """Build fake driver telemetry from `(index, uuid, total_bytes, used_bytes)` tuples."""
    return tuple(
        DeviceTelemetry(index=i, uuid=uuid, memory_total_bytes=total, memory_used_bytes=used)
        for i, uuid, total, used in specs
    )


@pytest.fixture
def eight_gpus(monkeypatch):
    """A node with eight identical 80 GiB devices, the fourth of which is nearly full."""
    devices = _telemetry(
        *((i, f"GPU-{i:08d}", 80 * _GIB, 70 * _GIB if i == 3 else 2 * _GIB) for i in range(8))
    )
    monkeypatch.setattr(scope_mod, "device_telemetry", lambda: devices)
    for env in scope_mod.VISIBLE_DEVICE_ENVS:
        monkeypatch.delenv(env, raising=False)
    return devices


class TestVisibleDevices:
    """Which boards this process may touch. Getting it wrong sizes against a stranger's GPU."""

    def test_ordinals_pin_to_those_devices(self, eight_gpus, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
        assert scope_mod.visible_device_indices() == (2, 5)

    def test_uuids_resolve_instead_of_falling_back_to_the_whole_node(self, eight_gpus, monkeypatch):
        """The Kubernetes device plugin pins by UUID. An ordinal-only parse treated that as
        unparseable and handed the actor every device on the node — so on the fleets that pin
        hardest, every pinned actor measured all eight boards."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000005,GPU-00000001")
        assert scope_mod.visible_device_indices() == (5, 1)

    def test_a_mig_handle_resolves_to_its_parent_board(self, eight_gpus, monkeypatch):
        """A partition draws memory from the board it sits on, so that board is what bounds it."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-00000006/1/0")
        assert scope_mod.visible_device_indices() == (6,)

    def test_an_unset_variable_means_the_whole_node(self, eight_gpus):
        """An unpinned driver or monitor legitimately sees everything."""
        assert scope_mod.visible_device_indices() == tuple(range(8))

    def test_an_empty_variable_means_no_devices(self, eight_gpus, monkeypatch):
        """`CUDA_VISIBLE_DEVICES=""` is how a scheduler says "none", which is not the same as
        unset — treating it as unset hands a CPU-only actor the whole node."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert scope_mod.device_scope().indices == ()

    def test_an_unresolvable_value_does_not_claim_the_node_by_accident(
        self, eight_gpus, monkeypatch
    ):
        """A stale UUID means "I cannot tell which is mine", and the honest fallback is the
        conservative one every caller already handles."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")
        assert scope_mod.visible_device_indices() == tuple(range(8))


class TestDeviceMemory:
    def test_free_accounts_for_what_a_co_tenant_holds(self, eight_gpus):
        """The figure this process's own allocator statistics can never produce."""
        assert scope_mod.device_scope().free_of(3) == 10 * _GIB
        assert scope_mod.device_scope().free_of(0) == 78 * _GIB

    def test_the_emptiest_device_breaks_ties_on_the_lowest_index(self, eight_gpus):
        """Deterministic placement: a repeated run must not drift with dictionary order."""
        assert scope_mod.device_scope().emptiest == 0

    def test_a_mixed_node_sizes_against_its_smallest_board(self, monkeypatch):
        """One actor count is computed for the whole stage, so packing to the larger card
        produces a replica count that fits on some devices and OOMs on the rest."""
        monkeypatch.setattr(
            scope_mod,
            "device_telemetry",
            lambda: _telemetry((0, "GPU-a", 80 * _GIB, 0), (1, "GPU-b", 24 * _GIB, 0)),
        )
        for env in scope_mod.VISIBLE_DEVICE_ENVS:
            monkeypatch.delenv(env, raising=False)
        assert scope_mod.min_visible_capacity_bytes() == 24 * _GIB
        assert scope_mod.device_scope().heterogeneous is True

    def test_a_uniform_node_is_not_reported_as_heterogeneous(self, eight_gpus):
        assert scope_mod.device_scope().heterogeneous is False

    def test_no_accelerator_degrades_to_empty_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(scope_mod, "device_telemetry", tuple)
        assert scope_mod.device_scope().count == 0
        assert scope_mod.min_visible_capacity_bytes() is None


class TestOomRecognition:
    """Every vendor spells exhaustion differently, and a phrasing we miss disables the retry."""

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            RuntimeError("RESOURCE_EXHAUSTED: Attempting to allocate 1GB"),
            RuntimeError("HIP_ERROR_OUT_OF_MEMORY"),
        ],
    )
    def test_recognized(self, exc):
        assert is_device_oom(exc) is True

    def test_a_typed_torch_error_is_recognized_without_importing_torch(self):
        oom = type("OutOfMemoryError", (Exception,), {})
        assert is_device_oom(oom("device oom")) is True

    def test_an_ordinary_error_is_not_swallowed_as_an_oom(self):
        """Misclassifying a model bug as an OOM would retry it sixteen times and then report
        the wrong cause."""
        assert is_device_oom(ValueError("unknown column")) is False
        assert is_device_oom(RuntimeError("shape mismatch")) is False


class TestOomClassification:
    """The three device OOMs need opposite responses and arrive with the same message."""

    def _measured(self, monkeypatch, *, fragmentation, own_share):
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.oom.fragmentation_ratio", lambda: fragmentation
        )
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.oom._own_share_of_device", lambda _d: own_share
        )

    def test_a_full_device_this_process_filled_is_shrunk(self, monkeypatch):
        self._measured(monkeypatch, fragmentation=0.02, own_share=0.9)
        verdict = classify_oom(RuntimeError("CUDA out of memory"))
        assert verdict.kind is OomKind.TOO_LARGE
        assert verdict.should_shrink is True
        assert verdict.should_retry_same_size is False

    def test_a_fragmented_allocator_is_retried_at_the_same_size_first(self, monkeypatch):
        """The memory was there in pieces too small to serve the request. Halving throws away
        half the throughput to work around something releasing the cache just fixed."""
        self._measured(monkeypatch, fragmentation=0.6, own_share=0.9)
        verdict = classify_oom(RuntimeError("CUDA out of memory"))
        assert verdict.kind is OomKind.FRAGMENTED
        assert verdict.should_retry_same_size is True

    def test_a_device_a_co_tenant_filled_is_not_shrunk_at_all(self, monkeypatch):
        """Halving this process's batch to one row cannot recover a neighbour's memory, and
        retrying turns one placement mistake into minutes of wasted GPU time."""
        self._measured(monkeypatch, fragmentation=0.6, own_share=0.05)
        verdict = classify_oom(RuntimeError("CUDA out of memory"))
        assert verdict.kind is OomKind.OCCUPIED
        assert verdict.should_shrink is False

    def test_with_nothing_measurable_it_falls_back_to_the_always_safe_answer(self, monkeypatch):
        """No torch, no driver attribution: shrinking is the response that cannot make it worse."""
        self._measured(monkeypatch, fragmentation=None, own_share=None)
        assert classify_oom(RuntimeError("CUDA out of memory")).kind is OomKind.TOO_LARGE

    def test_a_non_oom_error_still_gets_a_safe_verdict(self):
        assert classify_oom(ValueError("nope")).kind is OomKind.TOO_LARGE


class TestTorchAllocatorPlan:
    def test_the_default_asks_for_expandable_segments(self):
        """The single largest fragmentation fix PyTorch ships, and off unless someone knows to
        set an environment variable before the process starts."""
        plan = plan_torch_allocator(AcceleratorConfig())
        assert plan.expandable_segments is True
        assert plan.alloc_conf() == "expandable_segments:True"

    def test_packed_actors_each_get_a_share_rather_than_the_whole_device(self):
        """Four actors each believing they may address the whole board is no cap at all."""
        assert plan_torch_allocator(AcceleratorConfig(), tenants=4).memory_fraction == 0.21

    def test_a_sole_tenant_keeps_everything_the_vram_headroom_leaves(self):
        assert plan_torch_allocator(AcceleratorConfig()).memory_fraction == 0.85

    def test_the_gc_threshold_reaches_the_settings_string(self):
        cfg = AcceleratorConfig(memory=DeviceMemoryConfig(torch_gc_threshold=0.8))
        assert "garbage_collection_threshold:0.8" in plan_torch_allocator(cfg).alloc_conf()

    def test_turning_both_switches_off_is_inert(self):
        """A deployment that opts out must get exactly the process it had before."""
        cfg = AcceleratorConfig(
            memory=DeviceMemoryConfig(torch_expandable_segments=False, torch_memory_fraction=False)
        )
        assert plan_torch_allocator(cfg).is_inert is True


class TestBatchSizeLearnsFromOom:
    """Without this the retry recovers the rows and the climb walks straight back into the
    same failure, so a job can spend most of its life failing and retrying while every
    throughput measurement says it is improving."""

    def test_an_oom_caps_the_climb_below_the_size_that_failed(self):
        ctl = ThroughputController(initial=1000, min_rows=1, max_rows=65_536)
        after = ctl.note_oom(rows=1000)
        assert after < 1000
        # Even a long run of excellent throughput cannot climb back past the ceiling.
        for _ in range(20):
            ctl.update(1e9)
        assert ctl.current() <= after

    def test_consecutive_failures_ratchet_the_ceiling_further_down(self):
        """One OOM proves the size is too big; a third in a row means the estimate of *how*
        too big is itself wrong."""
        ctl = ThroughputController(initial=1000, min_rows=1, max_rows=65_536)
        first = ctl.note_oom(rows=1000)
        second = ctl.note_oom(rows=first)
        assert second < first

    def test_the_persisted_plateau_cannot_escape_through_best_size(self):
        """`best_size` is what the learned store writes for the next run to warm-start from,
        so a plateau measured before the failure must not be handed back."""
        ctl = ThroughputController(initial=1000, min_rows=1, max_rows=65_536)
        ctl.update(1000.0)  # establishes a plateau at 1000
        ctl.note_oom(rows=1000)
        assert ctl.best_size() < 1000

    def test_a_success_clears_the_streak_but_not_the_ceiling(self):
        """One batch fitting is not evidence that a size which already failed became safe."""
        ctl = ThroughputController(initial=1000, min_rows=1, max_rows=65_536)
        ceiling = ctl.note_oom(rows=1000)
        ctl.update(500.0)
        assert ctl.current() <= ceiling

    def test_the_floor_is_still_respected(self):
        """A pathological run of failures must not drive the batch below the caller's bound."""
        ctl = ThroughputController(initial=8, min_rows=4, max_rows=64)
        for _ in range(10):
            ctl.note_oom()
        assert ctl.current() == 4


class TestPackingRespectsCoTenants:
    """Packing against capacity counts memory a neighbour is already holding — the most
    common way a fractional-GPU stage that "obviously fits" OOMs the moment it lands."""

    def test_a_half_full_device_fits_half_as_many_actors(self, monkeypatch):
        from batcher.ml import gpu

        monkeypatch.setattr(gpu, "vram_context_overhead", lambda: 0.0)
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.device_free_bytes", lambda: 40 * _GIB
        )
        packed = gpu.max_actors_per_gpu(4.0, 80.0, inference_multiplier=1.0, headroom=0.0)
        assert packed == 10

    def test_opting_out_sizes_against_capacity(self, monkeypatch):
        from batcher.ml import gpu

        monkeypatch.setattr(gpu, "vram_context_overhead", lambda: 0.0)
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.device_free_bytes", lambda: 40 * _GIB
        )
        packed = gpu.max_actors_per_gpu(
            4.0, 80.0, inference_multiplier=1.0, headroom=0.0, respect_co_tenants=False
        )
        assert packed == 20

    def test_an_unreadable_driver_leaves_the_declared_capacity_alone(self, monkeypatch):
        """No measurement must never *widen* or narrow the budget on a guess."""
        from batcher.ml import gpu

        monkeypatch.setattr(gpu, "vram_context_overhead", lambda: 0.0)
        monkeypatch.setattr("batcher._internal.hardware.devices.device_free_bytes", lambda: None)
        assert gpu.max_actors_per_gpu(4.0, 80.0, inference_multiplier=1.0, headroom=0.0) == 20

    def test_a_driver_reading_never_widens_a_deliberately_reduced_budget(self, monkeypatch):
        """A caller passing a MIG slice's size must not have it replaced by the whole board."""
        from batcher.ml import gpu

        monkeypatch.setattr(gpu, "vram_context_overhead", lambda: 0.0)
        monkeypatch.setattr(
            "batcher._internal.hardware.devices.device_free_bytes", lambda: 80 * _GIB
        )
        assert gpu.max_actors_per_gpu(4.0, 20.0, inference_multiplier=1.0, headroom=0.0) == 5


# --- The device tier does not raise a RuntimeError ----------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # RMM, the allocator every cuDF kernel runs on, raises this.
        "std::bad_alloc: out_of_memory: CUDA error at: rmm/mr/device/cuda_memory_resource.hpp",
        "RMM failure at: pool_memory_resource.hpp: Maximum pool size exceeded",
        "HIP error: hipErrorOutOfMemory",
    ],
)
def test_a_device_allocators_memory_error_is_a_device_oom(message):
    # `MemoryError` is not a `RuntimeError`, so the whole GPU relational path missed the
    # halving retry and failed a query a smaller batch would have completed.
    exc = MemoryError(message)
    assert is_device_oom(exc) is True
    assert classify_oom(exc).should_shrink is True


@pytest.mark.parametrize(
    "message",
    [
        "Unable to allocate 1.2 GiB for an array with shape (160000000,) and data type int64",
        "",
    ],
)
def test_a_host_memory_error_is_not_a_device_oom(message):
    # The two want opposite responses: halving a batch relieves the device and does nothing
    # for the host, so treating a host exhaustion as a device one retries into the same wall.
    assert is_device_oom(MemoryError(message)) is False


@pytest.mark.parametrize(
    "message",
    [
        # `npu` is inside `input`, and `hip` is inside `relationship`. A short vendor prefix
        # is exactly the marker that looks harmless and then fires on an ordinary word — and
        # the two failures want opposite responses, so a false positive retries into the wall.
        "Unable to allocate output buffer for input column 3",
        "cannot build the relationship index: out of space",
        "input too large",
    ],
)
def test_a_vendor_prefix_does_not_fire_on_an_ordinary_word(message):
    assert is_device_oom(MemoryError(message)) is False


@pytest.mark.parametrize(
    "message",
    [
        "NPU out of memory. Tried to allocate 2.00 GiB",
        "synapse allocator: failed to allocate on device 0",
        "HIP error: hipErrorOutOfMemory",
    ],
)
def test_the_other_vendors_still_report_their_own_exhaustion(message):
    assert is_device_oom(MemoryError(message)) is True
