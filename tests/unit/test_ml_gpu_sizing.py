"""GPU sizing and VRAM budgeting for packed, multi-actor batch inference (no GPU needed).

Every test here pins a bug that shipped green: NVML sampling that made packed actors
oscillate together, a packing floor that capped density at 4 actors/GPU, a VRAM budget
blind to the batch, pool sizing blind to tensor parallelism, a CUDA-only `torch.compile`
gate, an autocast probe that re-ran a UDF eight times, and unbounded pool submission.

NVML and torch are faked with monkeypatch, and the assertions are on the pure sizing math.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

from batcher.ml import gpu


@pytest.fixture
def clear_nvml_caches():
    """Drop the process-wide NVML session/handle caches around a fake-NVML test."""
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()
    yield
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()


def _fake_nvml(*, total: int, procs: list | None, used: int = 0) -> types.ModuleType:
    """A `pynvml` stand-in with one device; `procs=None` omits per-process accounting."""
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlDeviceGetCount = lambda: 1
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=used, total=total)
    if procs is not None:
        mod.nvmlDeviceGetComputeRunningProcesses = lambda h: procs
    return mod


# --- 1. per-process VRAM attribution -----------------------------------------------------


def test_packed_actors_do_not_observe_their_peers_vram(monkeypatch, clear_nvml_caches):
    """The synchronized-oscillation bug: NVML `info.used` is every process on the device.

    Four packed inference actors each holding 20% of an 80 GB card made *every* actor read
    0.80 and shrink at once, even though no single actor had grown. Each must instead see
    its own use against its own share of the device: 20% of 80 GB against a 20 GB budget is
    a full budget, and an actor holding a tenth of that is nowhere near its cap.
    """
    import os

    me = os.getpid()
    total = 80 << 30
    peers = [
        types.SimpleNamespace(pid=me, usedGpuMemory=2 << 30),  # this actor: 2 GB
        types.SimpleNamespace(pid=me + 1, usedGpuMemory=16 << 30),
        types.SimpleNamespace(pid=me + 2, usedGpuMemory=16 << 30),
        types.SimpleNamespace(pid=me + 3, usedGpuMemory=16 << 30),
    ]
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(total=total, procs=peers, used=50 << 30))
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()

    frac = gpu.sample_gpu_vram_fraction()
    # Own budget is 80/4 = 20 GB; this actor holds 2 GB of it.
    assert frac == pytest.approx(2 / 20)
    # The device-wide reading (50/80 = 0.625) is what caused every peer to shrink together.
    assert frac < 0.625


def test_sole_tenant_vram_fraction_is_unchanged(monkeypatch, clear_nvml_caches):
    """One process on the device must still report exactly used/total (no behavior change)."""
    import os

    total = 10 << 30
    procs = [types.SimpleNamespace(pid=os.getpid(), usedGpuMemory=3 << 30)]
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(total=total, procs=procs, used=3 << 30))
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()
    assert gpu.sample_gpu_vram_fraction() == pytest.approx(0.30)


def test_vram_falls_back_to_device_wide_without_per_process_accounting(
    monkeypatch, clear_nvml_caches
):
    """An old driver with no `nvmlDeviceGetComputeRunningProcesses` keeps the old reading.

    Reporting 0.0 there would disable the autobatcher's OOM guard entirely, which is worse
    than the aggregate it replaces."""
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(total=8, procs=None, used=2))
    gpu._nvml.cache_clear()
    gpu._nvml_handles.cache_clear()
    assert gpu.sample_gpu_vram_fraction() == pytest.approx(0.25)


def test_own_budget_fraction_is_pure_and_degenerates():
    assert gpu._own_budget_fraction(2.0, 8.0, 1) == pytest.approx(0.25)  # sole tenant
    assert gpu._own_budget_fraction(2.0, 8.0, 4) == pytest.approx(1.0)  # 2 GB of a 2 GB share
    assert gpu._own_budget_fraction(0.0, 8.0, 4) == 0.0  # nothing allocated yet
    assert gpu._own_budget_fraction(1.0, 0.0, 1) is None  # unknown device size


# --- 2. packing density is no longer capped at 4 actors/GPU ------------------------------


def test_small_model_packs_past_four_actors_per_gpu():
    """A 0.1 GB embedding model VRAM-fits ~30 actors on an 80 GB card; the 0.25 fraction
    floor handed back 4 and stranded ~87% of the device."""
    fits = gpu.max_actors_per_gpu(0.1, 80.0, context_overhead_gb=0.4)
    assert fits > 4
    frac = gpu.recommend_gpu_fraction(0.1, 80.0, context_overhead_gb=0.4)
    assert frac < 0.25  # the old floor
    assert int(1.0 / frac) > 4  # Ray packs floor(1/fraction) actors


def test_packing_fraction_never_goes_below_the_schedulability_floor():
    """The floor still exists — it just guards schedulability, not a 4-actor cap."""
    frac = gpu.recommend_gpu_fraction(0.0001, 80.0, context_overhead_gb=0.0)
    assert frac >= gpu._MIN_FRACTION == 0.05


def test_single_actor_model_still_takes_a_whole_gpu():
    assert gpu.recommend_gpu_fraction(40.0, 48.0) == 1.0


def test_utilization_packing_keeps_the_coarse_floor():
    """`recommend_num_gpus` packs from utilization, which cannot prove the weights fit in
    the slice, so it must keep the conservative 0.25 floor."""
    assert gpu.recommend_num_gpus(0.01, 1.0) == 0.25


# --- 3. the VRAM budget depends on batch size, sequence length, and dtype -----------------


def test_vram_multiplier_is_the_old_flat_value_at_the_reference_workload():
    assert gpu.inference_vram_multiplier() == pytest.approx(1.5)
    assert gpu.inference_vram_multiplier(
        batch_rows=32, seq_len=512, activation_dtype_bytes=2
    ) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_rows": 256},  # 8x the reference batch
        {"seq_len": 4096},  # 8x the reference context
        {"activation_dtype_bytes": 4},  # fp32 activations
    ],
    ids=["batch_rows", "seq_len", "dtype"],
)
def test_each_activation_driver_raises_the_vram_budget(kwargs):
    """A flat 1.5x saw none of these, so a long-context fp32 job packed as densely as a
    short fp16 one and OOMed."""
    assert gpu.inference_vram_multiplier(**kwargs) > 1.5


def test_bigger_batch_packs_fewer_actors_per_gpu():
    small = gpu.max_actors_per_gpu(2.0, 80.0, batch_rows=8, seq_len=128)
    large = gpu.max_actors_per_gpu(2.0, 80.0, batch_rows=512, seq_len=4096)
    assert large < small


def test_vram_multiplier_is_clamped_both_ways():
    assert gpu.inference_vram_multiplier(batch_rows=1, seq_len=1) == pytest.approx(1.1)
    assert gpu.inference_vram_multiplier(batch_rows=10**6, seq_len=10**6) == pytest.approx(4.0)


def test_explicit_multiplier_still_overrides_the_workload_scaling():
    assert gpu.max_actors_per_gpu(
        1.0, 24.0, context_overhead_gb=0.4, inference_multiplier=1.5, batch_rows=4096
    ) == gpu.max_actors_per_gpu(1.0, 24.0, context_overhead_gb=0.4, inference_multiplier=1.5)


# --- 4. tensor-parallel awareness in pool sizing ------------------------------------------


def _fake_ray(gpus: float) -> types.ModuleType:
    mod = types.ModuleType("ray")
    mod.cluster_resources = lambda: {"GPU": gpus}
    return mod


def test_pool_sizing_divides_by_tensor_parallel_size(monkeypatch):
    """A vLLM engine with `tensor_parallel_size=4` consumes 4 GPUs per replica. Sizing on
    `num_gpus` alone asked for 8 replicas on 8 GPUs, oversubscribing the cluster 4x so the
    pool never fully scheduled."""
    monkeypatch.setitem(sys.modules, "ray", _fake_ray(8.0))
    assert gpu.gpu_aware_pool_default(1.0, 2, 100, tensor_parallel_size=4) == 2
    assert gpu.gpu_aware_pool_default(1.0, 2, 100) == 8  # tp=1 default is unchanged


def test_tensor_parallel_pool_sizing_never_returns_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", _fake_ray(2.0))
    assert gpu.gpu_aware_pool_default(1.0, 3, 100, tensor_parallel_size=8) == 1


# --- 5/6. transformers placement + a vendor-agnostic compile gate -------------------------


def test_multi_gpu_process_shards_with_device_map(monkeypatch):
    """`device=0` pins the model to one card, so a model larger than one GPU OOMs at load
    on a node that had room across its devices."""
    from batcher.ml import inference

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 4)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    import importlib.util

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "accelerate" else None
    )
    assert inference._should_shard_across_devices("cuda") is True


def test_single_gpu_process_keeps_the_explicit_device_pin(monkeypatch):
    from batcher.ml import inference

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert inference._should_shard_across_devices("cuda") is False
    assert inference._should_shard_across_devices("xpu") is False


def test_compile_gate_is_not_cuda_only(monkeypatch):
    """`torch.cuda.is_available()` denied `torch.compile` to Intel XPU and Apple MPS,
    although every other accelerator decision in `ml/gpu.py` is vendor-agnostic."""
    from batcher.ml import inference

    compiled: list[object] = []

    class _Conv2d:
        pass

    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.SimpleNamespace(Conv2d=_Conv2d)
    fake_torch.channels_last = "channels_last"
    fake_torch.compile = lambda m: compiled.append(m) or "compiled"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # not NVIDIA
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gpu, "detect_backend", lambda: "xpu")

    model = _Conv2d()
    model.modules = lambda: [model]
    model.to = lambda memory_format: model
    pipe = types.SimpleNamespace(model=model)
    inference._maybe_compile_pipeline(pipe)
    assert pipe.model == "compiled"


def test_compile_still_skipped_on_cpu(monkeypatch):
    from batcher.ml import inference

    monkeypatch.setattr(gpu, "detect_backend", lambda: "cpu")
    pipe = types.SimpleNamespace(model=object())
    inference._maybe_compile_pipeline(pipe)
    assert not isinstance(pipe.model, str)  # untouched


# --- 7. bounded in-flight submission ------------------------------------------------------


def test_pool_submission_is_bounded(monkeypatch):
    """Unbounded submission put the ENTIRE input in flight before the first consumer pull
    returned, so peak memory scaled with the dataset instead of the pool."""
    from batcher.ml.inference import InferencePool

    pulled = {"n": 0}
    n_batches = 64

    def source():
        for i in range(n_batches):
            pulled["n"] += 1
            yield pa.record_batch({"x": pa.array([i])})

    def factory():
        def worker(batch):
            return batch

        return worker

    pool = InferencePool(factory, num_workers=2, target_batch_rows=1, max_inflight=4)
    it = pool.run(source())
    next(it)  # one output
    # 4 in flight + the yielded one + the rebatcher's buffered head: comfortably bounded.
    assert pulled["n"] <= 8, f"submitted {pulled['n']} of {n_batches} before the first pull"


def test_bounded_pool_still_conserves_every_row():
    """Backpressure must not lose or reorder a row (the guard the flush fix also pins)."""
    from batcher.ml.inference import InferencePool

    def factory():
        return lambda batch: batch

    batches = [pa.record_batch({"x": pa.array([i, i + 1])}) for i in range(0, 100, 2)]
    pool = InferencePool(factory, num_workers=3, target_batch_rows=7, max_inflight=2)
    out = list(pool.run(iter(batches)))
    got = [v for b in out for v in b.column("x").to_pylist()]
    assert got == [v for b in batches for v in b.column("x").to_pylist()]


def test_default_inflight_bound_scales_with_the_pool():
    from batcher.ml.inference import InferencePool

    def factory():
        return lambda batch: batch

    assert InferencePool(factory, num_workers=5)._max_inflight == 20


# --- 8. the autocast probe no longer re-runs a side-effecting UDF ------------------------


def _fake_torch_with_allocator(peak_values: list[int]) -> types.ModuleType:
    """A torch stand-in whose `max_memory_allocated` walks `peak_values`."""
    import contextlib

    mod = types.ModuleType("torch")
    seq = iter(peak_values)
    mod.autocast = lambda **_: contextlib.nullcontext()
    mod.cuda = types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: next(seq, peak_values[-1]),
        synchronize=lambda: None,
    )
    return mod


def test_probe_stops_after_one_run_when_the_call_allocates_nothing(monkeypatch):
    """A UDF that calls a hosted LLM allocates no local VRAM. The probe used to run it
    eight times, which is eight billed requests instead of one."""
    calls = {"n": 0}

    def hosted_llm(batch):
        calls["n"] += 1
        return batch

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_with_allocator([0, 0]))
    batch = pa.record_batch({"x": pa.array(list(range(8)))})
    assert gpu._autocast_speeds_up(hosted_llm, batch, "cuda", "float16") is False
    assert calls["n"] == 1, f"ran the UDF {calls['n']} times"


def test_probe_still_times_a_real_gpu_model(monkeypatch):
    """A call that DOES allocate accelerator memory keeps the full timing sweep."""
    calls = {"n": 0}

    def model(batch):
        calls["n"] += 1
        return batch

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_with_allocator([0, 1 << 20]))
    batch = pa.record_batch({"x": pa.array(list(range(8)))})
    gpu._autocast_speeds_up(model, batch, "cuda", "float16")
    assert calls["n"] > 1


def test_udf_can_decline_the_probe_entirely(monkeypatch):
    """`batcher_autocast = False` is the escape hatch for a UDF with side effects the
    engine cannot see (a paid API call, a write)."""
    calls = {"n": 0}

    def paid(batch):
        calls["n"] += 1
        return batch

    paid.batcher_autocast = False
    monkeypatch.setattr(gpu, "_autocast_device_dtype", lambda: ("cuda", "float16"))
    wrapped = gpu.autocast_call(paid)
    assert wrapped is paid  # untouched: no wrapper, so no probe can ever run
    wrapped(pa.record_batch({"x": pa.array([1])}))
    assert calls["n"] == 1


def test_opt_out_honored_on_a_class_udf(monkeypatch):
    class Model:
        batcher_autocast = False

        def __call__(self, batch):
            return batch

    monkeypatch.setattr(gpu, "_autocast_device_dtype", lambda: ("cuda", "float16"))
    model = Model()
    assert gpu.autocast_call(model) is model


def test_probe_allowed_by_default():
    def plain(batch):
        return batch

    assert gpu._autocast_probe_allowed(plain) is True
