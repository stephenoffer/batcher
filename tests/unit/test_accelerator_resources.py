"""Non-GPU accelerators must be requestable, and GPU/CPU placement must not change.

Ray reports NVIDIA, AMD, Intel, and MetaX devices as the ``GPU`` resource, which
`num_gpus` covers. Every other accelerator is a *named custom resource* instead — ``TPU``,
``neuron_cores`` (Trainium/Inferentia), ``HPU`` (Gaudi), ``NPU`` — and so is any resource
an operator defines on their own on-prem cluster. Without a passthrough those are
unreachable: a TPU stage would request `num_gpus` on a node that has no GPU resource and
pend forever, which presents as a hung job rather than an error.

Two failure modes are covered here because both are silent: requesting the wrong resource,
and dropping `accelerator_type` (which used to be gated on `num_gpus`, so the device-model
pin vanished on exactly the accelerators that are not GPUs).
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.map import _gpu_options
from batcher.plan.resource import SchedulingEnvelope

pytestmark = pytest.mark.unit


def _opts(env: SchedulingEnvelope) -> dict:
    """`task_options` minus `runtime_env`, which is orthogonal package-shipping."""
    from batcher.dist.executors.ray_runtime.scheduling import task_options

    return {k: v for k, v in task_options(env).items() if k != "runtime_env"}


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        ((("TPU", 4.0),), {"TPU": 4.0}),
        ((("neuron_cores", 2.0),), {"neuron_cores": 2.0}),
        ((("HPU", 8.0),), {"HPU": 8.0}),
        # An operator's own on-prem resource — the reason this is a generic passthrough
        # rather than an enumeration of vendors.
        ((("fpga_slot", 1.0),), {"fpga_slot": 1.0}),
    ],
)
def test_custom_accelerator_resources_reach_ray(resources, expected):
    assert _opts(SchedulingEnvelope(resources=resources))["resources"] == expected


def test_accelerator_type_applies_without_gpus():
    """The regression: a TPU node has `num_gpus == 0`, so gating the device-model pin on
    GPUs silently dropped it and let the task land on any node in the cluster."""
    opts = _opts(SchedulingEnvelope(resources=(("TPU", 4.0),), accelerator_type="TPU-V6E"))
    assert opts["accelerator_type"] == "TPU-V6E"
    assert opts["resources"] == {"TPU": 4.0}
    assert "num_gpus" not in opts


def test_accelerator_type_is_dropped_when_nothing_is_requested():
    """A pin with no accelerator request would constrain placement for no reason."""
    assert "accelerator_type" not in _opts(SchedulingEnvelope(accelerator_type="TPU-V6E"))


def test_gpu_and_cpu_placement_are_unchanged():
    """The existing paths must be byte-identical — this is added reach, not a behavior change."""
    gpu = _opts(SchedulingEnvelope(num_gpus=1.0, accelerator_type="NVIDIA_A100"))
    assert gpu == {"num_cpus": 1.0, "num_gpus": 1.0, "accelerator_type": "NVIDIA_A100"}
    assert _opts(SchedulingEnvelope()) == {"num_cpus": 1.0}


def test_envelope_stays_hashable_with_resources():
    """`SchedulingEnvelope` is a frozen, hashable dataclass. Storing the resources as a
    dict would make `hash()` raise, which is why the field is a tuple."""
    assert hash(SchedulingEnvelope(resources=(("TPU", 4.0),))) is not None


@pytest.mark.parametrize(
    ("num_gpus", "accel", "resources", "expected"),
    [
        (0.0, "TPU-V6E", {"TPU": 4}, {"resources": {"TPU": 4}, "accelerator_type": "TPU-V6E"}),
        (0.0, None, {"neuron_cores": 2}, {"resources": {"neuron_cores": 2}}),
        (1.0, "NVIDIA_A100", None, {"num_gpus": 1.0, "accelerator_type": "NVIDIA_A100"}),
        (0.0, None, None, {}),
    ],
)
def test_map_stage_options(num_gpus, accel, resources, expected):
    assert _gpu_options(num_gpus, accel, resources) == expected


def test_public_api_threads_resources_to_the_plan():
    import batcher as bt

    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(lambda b: b, resources={"TPU": 4})
    assert ds._plan.resources == (("TPU", 4),)


def test_stacked_stages_take_the_max_per_resource():
    """Stacked maps fuse into one task, so their needs combine the way `num_gpus` does:
    a stage needing 4 chips and one needing 2 must get 4, or the larger stage is starved."""
    import batcher as bt
    from batcher.dist.executors.map import _map_resources

    ds = bt.from_pydict({"x": [1]})
    ds = ds.ml.map_batches(lambda b: b, resources={"TPU": 2})
    ds = ds.ml.map_batches(lambda b: b, resources={"TPU": 4})
    assert _map_resources(ds._plan)[4] == {"TPU": 4}


def test_autoscaler_asks_for_the_accelerator_the_query_needs(monkeypatch):
    """A `{"GPU": 1}` bundle asks for GPU nodes, which a TPU cluster has none of — so a TPU
    query would wait out the autoscale window and then run on whatever was already up."""
    # Patch the module that *defines* `_apply_autoscale_floor`. `scaling` only re-exports
    # `request_autoscale`, so patching there would never apply — the exact trap that made
    # this test start failing when the lifecycle moved into its own module.
    import batcher.dist.executors.ray_runtime.autoscale_request as autoscale

    calls: list[tuple] = []
    monkeypatch.setattr(
        autoscale, "_apply_autoscale_floor", lambda c, g=0, r=(): calls.append((c, g, r))
    )
    autoscale.request_autoscale(8, 0.0, (("TPU", 4.0),))
    try:
        assert calls[-1] == (8, 0, (("TPU", 4.0),))
        # Concurrent scopes compose by high-water mark, as the CPU and GPU floors do.
        autoscale.request_autoscale(4, 0.0, (("TPU", 8.0),))
        assert calls[-1][2] == (("TPU", 8.0),)
        autoscale.release_autoscale()
    finally:
        autoscale.release_autoscale()
    # The floor drops on the last release, so a finished query stops pinning the cluster.
    assert calls[-1] == (0, 0, ())


def test_a_tpu_host_is_not_reported_as_having_no_accelerator(monkeypatch):
    """`gpu_devices_absent` promises never to give a false negative. Knowing only the NVIDIA
    and AMD device nodes broke that promise on TPU/Trainium/Gaudi, and the false 'nothing
    here' then suppressed the real probe."""
    import batcher._internal.hardware as hw

    monkeypatch.setattr(
        hw.glob, "glob", lambda pattern, **kw: ["/dev/accel0"] if "accel" in pattern else []
    )
    hw.gpu_devices_absent.cache_clear()
    try:
        assert hw.gpu_devices_absent() is False
        # And diagnostics name the device instead of rendering an empty list.
        assert any("TPU" in str(d["name"]) for d in hw.gpu_inventory())
    finally:
        hw.gpu_devices_absent.cache_clear()


@pytest.mark.parametrize(
    ("message", "is_oom"),
    [
        ("CUDA out of memory", True),
        # XLA reports exhaustion with its own wording, so matching only the CUDA phrasing
        # left the halving retry disabled on a TPU — the batch just failed.
        ("RESOURCE_EXHAUSTED: XLA ran out of memory", True),
        ("some unrelated failure", False),
    ],
)
def test_out_of_memory_is_recognized_across_accelerators(message, is_oom):
    from batcher.ml.inference import _is_cuda_oom

    assert _is_cuda_oom(RuntimeError(message)) is is_oom
