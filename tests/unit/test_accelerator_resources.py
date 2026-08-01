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
        (
            0.0,
            "TPU-V6E",
            {"TPU": 4},
            {"resources": {"TPU": 4}, "accelerator_type": "TPU-V6E", "num_cpus": 0},
        ),
        (0.0, None, {"neuron_cores": 2}, {"resources": {"neuron_cores": 2}, "num_cpus": 0}),
        (
            1.0,
            "NVIDIA_A100",
            None,
            {"num_gpus": 1.0, "accelerator_type": "NVIDIA_A100", "num_cpus": 0},
        ),
        (0.0, None, None, {}),
    ],
)
def test_map_stage_options(num_gpus, accel, resources, expected):
    assert _gpu_options(num_gpus, accel, resources) == expected


def test_an_accelerator_stage_reserves_no_cpu():
    """The deadlock this prevents, and why the CPU grant has to be spelled explicitly.

    Ray's rule is that an actor naming *any* resource takes a core for its whole lifetime
    (`DEFAULT_ACTOR_CREATION_CPU_SPECIFIED = 1`), and naming `num_gpus` is naming one. The
    shuffle fleet takes its workers in a placement group sized to the cluster's whole CPU
    capacity, so on any pipeline that shuffles before it infers — a `group_by`/`join`/`sort`
    feeding `ds.ml.map_batches`, which is the heterogeneous CPU+GPU shape the API documents —
    that core never comes free. The pool never places, every device sits idle, and
    `ray status` reports a fully reserved cluster, so it reads as busy rather than stuck.

    The GPU *relational* path already fixed this in `gpu_task_options`; this is the
    *inference* path, which is the one an ML user reaches for.
    """
    assert _gpu_options(1.0, None, None)["num_cpus"] == 0
    assert _gpu_options(0.25, None, None)["num_cpus"] == 0
    # Every other accelerator is a custom resource rather than `num_gpus`, and queues behind
    # exactly the same core.
    assert _gpu_options(0.0, None, {"TPU": 4})["num_cpus"] == 0
    assert _gpu_options(0.0, None, {"neuron_cores": 2})["num_cpus"] == 0
    # A CPU-only stage is untouched: it has no device to bound its concurrency, so the core
    # *is* the resource it contends for and it must keep asking for one.
    assert "num_cpus" not in _gpu_options(0.0, None, None)


def test_an_accelerator_task_keeps_its_zero_over_the_skew_adaptive_share():
    """The stateless-task branch spells `num_cpus` itself, from the skew-adaptive share.

    A custom-resource stage (TPU/Trainium) takes that branch — it wants no actor pool — so
    ordering decides whether it escapes the reservation. Spelling the share last overrode
    the accelerator zero and put it straight back behind the core.
    """
    opts = _gpu_options(0.0, None, {"TPU": 4})
    merged = {"num_cpus": 3.0, **opts}
    assert merged["num_cpus"] == 0, "the accelerator grant must win over the CPU share"
    # ...and a CPU-only stage still gets its adaptive share, since `opts` names no CPU.
    assert {"num_cpus": 3.0, **_gpu_options(0.0, None, None)}["num_cpus"] == 3.0


def test_custom_resources_are_reserved_in_the_placement_bundle():
    """The regression: a bundle reserves by resource, so a `TPU`/`neuron_cores` request that
    lived only in `.options()` reserved nothing. The gang was then placed on nodes that
    satisfied CPU alone, and every task afterwards demanded an accelerator its own bundle
    never held — pending forever on a CPU node rather than failing."""
    from batcher.dist.executors.ray_runtime.scheduling import _bundle

    bundle = _bundle(SchedulingEnvelope(num_cpus=2.0, resources=(("TPU", 4.0),)), node_class={})
    assert bundle == {"CPU": 2.0, "TPU": 4.0}


def test_gpu_and_cpu_bundles_are_unchanged():
    """Added reach, not a behavior change: the GPU and CPU bundles must be byte-identical."""
    from batcher.dist.executors.ray_runtime.scheduling import _bundle

    assert _bundle(SchedulingEnvelope(num_gpus=1.0), node_class={}) == {"CPU": 1.0, "GPU": 1.0}
    assert _bundle(SchedulingEnvelope(), node_class={}) == {"CPU": 1.0}


def test_a_resources_only_stage_forces_distribution():
    """The regression: `distributed="auto"` looked only at `num_gpus`, so a TPU/Trainium
    stage — which carries `num_gpus == 0` plus a custom resource — was invisible to the
    predicate and got routed by input size alone, running on the CPU-only driver and never
    reaching the accelerator nodes it asked for."""
    import batcher as bt
    from batcher.api.terminal.routing import _plan_has_gpu_stage

    tpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b, resources={"TPU": 4})
    assert _plan_has_gpu_stage(tpu._plan) is True
    # The GPU and plain-CPU verdicts are unchanged.
    gpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b, num_gpus=1)
    assert _plan_has_gpu_stage(gpu._plan) is True
    cpu = bt.from_pydict({"x": [1]}).ml.map_batches(lambda b: b)
    assert _plan_has_gpu_stage(cpu._plan) is False


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
    from batcher._internal import accelerators

    # Patched on `accelerators`, where the device-node scan actually lives; `hardware`
    # re-exports the probe but does not own it.
    monkeypatch.setattr(
        accelerators.glob,
        "glob",
        lambda pattern, **kw: ["/dev/accel0"] if "accel" in pattern else [],
    )
    # Both probes memoize their answer for the process (the device set cannot change under
    # a running process), so a simulated host has to invalidate them the same way.
    hw.reset_hardware_probes()
    try:
        assert hw.gpu_devices_absent() is False
        # And diagnostics name the device instead of rendering an empty list.
        assert any("TPU" in str(d["name"]) for d in hw.gpu_inventory())
    finally:
        hw.reset_hardware_probes()


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


def test_a_non_gpu_accelerator_stage_is_costed_as_inference():
    """The regression: the inference cost factor gated on `num_gpus > 0`, so a TPU/Trainium
    stage — `num_gpus == 0` plus a custom resource — was costed as a trivial column map, the
    cheapest node in the plan. Kyber then had no reason to push a filter below it, losing the
    optimization on exactly the hardware whose forward pass is most expensive."""
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    base = bt.from_pydict({"x": list(range(100))})
    model = CostModel(CardinalityEstimator(base._sources))
    plain = base.ml.map_batches(lambda b: b)._plan
    tpu = base.ml.map_batches(lambda b: b, resources={"TPU": 4}, model_memory_gb=8.0)._plan
    gpu = base.ml.map_batches(lambda b: b, num_gpus=1, model_memory_gb=8.0)._plan

    assert model.op_cost(tpu).cpu > model.op_cost(plain).cpu
    # And it is costed the same as the equivalent GPU stage — the device differs, not the work.
    assert model.op_cost(tpu).cpu == model.op_cost(gpu).cpu


def test_the_pool_bundle_reserves_what_the_actor_requests():
    """A bundle reserves; the actor then requests from its bundle. The two must agree.

    They did not for an inference pool. The device-tiled fan-out grants each worker a whole
    accelerator node's cores divided by its devices, so on a 4-node, 8-core, 1-device cluster
    a 4-actor pool reserved all 32 cores — for cores its actors no longer ask for. It is the
    reservation that then fails: anything else holding one core makes the gang unsatisfiable,
    and the pool burns the entire placement timeout before degrading to the default
    scheduling that would have worked at once.
    """
    from batcher.dist.executors.map import _gpu_options, _pool_placement_envelope

    device = SchedulingEnvelope(num_cpus=8.0, num_gpus=1.0, n_tasks=4)
    tuned = _pool_placement_envelope(device, _gpu_options(1.0, None, None))
    assert tuned.num_cpus == 0.0, "the bundle must not reserve cores the actor never requests"
    assert tuned.num_gpus == 1.0, "the device grant is what the bundle is actually for"
    assert tuned.n_tasks == 4, "only the CPU grant is restated"

    # A CPU-only pool (a class `fn` with no device) is untouched: there the core *is* the
    # resource being reserved, and the bundle was already honest.
    cpu = SchedulingEnvelope(num_cpus=4.0, n_tasks=2)
    assert _pool_placement_envelope(cpu, _gpu_options(0.0, None, None)) is cpu
    assert _pool_placement_envelope(None, _gpu_options(1.0, None, None)) is None
