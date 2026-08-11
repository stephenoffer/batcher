"""Accelerator model to device memory — the one hardware fact a cluster cannot report.

Ray advertises an accelerator as a *count* (the `GPU` resource) and a *model name* (the
`ray.io/accelerator-type` node label). It never reports device memory. That leaves a
distributed run unable to answer "how much VRAM does a worker's device have?", and the
answer it fell back on was the driver's own `gpu_inventory()` — which on the usual
topology (a CPU-only head node scheduling GPU workers) sees no device at all and returns
a hardcoded 12 GB. Every downstream sizing decision was then made for a T4 that nothing
in the cluster was running.

This module closes that hole the only way available without an RPC to every node: a
lookup from the model name Ray already tells us to that model's nameplate memory. It
lives at layer 0 as *inventory*, next to `hardware.py`, for the same reason that module
gives — "what is this device" is a hardware fact, while "how much of it to use" is
policy that belongs to `ml.gpu` and Kyber.

**Deliberately conservative in two ways.** Where a model ships in several memory
configurations under one Ray name (an A100 is 40 GB or 80 GB), the table records the
*smallest*, because these figures size a working set that must be valid on every device
it might land on — under-estimating shards more than necessary, over-estimating OOMs.
And an unrecognized name returns `0` ("unknown") rather than a guess, so a device this
table has never heard of degrades to the caller's existing default instead of to a
wrong number.
"""

from __future__ import annotations

import functools
import glob
import os
import sys

__all__ = [
    "ACCELERATOR_RESOURCE_NAMES",
    "accelerator_backend",
    "accelerator_memory_bytes",
    "accelerator_units",
    "binding_gpu_memory_bytes",
    "gpu_devices_absent",
    "gpu_inventory",
    "has_gaudi_device",
    "has_neuron_device",
    "is_accelerator_node",
    "reset_accelerator_probes",
]


def reset_accelerator_probes() -> None:
    """Forget the memoized device probes, so the next call re-reads the machine.

    Both probes below answer once and remember: the device set attached to a running
    process does not change, and the probing itself is expensive (two optional-package
    imports and several `/dev` globs) on a path every terminal op reaches. A test that
    fakes the device nodes has to invalidate them; this is that hook, re-exported by
    `hardware.reset_hardware_probes` so there is one call to make.
    """
    gpu_devices_absent.cache_clear()
    _gpu_inventory_probe.cache_clear()


def has_neuron_device() -> bool:
    """AWS Trainium/Inferentia via `/dev/neuron*` — a free, unambiguous device-node check
    (`torch_neuronx` initializes its runtime on import, so probing the framework costs seconds)."""
    return bool(glob.glob("/dev/neuron[0-9]*"))


def has_gaudi_device() -> bool:
    """Intel Gaudi (Habana) via its `/dev/hl*` device nodes."""
    return bool(glob.glob("/dev/hl[0-9]*"))


_GIB = 1 << 30

#: Ray custom-resource names for accelerators that are *not* reported as `GPU`. Ray advertises
#: NVIDIA/AMD/Intel/MetaX as `GPU`; everything else is a named resource — `TPU` (Google Cloud
#: TPU), `neuron_cores` (AWS Trainium/Inferentia), `HPU` (Intel Gaudi), `NPU`. A node exposing
#: any of these is an accelerator node even though its `GPU` count is zero. This is the one place
#: the vocabulary lives, so node classification and pool sizing agree on what counts as one.
ACCELERATOR_RESOURCE_NAMES = ("TPU", "neuron_cores", "HPU", "NPU")


def accelerator_units(resources: dict[str, float] | None) -> float:
    """Total non-GPU accelerator units a node's Ray `Resources` advertises (`0.0` for none).

    The max across the known accelerator resource names rather than the sum: a node is a TPU
    node *or* a Trainium node, not both, and mixing units of different devices would be
    meaningless. Used to recognize an accelerator node whose `GPU` count is zero.

    Args:
        resources: A Ray node's `Resources` mapping, or `None`.

    Returns:
        The largest accelerator-resource amount present, or `0.0` when none is.
    """
    if not resources:
        return 0.0
    return max((float(resources.get(n, 0.0)) for n in ACCELERATOR_RESOURCE_NAMES), default=0.0)


def is_accelerator_node(node_class: dict) -> bool:
    """Whether a `node_classes()` entry is an accelerator node: a GPU *or* a custom-accelerator
    (TPU / Trainium / Gaudi / NPU) node. Keying on GPUs alone left a pure TPU-plus-CPU cluster
    with no CPU-fleet isolation and counted a TPU node's cores as free for a CPU shuffle."""
    return node_class.get("gpus", 0.0) > 0 or node_class.get("accelerators", 0.0) > 0


#: Device memory lives in `device_specs.py`, which records it beside the rest of a model's
#: nameplate figures (power, bandwidth, fabric width, partitionability). It is read from
#: there rather than kept a second time here: two tables of the same fact drift, and the one
#: that drifts is always the one a given caller does not happen to use.
#:
#: Deliberately absent from that table: AWS Neuron (`aws-neuron-core`) and Intel Gaudi
#: (`Intel-GAUDI`). Ray exposes each generation under ONE label — `aws-neuron-core` covers
#: inf2 (32 GB/chip) and trn1/trn2 (32/96 GB) alike, `Intel-GAUDI` covers Gaudi2 (96 GB) and
#: Gaudi3 (128 GB) — so the label does not determine the memory. Per this module's contract an
#: ambiguous name returns `0` ("unknown") rather than a fabricated figure.


def binding_gpu_memory_bytes(classes: list[dict]) -> int:
    """VRAM of the smallest device model across a cluster's GPU nodes, or `0` when unknowable.

    Ray reports GPU *count* and a model *name* (`ray.io/accelerator-type`) but never device
    memory, so the size is recovered from the name via `accelerator_memory_bytes`. The minimum
    is the binding figure: a working set sized to the largest device would OOM every smaller
    one it lands on.

    Returns `0` ("unknown") when any GPU node's model is unrecognized or unlabelled, rather than
    the minimum of the ones that *were* recognized — an unknown device could be smaller than
    every known one, so a partial minimum is not a bound, and the caller's default is safer.

    Args:
        classes: `node_classes()`-shaped entries, each carrying `gpus` and `accelerator_type`.

    Returns:
        Smallest recognized GPU-node VRAM in bytes, or `0` when it can't be determined.
    """
    sizes = [accelerator_memory_bytes(c.get("accelerator_type")) for c in classes if c["gpus"] > 0]
    if not sizes or any(s <= 0 for s in sizes):
        return 0
    return min(sizes)


def accelerator_memory_bytes(accelerator_type: str | None) -> int:
    """Device memory in bytes for a Ray accelerator-type name, or `0` when unknown.

    `0` is the same "unknown" sentinel `HardwareProfile` uses, so an unrecognized or
    absent model name leaves the caller on whatever default it already had rather than
    substituting a fabricated figure.

    Args:
        accelerator_type: A `ray.util.accelerators` model name such as `"NVIDIA_A100"`,
            typically read from the `ray.io/accelerator-type` node label. `None` and the
            empty string are accepted and report unknown.

    Returns:
        Total device memory in bytes, or `0` if the model is not recognized.
    """
    from batcher._internal.device_specs import device_spec

    spec = device_spec(accelerator_type)
    return spec.memory_gib * _GIB if spec is not None else 0


# Device nodes of accelerators that are not NVIDIA GPUs: Google TPU (`/dev/accel*`, and
# `/dev/vfio` for v5+), AWS Neuron / Trainium / Inferentia (`/dev/neuron*`), and Intel
# Gaudi (`/dev/hl*`). Used only to *refuse* the cheap negative — presence here means "ask
# properly", never "an accelerator is usable".
_ACCELERATOR_DEVICE_GLOBS = (
    "/dev/accel[0-9]*",
    "/dev/neuron[0-9]*",
    "/dev/hl[0-9]*",
    "/dev/vfio/[0-9]*",
)


@functools.lru_cache(maxsize=1)
def gpu_devices_absent() -> bool:
    """True when this host demonstrably has no GPU, decided without importing a framework.

    Answers only the *cheap negative*. Proving a GPU is present needs the vendor runtime, but
    proving one is absent usually does not, and that asymmetry is worth exploiting: the natural
    way to ask "is there a GPU" is `torch.cuda.is_available()`, which costs ~2 s of import on
    first call — paid on the first query of every GPU-less run, to learn there is nothing to
    accelerate. A `stat` of a device node answers the same question in microseconds.

    Deliberately conservative: it returns True only on Linux, where the vendor device nodes are
    authoritative, and only when every one of them is missing. Anywhere else — macOS, whose
    Metal devices have no node, or an unrecognized accelerator — it returns False, meaning "ask
    properly", so the cheap path can never produce a false negative on a machine that has a
    device. An explicitly empty ``CUDA_VISIBLE_DEVICES`` is honored as a definitive no.

    Returns:
        True only if a real probe is certain to find nothing.
    """
    if _devices_masked_off():
        return True  # the runtime or the user masked every device; nothing to find
    if not sys.platform.startswith("linux"):
        return False  # no authoritative node to check — make the caller probe for real
    # Numbered *device* nodes, not the driver's control node: a GPU-less machine built from a
    # GPU-capable cloud image has `/dev/nvidiactl` (the driver is loaded) and no `/dev/nvidia0`
    # at all. Keying on the control node would call that host GPU-equipped and re-introduce the
    # very import this exists to avoid, on exactly the fleet where it is most common.
    if glob.glob("/dev/nvidia[0-9]*"):
        return False
    # Non-NVIDIA accelerators expose their own nodes. Without these, a TPU, Trainium, or
    # Gaudi host answered "definitely nothing here" — a *false* negative, which is the one
    # answer this function promises never to give, and it suppressed the real probe on the
    # machines that most need it. Globs, because these are numbered per device.
    if any(glob.glob(pattern) for pattern in _ACCELERATOR_DEVICE_GLOBS):
        return False
    return not any(os.path.exists(p) for p in ("/dev/kfd", "/dev/dxg"))


#: Values of `CUDA_VISIBLE_DEVICES` (and its AMD equivalents) that mean "no device at all".
#: The empty string is the spelling this module already honored; `-1` is the one CUDA itself
#: documents and the one every framework and CI script actually writes, and it was read as an
#: ordinal list, so a run explicitly asked to stay off the GPU still paid the ~2 s `import
#: torch` to be told there was nothing to accelerate — and still enumerated the node's devices
#: through NVML afterwards.
_NO_DEVICES = frozenset({"", "-1", "none", "void"})

#: The container-runtime variable, which is set *before* the process starts and decides which
#: devices the NVIDIA container toolkit injects. `none` and `void` are its documented ways of
#: saying "inject nothing", and a container started that way has no device nodes to find —
#: reading it lets the cheap negative answer without touching the filesystem.
_CONTAINER_VISIBLE_VAR = "NVIDIA_VISIBLE_DEVICES"


def _devices_masked_off() -> bool:
    """Whether an environment variable has definitively hidden every accelerator."""
    for var in (*_VISIBLE_DEVICE_VARS, _CONTAINER_VISIBLE_VAR):
        raw = os.environ.get(var)
        if raw is not None and raw.strip().lower() in _NO_DEVICES:
            return True
    return False


def accelerator_backend() -> str:
    """The accelerator this host can compute on: ``cuda``/``rocm``/``xpu``/``mps``/``tpu``/
    ``neuron``/``hpu``/``cpu``.

    A hardware *fact*, so it lives here (the executor picks a device without importing `ml`);
    `ml.gpu.detect_backend` re-exports it. Detected via torch where one exists (ROCm through
    the CUDA API with ``torch.version.hip``; Intel ``torch.xpu``; Apple MPS; Cloud TPU
    ``torch_xla``) and via device nodes for Trainium/Inferentia (``neuron``) and Gaudi
    (``hpu``), whose frameworks are expensive to import. Naming the specific backend rather
    than ``cpu`` lets such a host self-identify for diagnostics and `torch_device`.
    """
    if gpu_devices_absent() and not _tpu_available():
        return "cpu"  # cheap negative first: `torch.cuda.is_available()` costs a ~2 s import
    try:
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass  # no torch → fall through to the device-node accelerators
    from batcher._internal.accelerators import has_gaudi_device, has_neuron_device

    if _tpu_available():
        return "tpu"
    if has_neuron_device():
        return "neuron"
    if has_gaudi_device():
        return "hpu"
    # AMD last among the device-node checks, and only once torch has declined: a ROCm torch
    # answers `torch.cuda.is_available()` above and is the better source. Without one, an
    # Instinct node with the driver loaded reported `cpu` — the same "nobody looked" that the
    # rest of the AMD path exists to remove, and here it named the wrong hardware rather than
    # naming none. Device nodes are already the accepted evidence for `neuron` and `hpu`.
    from batcher._internal.hardware.amd import amd_present

    return "rocm" if amd_present() else "cpu"


def _tpu_available() -> bool:
    """Whether a Cloud TPU is present, via `torch_xla` — import-gated and side-effect-free.

    `find_spec` avoids importing (and so initializing) the XLA runtime on the common
    no-TPU host; only when `torch_xla` is actually installed do we ask its runtime for the
    device type. Any failure (older API, no device) reads as "no TPU"."""
    import importlib.util

    if importlib.util.find_spec("torch_xla") is None:
        return False
    try:
        import torch_xla.runtime as xr  # type: ignore[import-not-found]

        return xr.device_type() == "TPU"
    except Exception:
        return False


def gpu_inventory() -> list[dict[str, object]]:
    """Visible GPUs as ``{"index", "name", "memory_bytes"}``, or `[]` when none/undetectable.

    **Inventory only, deliberately.** This answers "what devices are attached", which is a
    hardware fact and so belongs in this neutral layer where any package may read it. It does
    *not* decide how to use them — VRAM budgeting, actors-per-GPU, backend selection, and the
    autocast policy all live in `ml.gpu`, which is the executor-facing owner of those
    decisions. Keeping the split at inventory-vs-policy is what stops this from becoming a
    second, competing GPU module: there is one place that says what exists and one that says
    what to do about it.

    Best-effort and dependency-free at import: NVML first (accurate, no CUDA context, and it
    sees devices even when no framework is installed), then torch as a fallback. Both are
    optional, so a CPU-only or stripped-down install gets `[]` rather than an error.

    Returns:
        One dict per visible device, in device order.
    """
    # Fresh dicts over a memoized probe: the *probe* is what costs (two optional-package
    # imports that, when absent, re-walk `sys.path` on every call because a failed import is
    # never cached, plus three `/dev` globs), and it runs on every terminal op through the
    # GPU-routing decision. The device set cannot change under a running process, so probing
    # once is correct; copying the dicts out keeps a caller that annotates a device entry
    # from mutating what every later caller sees.
    return [dict(device) for device in _gpu_inventory_probe()]


@functools.lru_cache(maxsize=1)
def _gpu_inventory_probe() -> tuple[dict[str, object], ...]:
    """The one-shot device probe behind `gpu_inventory` — NVML, then torch, then device nodes.

    The torch fallback is gated on [`gpu_devices_absent`], which is the whole reason that
    function exists. Without the gate, a host with torch installed and no GPU paid a ~1.4 s
    `import torch` on its **first relational query** — `Optimizer.__init__` builds a
    `HardwareProfile`, which asks for the GPU inventory — purely to be told there is nothing
    to accelerate. That is the single largest fixed cost in a cold process, and it lands on
    pure SQL that will never touch a device. Skipping is not a heuristic: the gate returns
    True only when every vendor device node is missing, and `torch.cuda.is_available()` on
    such a host returns False, so the branch it skips is provably empty.
    """
    # Both driver probes below enumerate what is physically attached, so each is narrowed to
    # what this process may address. `_torch_inventory` is deliberately NOT narrowed: torch
    # applies the same variable itself and already reports the visible set renumbered from
    # zero, so filtering it again would select visible devices by physical slot.
    nvml = _visible_devices(_nvml_inventory())
    if nvml:
        return tuple(nvml)
    # AMD before torch, and for the same reason NVML comes before torch: it is a handful of
    # sysfs reads against an import that costs over a second, it needs no ROCm install, and it
    # reports the real HBM size where the device-node fallback below reports zero. Without it
    # an MI300X node with a CPU-only wheel enumerated no devices at all.
    amd = _visible_devices(_amd_inventory())
    if amd:
        return tuple(amd)
    torch_devices = [] if gpu_devices_absent() else _torch_inventory()
    return tuple(torch_devices or _other_accelerator_inventory())


def _amd_inventory() -> list[dict[str, object]]:
    """AMD accelerators via the `amdgpu` driver's sysfs tree, or `[]`.

    Dependency-free by construction: `amdsmi` ships with ROCm and is absent from the framework
    containers most of this hardware is rented with, so the driver's own files are the only
    source that is always there.
    """
    from batcher._internal.hardware.amd import amd_devices

    return [
        {
            "index": device.index,
            "name": device.name or f"AMD GPU ({device.card})",
            "memory_bytes": device.memory_total_bytes,
        }
        for device in amd_devices()
    ]


#: The env vars a runtime uses to hand a process a *subset* of a node's accelerators, per
#: vendor. Ray sets the NVIDIA one on every task and actor holding a `num_gpus` grant, which
#: is what makes this the normal case on a multi-device node rather than an exotic one.
_VISIBLE_DEVICE_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def _visible_devices(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    """Restrict a driver-probed device list to the ones this process may actually use.

    NVML and the AMD sysfs tree both enumerate every device *physically present*, and neither
    honors the visibility env var the way a framework does. That is the whole gap: on a
    four-device node an actor granted one GPU sees `CUDA_VISIBLE_DEVICES=0`, `torch` reports a
    device count of one, and the driver probe reports four — so the same function answered
    "four devices, 60 GiB" for a process entitled to one device and 15 GiB, purely according
    to which backend happened to answer. The torch fallback in this module already returns the
    visible set, so the two paths disagreed with each other as well as with the docstring.

    Indices are renumbered from zero in the order the variable lists them, which is what CUDA
    itself does and therefore what makes `gpu_inventory()[i]` line up with `torch.cuda`'s
    device `i` rather than with a physical slot the process cannot address.

    An **unset** variable means "everything is visible", the pre-existing answer.

    A value that is *not* a list of ordinals — the UUID form (`GPU-<uuid>`) a Kubernetes device
    plugin writes, or a `MIG-<uuid>` partition handle — is resolved through
    `hardware.devices.scope`, which asks the driver which board each identifier names. That
    matters most on the fleets that pin hardest: index-only parsing could not read either form,
    and the fallback was to report *every device on the node*, so the pool sizing, the health
    check, and the accelerator report all described eight boards to a pod entitled to one.

    Falling back to `devices` unchanged when that resolution finds nothing is why this is not
    delegated wholesale: `scope` reads NVML and an AMD host has none, so a wholesale fold would
    hide every device on an Instinct node — a worse error than reporting too many, which is
    also what every caller already handled.

    Args:
        devices: The physically probed devices, in driver order.

    Returns:
        The visible subset, renumbered from zero, or `devices` unchanged when visibility is
        not restricted or cannot be resolved at all.
    """
    raw = next((os.environ[v] for v in _VISIBLE_DEVICE_VARS if v in os.environ), None)
    if raw is None:
        return devices
    if raw.strip().lower() in _NO_DEVICES:
        # `-1` is CUDA's own documented "no devices", and it is what a framework or a CI script
        # writes to keep a run off the GPU. Only the empty string was recognized, so `-1` fell
        # through the ordinal parse (it is not `isdigit`), failed to resolve as a UUID, and hit
        # the "could not resolve" fallback — which returns EVERY device on the node. A pod
        # explicitly denied the GPU was therefore reported as owning all eight of them, and the
        # pool sized itself accordingly.
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []  # explicitly empty: the runtime hid every device
    if not all(t.isdigit() for t in tokens):
        return _resolved_by_driver(devices) or devices
    picked: list[dict[str, object]] = []
    for token in tokens:
        # CUDA stops enumerating at the first entry it cannot resolve, so a trailing bad
        # index truncates rather than invalidating the whole list.
        slot = int(token)
        if slot >= len(devices):
            break
        picked.append(dict(devices[slot], index=len(picked)))
    return picked


def _resolved_by_driver(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    """The visible subset for a UUID/MIG pin, resolved against the driver, or `[]`.

    Imported inside the call because this is the uncommon branch and the module it reaches for
    initializes NVML; a host pinned by ordinals — every Ray worker — must not pay for it.
    """
    try:
        from batcher._internal.hardware.devices.scope import visible_device_indices

        indices = visible_device_indices()
    except Exception:
        return []
    # A scope covering every device is what an *unresolvable* pin also produces, so it carries
    # no information here and is treated as "could not resolve".
    if not indices or len(indices) >= len(devices):
        return []
    return [
        dict(devices[i], index=position)
        for position, i in enumerate(indices)
        if 0 <= i < len(devices)
    ]


def _nvml_inventory() -> list[dict[str, object]]:
    """GPUs via NVML, or `[]`. Preferred: no CUDA context, and no framework required."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            out: list[dict[str, object]] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                out.append(
                    {
                        "index": index,
                        "name": name.decode() if isinstance(name, bytes) else str(name),
                        "memory_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    }
                )
            return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - NVML absent or no NVIDIA driver
        return []


def _torch_inventory() -> list[dict[str, object]]:
    """GPUs via torch, or `[]`. The fallback when NVML is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception:  # pragma: no cover - torch absent or CUDA unusable
        return []


def _other_accelerator_inventory() -> list[dict[str, object]]:
    """Non-NVIDIA accelerators, so diagnostics stop reporting `[]` on a machine that has one.

    NVML and `torch.cuda` between them cover NVIDIA and (via HIP) AMD; everything else fell
    through to an empty list, which the observability page then rendered as "no GPUs" on a
    TPU, Trainium, or Gaudi host. Reports what the device nodes say, since there is no
    portable cross-vendor API and no memory figure to be had without each vendor's runtime —
    an accurate name with unknown memory beats a confident, wrong "nothing here".
    """
    devices: list[dict[str, object]] = []
    for kind, pattern in (
        ("TPU", "/dev/accel[0-9]*"),
        ("Neuron", "/dev/neuron[0-9]*"),
        ("Gaudi", "/dev/hl[0-9]*"),
    ):
        for index, node in enumerate(sorted(glob.glob(pattern))):
            devices.append(
                {"index": index, "name": f"{kind} ({os.path.basename(node)})", "memory_bytes": 0}
            )
    return devices
