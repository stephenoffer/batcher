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

import glob

__all__ = [
    "ACCELERATOR_RESOURCE_NAMES",
    "accelerator_memory_bytes",
    "accelerator_units",
    "binding_gpu_memory_bytes",
    "has_gaudi_device",
    "has_neuron_device",
    "is_accelerator_node",
]


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


#: Nameplate device memory per `ray.util.accelerators` model name, in GiB. Keys are
#: uppercased at lookup, so Ray's inconsistent casing (`NVIDIA_TESLA_T4` beside
#: `AMD_Instinct_MI300X`) resolves either way. Where one name covers several memory
#: configurations the smallest shipping variant is recorded — see the module docstring.
_DEVICE_MEMORY_GIB: dict[str, int] = {
    # NVIDIA datacenter
    "NVIDIA_TESLA_K80": 12,
    "NVIDIA_TESLA_P4": 8,
    "NVIDIA_TESLA_P100": 16,
    "NVIDIA_TESLA_V100": 16,  # also ships 32 GB
    "NVIDIA_TESLA_T4": 16,
    "NVIDIA_A10": 24,
    "NVIDIA_A10G": 24,
    "NVIDIA_L4": 24,
    "NVIDIA_L40S": 48,
    "NVIDIA_A100": 40,  # also ships 80 GB — see the explicit variants below
    "NVIDIA_A100_40G": 40,
    "NVIDIA_A100_80G": 80,
    "NVIDIA_H100": 80,
    "NVIDIA_H200": 141,
    "NVIDIA_B200": 180,
    # AMD Instinct
    "AMD_INSTINCT_MI210": 64,
    "AMD_INSTINCT_MI250X": 128,
    "AMD_INSTINCT_MI300X": 192,
    # Intel Data Center GPU Max
    "INTEL_MAX_1100": 48,
    "INTEL_MAX_1550": 128,
    # Google Cloud TPU — HBM per chip (the unit Ray's `TPU` resource counts). Version names
    # are determinate, unlike the vendor-generic labels below, so their memory is knowable.
    "TPU-V2": 8,
    "TPU-V3": 16,
    "TPU-V4": 32,
    "TPU-V5E": 16,
    "TPU-V5LITEPOD": 16,  # Ray's name for the v5e generation
    "TPU-V5P": 95,
    "TPU-V6E": 32,  # Trillium
    #
    # Deliberately absent: AWS Neuron (`aws-neuron-core`) and Intel Gaudi (`Intel-GAUDI`).
    # Ray exposes each generation under ONE label — `aws-neuron-core` covers inf2 (32 GB/chip)
    # and trn1/trn2 (32/96 GB) alike, `Intel-GAUDI` covers Gaudi2 (96 GB) and Gaudi3 (128 GB) —
    # so the label does not determine the memory. Per this module's contract an ambiguous name
    # returns `0` ("unknown") rather than a fabricated figure, which is safer than guessing wrong.
}


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
    if not accelerator_type:
        return 0
    return _DEVICE_MEMORY_GIB.get(accelerator_type.upper(), 0) * _GIB
