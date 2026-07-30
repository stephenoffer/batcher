"""Reading the device table: one accessor per fact, and the name resolver in front of them.

Every accessor answers `0`, `0.0`, `""`, or `None` for a model the table does not carry, which
is the contract the whole GPU stack rests on: an unrecognized device leaves each decision with
whatever default it already had rather than acting on a fabricated figure.
"""

from __future__ import annotations

from batcher._internal.device_specs.table import SPECS, DeviceSpec

__all__ = [
    "device_fp8_tflops",
    "device_generation",
    "device_half_tflops",
    "device_host_link",
    "device_host_link_gbps",
    "device_idle_watts",
    "device_memory_bandwidth_gbps",
    "device_mig_slices",
    "device_nvlink_domain",
    "device_nvlink_gbps",
    "device_spec",
    "device_tdp_watts",
    "device_tflops_per_watt",
    "host_transfer_seconds",
    "known_device_names",
    "rank_devices_by_efficiency",
    "resolve_device_name",
]


def device_spec(accelerator_type: str | None) -> DeviceSpec | None:
    """The full specification for a Ray accelerator-type name, or `None` when unrecognized.

    Args:
        accelerator_type: A `ray.util.accelerators` model name such as `"NVIDIA_H100"`,
            matched case-insensitively. `None` and the empty string report unknown.

    Returns:
        The `DeviceSpec`, or `None` when the model is not in the table.
    """
    if not accelerator_type:
        return None
    return SPECS.get(accelerator_type.upper())


def known_device_names() -> tuple[str, ...]:
    """Every accelerator model name the table recognizes, newest generation first.

    Returns:
        The canonical uppercased names, in table order.
    """
    return tuple(SPECS)


def _field(accelerator_type: str | None, attr: str) -> float:
    spec = device_spec(accelerator_type)
    return float(getattr(spec, attr)) if spec is not None else 0.0


def device_tdp_watts(accelerator_type: str | None) -> float:
    """Board power limit at full load in watts, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Watts at the device's power limit, `0.0` if unrecognized or unpublished.
    """
    return _field(accelerator_type, "tdp_watts")


def device_idle_watts(accelerator_type: str | None) -> float:
    """Board power at idle in watts — what a reserved but unused device still burns.

    This is the figure that makes an idle GPU expensive rather than free, and it is why
    holding a device across a long CPU-bound stage is a real cost rather than a bookkeeping
    one.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Idle watts, `0.0` if unrecognized or unpublished.
    """
    return _field(accelerator_type, "idle_watts")


def device_memory_bandwidth_gbps(accelerator_type: str | None) -> float:
    """Peak device-memory bandwidth in GB/s, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Peak HBM/GDDR bandwidth in GB/s.
    """
    return _field(accelerator_type, "memory_bandwidth_gbps")


def device_half_tflops(accelerator_type: str | None) -> float:
    """Peak dense BF16/FP16 tensor throughput in TFLOP/s, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Dense half-precision TFLOP/s, without the structured-sparsity multiplier.
    """
    return _field(accelerator_type, "half_tflops")


def device_nvlink_domain(accelerator_type: str | None) -> int:
    """Accelerators reachable over one coherent vendor fabric, or `0` when unknown.

    `1` means the device is PCIe-attached only, so any multi-device collective crosses the
    host bus. A figure above one bounds how wide a tensor-parallel shard may go before its
    all-reduce leaves the fabric.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Devices in one NVLink/NVSwitch (or Infinity Fabric) domain, `0` if unrecognized.
    """
    spec = device_spec(accelerator_type)
    return spec.nvlink_domain if spec is not None else 0


def device_nvlink_gbps(accelerator_type: str | None) -> float:
    """Per-device bidirectional fabric bandwidth in GB/s, or `0.0` for PCIe-only devices.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        NVLink/Infinity Fabric bandwidth in GB/s.
    """
    return _field(accelerator_type, "nvlink_gbps")


def device_mig_slices(accelerator_type: str | None) -> int:
    """Maximum hardware partitions of one device, or `0` when it cannot be partitioned.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Maximum MIG instances per device, `0` for a device with no partitioning support.
    """
    spec = device_spec(accelerator_type)
    return spec.mig_slices if spec is not None else 0


def device_tflops_per_watt(accelerator_type: str | None) -> float:
    """Dense half-precision TFLOP/s per watt of board power, or `0.0` when unknown.

    The efficiency figure a power-constrained datacenter schedules on: two devices that
    deliver the same throughput are not equivalent if one draws twice the power, because the
    binding constraint on a full rack is the breaker, not the slot.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        TFLOP/s per watt, `0.0` when either figure is unknown.
    """
    spec = device_spec(accelerator_type)
    if spec is None or spec.tdp_watts <= 0 or spec.half_tflops <= 0:
        return 0.0
    return spec.half_tflops / spec.tdp_watts


def rank_devices_by_efficiency(names: list[str] | tuple[str, ...]) -> list[str]:
    """Recognized device names ordered most to least TFLOP/s per watt.

    Unrecognized names and devices with no published power figure are dropped rather than
    sorted to one end, because their position would be an invention: a datacenter placing
    work by efficiency needs the devices it can actually rank, not a list padded with
    guesses.

    Args:
        names: Candidate Ray accelerator-type names, in any order.

    Returns:
        The rankable subset, most efficient first; ties break on the name for determinism.
    """
    rankable = [(n, device_tflops_per_watt(n)) for n in names]
    return [n for n, eff in sorted(rankable, key=lambda p: (-p[1], p[0])) if eff > 0]


def device_fp8_tflops(accelerator_type: str | None) -> float:
    """Peak dense FP8 tensor throughput in TFLOP/s, `0.0` on a generation with no FP8 unit.

    The figure that decides whether quantizing a model buys throughput or only memory: on a
    part with an FP8 unit it roughly doubles the compute rate as well as halving the weights,
    and on one without it buys the memory alone.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Dense FP8 TFLOP/s, `0.0` when unsupported or unknown.
    """
    return _field(accelerator_type, "fp8_tflops")


def device_generation(accelerator_type: str | None) -> str:
    """The device's architecture family, or `""` when unrecognized.

    The right key for anything learned *per capability set* rather than per model: an H100 and
    an H200 differ in memory and bandwidth but share a instruction set and an FP8 unit, so a
    measurement from one transfers to the other in a way an Ampere measurement does not.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        The generation name (`"hopper"`, `"blackwell"`, `"ampere"`, ...), lowercase.
    """
    spec = device_spec(accelerator_type)
    return spec.generation if spec is not None else ""


def _tokens(name: str) -> list[str]:
    """A device name split into uppercase alphanumeric tokens."""
    return [t for t in "".join(c if c.isalnum() else " " for c in name).upper().split() if t]


def _model_token(key: str) -> str:
    """The token that identifies the *part* in a table key: the first one carrying a digit.

    `NVIDIA_A100_80G` yields `A100`, not `80G`. The distinction is the whole point — a memory
    token matches across parts (`80G` prefixes the `80GB` in an H100's reported name), so
    keying on it silently resolves an H100 to an A100 and plans against the wrong bandwidth,
    the wrong power, and the wrong tensor rate.
    """
    for token in _tokens(key):
        if any(c.isdigit() for c in token):
            return token
    return ""


def resolve_device_name(reported: str | None) -> str | None:
    """Map a driver-reported device name onto a table key, or `None` when nothing matches.

    The two vocabularies do not agree and never will. Ray labels a node `NVIDIA_A100_80G`;
    the driver reports `"NVIDIA A100-SXM4-80GB"` and NVML reports `"NVIDIA H100 80GB HBM3"`,
    each carrying a board variant, a form factor, and a memory size in its own spelling.
    Matching by equality therefore fails on every locally probed device, which is exactly
    where the power and bandwidth figures are most useful.

    Two rules, in order. The key's **part token** — its first token carrying a digit — must
    match a token of the reported name, which is what keeps an H100 from resolving to an A100
    through a shared `80G`. Among the keys that pass, the one sharing the most tokens wins,
    and a tie goes to the *shorter* key, so a name with no memory size resolves to the
    conservative base entry rather than to its largest variant.

    Args:
        reported: A device name as a driver, framework, or node label reports it.

    Returns:
        The canonical table key, or `None` when nothing matches. `None` means unknown, and
        every accessor then reports unknown rather than a nearest guess.
    """
    if not reported:
        return None
    normalized = "".join(c if c.isalnum() else "_" for c in reported).upper()
    if normalized in SPECS:
        return normalized
    name_tokens = _tokens(reported)
    if not name_tokens:
        return None
    best: tuple[int, int, str] | None = None
    for key in SPECS:
        part = _model_token(key)
        if not part or not any(t.startswith(part) or part.startswith(t) for t in name_tokens):
            continue
        key_tokens = _tokens(key)
        score = sum(1 for kt in key_tokens if any(nt.startswith(kt) for nt in name_tokens))
        if score < 2:
            continue
        candidate = (score, -len(key), key)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else None


def device_host_link(accelerator_type: str | None) -> str:
    """How the device reaches host memory: `pcie3`/`pcie4`/`pcie5`, `nvlink-c2c`, or `""`.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        The host interface label, `""` when unknown.
    """
    spec = device_spec(accelerator_type)
    return spec.host_link if spec is not None else ""


def device_host_link_gbps(accelerator_type: str | None) -> float:
    """Effective one-way host-to-device bandwidth in GB/s, or `0.0` when unknown.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        Effective (not theoretical) GB/s across the host link.
    """
    return _field(accelerator_type, "host_link_gbps")


def host_transfer_seconds(
    nbytes: float,
    accelerator_type: str | None,
    *,
    round_trip: bool = False,
) -> float:
    """Time to move `nbytes` across the host link, in seconds.

    The cost a data engine forgets and then cannot explain. A relational stage does not start
    with its data on the device: every byte crosses the host link first, and on PCIe that link
    is slower than a server's own memory bandwidth. Ten gigabytes over PCIe 4.0 is four tenths
    of a second before a single kernel launches, which is longer than the CPU takes to scan it
    outright. That is why a device wins on inference and loses on a projection, and it is the
    term that makes the two verdicts fall out of the same arithmetic instead of a heuristic.

    A coherent CPU-GPU package (`nvlink-c2c`) changes the answer rather than shading it: an
    order of magnitude more host bandwidth moves the break-even far enough that scan-shaped
    work becomes worth offloading, which is exactly what those parts were built for.

    Args:
        nbytes: Bytes to move.
        accelerator_type: A Ray accelerator-type name.
        round_trip: Charge the result's return trip as well. Results are usually far smaller
            than inputs, so a caller with a real output size should pass that separately
            rather than doubling the input.

    Returns:
        Seconds, or `0.0` when the device or its link is unknown — which reads as "no
        transfer cost modelled", so a caller falls back to whatever it did before rather
        than to a fabricated penalty.
    """
    gbps = device_host_link_gbps(accelerator_type)
    if gbps <= 0 or nbytes <= 0:
        return 0.0
    seconds = nbytes / (gbps * 1e9)
    return seconds * 2 if round_trip else seconds
