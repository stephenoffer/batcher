"""CPU identity and instruction-set features — what this silicon can actually execute.

Two machines with the same core count and cache sizes still differ in the width of their
vector units, and that width is a multiplier on every columnar kernel in the data plane: a
512-bit lane processes eight doubles per instruction where a 128-bit lane processes two. It
also decides whether a JIT-compiled expression is worth compiling at all, since the fixed
compile cost amortizes over a very different per-row saving on each.

The vendor and model name matter for a second reason: they are the stable part of a machine's
identity. Learned coefficients gathered on a Graviton3 do not transfer to an Ice Lake, and the
only way to keep them apart in a shared metadata store is to record which one measured them.

Linux-only (`/proc/cpuinfo`); reports an empty feature set elsewhere, which callers read as
"assume the baseline", exactly the assumption in force before this existed.
"""

from __future__ import annotations

import functools
import platform

__all__ = [
    "cpu_features",
    "cpu_model_name",
    "cpu_vendor",
    "simd_width_bits",
    "vendor_display_name",
]

# Feature flags worth distinguishing, in ascending order of vector width. Everything else
# /proc/cpuinfo lists is either universal on the targets Batcher supports or irrelevant to a
# columnar kernel; carrying the full flag list would make the hardware fingerprint churn on
# microcode updates that change nothing the engine can use.
_X86_WIDTHS: tuple[tuple[str, int], ...] = (
    ("sse2", 128),
    ("avx", 256),
    ("avx2", 256),
    ("avx512f", 512),
)
#: SVE has **no architectural width**: an implementation picks any multiple of 128 bits from
#: 128 to 2048, and the parts Batcher runs on genuinely differ — Neoverse V1 (Graviton3) is
#: 256-bit, Neoverse V2 (Graviton4) is 128, A64FX is 512. So the entries below are a *floor*
#: (SVE implies at least NEON's 128) and the real width is read from the kernel by
#: [`_sve_vector_bits`]. Assuming 256 for every SVE part overstated Graviton4 twofold in a
#: figure used as a per-row kernel throughput multiplier and as fingerprint material.
_ARM_WIDTHS: tuple[tuple[str, int], ...] = (
    ("asimd", 128),
    ("neon", 128),
    ("sve", 128),
    ("sve2", 128),
)

#: Where the kernel publishes the default SVE vector length, in **bytes**. Preferred over a
#: `prctl(PR_SVE_GET_VL)` call for the reason every probe in this package reads a file: it needs
#: no `ctypes` signature, it cannot fault, and it is absent rather than wrong on a kernel
#: without SVE support.
_SVE_VECTOR_LENGTH_PATH = "/proc/sys/abi/sve_default_vector_length"


@functools.lru_cache(maxsize=1)
def _cpuinfo_fields() -> dict[str, str]:
    """The first processor block of ``/proc/cpuinfo`` as a field map (empty off Linux).

    Only the first block is read: the fields this module wants (vendor, model, flags) are
    identical across cores on every machine Batcher targets, and parsing 128 identical blocks
    to prove it would be pure cost.
    """
    fields: dict[str, str] = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if not line.strip():
                    if fields:  # end of the first processor block
                        break
                    continue
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()
    except OSError:
        return {}
    return fields


@functools.lru_cache(maxsize=1)
def cpu_features() -> frozenset[str]:
    """The instruction-set feature flags this CPU advertises, lowercased.

    Restricted to the flags that change how the data plane should behave — the vector-width
    ladder — rather than the full `/proc/cpuinfo` list. A fingerprint built from every flag
    would change on a microcode update that alters nothing the engine can exploit, silently
    discarding every coefficient learned on the machine.

    Returns:
        The recognized feature flags present on this CPU, empty when undetectable.
    """
    raw = _cpuinfo_fields()
    # x86 calls the list "flags"; ARM calls it "features".
    listed = set((raw.get("flags") or raw.get("features") or "").lower().split())
    known = {name for name, _ in _X86_WIDTHS} | {name for name, _ in _ARM_WIDTHS}
    return frozenset(listed & known)


def _sve_vector_bits() -> int:
    """The kernel's SVE vector length in bits, or `0` when this machine has no SVE.

    The file reports **bytes**, and only on an SVE-capable aarch64 kernel. A value outside the
    architectural range (128 to 2048 bits, in multiples of 128) is discarded rather than
    trusted: it would have to come from a kernel reporting something this code does not
    understand, and a fabricated vector width propagates into the machine fingerprint.
    """
    try:
        with open(_SVE_VECTOR_LENGTH_PATH) as f:
            vector_bytes = int(f.read().strip())
    except (OSError, ValueError):
        return 0
    bits = vector_bytes * 8
    return bits if 128 <= bits <= 2048 and bits % 128 == 0 else 0


@functools.lru_cache(maxsize=1)
def simd_width_bits() -> int:
    """The widest SIMD register this CPU offers, in bits — `128` when undetectable.

    The multiplier on per-row kernel throughput and the number a JIT payoff estimate scales
    with. Falls back to 128 rather than 0 because every 64-bit target Batcher supports has at
    least SSE2 or NEON, so 128 is a floor rather than a guess.

    On an SVE part the width is **read from the kernel** rather than assumed, because SVE has
    no architectural width — an implementation picks anything from 128 to 2048 bits, and the
    server parts differ: Graviton3 is 256-bit, Graviton4 is 128, A64FX is 512. A flat 256 was
    therefore right on one of those three and overstated Graviton4 by 2x, in a figure that both
    scales a throughput estimate and keys every learned coefficient on the machine.

    Returns:
        The widest available vector width in bits, at least 128.
    """
    present = cpu_features()
    widths = [bits for name, bits in (*_X86_WIDTHS, *_ARM_WIDTHS) if name in present]
    if present & {"sve", "sve2"}:
        widths.append(_sve_vector_bits())
    return max([*widths, 128])


#: ARM implementer codes, as `/proc/cpuinfo` publishes them, to the name they identify. Used
#: **only for display** — never to build the fingerprint — because the code is already a
#: perfectly stable discriminator and remapping it would change every ARM machine's key and
#: silently discard everything learned on it. What it is not is *readable*: a log line or an
#: `EXPLAIN` reading `0x41/64c/128GiB` tells nobody which fleet ran the query.
_ARM_IMPLEMENTERS: dict[str, str] = {
    "0x41": "ARM",
    "0x42": "Broadcom",
    "0x43": "Cavium",
    "0x44": "DEC",
    "0x46": "Fujitsu",
    "0x48": "HiSilicon",
    "0x49": "Infineon",
    "0x4e": "NVIDIA",
    "0x50": "APM",
    "0x51": "Qualcomm",
    "0x53": "Samsung",
    "0x56": "Marvell",
    "0x61": "Apple",
    "0x69": "Intel",
    "0xc0": "Ampere",
}


def vendor_display_name(vendor: str) -> str:
    """A human-readable form of a `cpu_vendor()` value, for logs and `EXPLAIN`.

    Args:
        vendor: The value `cpu_vendor()` returned.

    Returns:
        The vendor's name where it is an ARM implementer code, otherwise `vendor` unchanged.
    """
    return _ARM_IMPLEMENTERS.get(vendor.strip().lower(), vendor)


@functools.lru_cache(maxsize=1)
def cpu_vendor() -> str:
    """The CPU vendor identifier — ``GenuineIntel``, ``AuthenticAMD``, an ARM implementer code
    such as ``0x41``, or the machine architecture when neither is published.

    Part of the machine's stable identity rather than a capability: it is what keeps an Intel
    machine's learned coefficients from being averaged with an AMD one's in a shared store.

    Deliberately the **raw** identifier, including the ARM implementer's hex code, because this
    value is fingerprint material and translating it would move every ARM machine's key —
    discarding, once, everything the engine had learned on it, in exchange for readability that
    `vendor_display_name` provides at no cost.

    Returns:
        The vendor identifier, or `""` when undetectable.
    """
    raw = _cpuinfo_fields()
    vendor = raw.get("vendor_id") or raw.get("cpu implementer") or ""
    if not vendor:
        # Some ARM parts omit both fields; the machine architecture is the next-best stable
        # discriminator, and it is available on every platform Python runs on.
        return platform.machine()
    return vendor


@functools.lru_cache(maxsize=1)
def cpu_model_name() -> str:
    """The CPU model string, or `""` when undetectable.

    The finest-grained stable identifier a machine offers without special privileges. Used
    only as fingerprint input — no decision branches on the model name itself, because a
    lookup table of model names is exactly the kind of hardware assumption that goes stale.

    Returns:
        The model name, or `""` when undetectable.
    """
    raw = _cpuinfo_fields()
    return raw.get("model name") or raw.get("cpu part") or ""
