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
_ARM_WIDTHS: tuple[tuple[str, int], ...] = (
    ("asimd", 128),
    ("neon", 128),
    ("sve", 256),  # SVE is width-agnostic; 256 is the common server implementation
    ("sve2", 256),
)


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


def simd_width_bits() -> int:
    """The widest SIMD register this CPU offers, in bits — `128` when undetectable.

    The multiplier on per-row kernel throughput and the number a JIT payoff estimate scales
    with. Falls back to 128 rather than 0 because every 64-bit target Batcher supports has at
    least SSE2 or NEON, so 128 is a floor rather than a guess.

    Returns:
        The widest available vector width in bits, at least 128.
    """
    present = cpu_features()
    widths = [bits for name, bits in (*_X86_WIDTHS, *_ARM_WIDTHS) if name in present]
    return max(widths) if widths else 128


@functools.lru_cache(maxsize=1)
def cpu_vendor() -> str:
    """The CPU vendor string (``GenuineIntel``, ``AuthenticAMD``, ``ARM``, ...), or `""`.

    Part of the machine's stable identity rather than a capability: it is what keeps an Intel
    machine's learned coefficients from being averaged with an AMD one's in a shared store.

    Returns:
        The vendor identifier, or `""` when undetectable.
    """
    raw = _cpuinfo_fields()
    vendor = raw.get("vendor_id") or raw.get("cpu implementer") or ""
    if not vendor:
        # ARM parts often omit vendor_id entirely; the machine architecture is the next-best
        # stable discriminator, and it is available on every platform Python runs on.
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
