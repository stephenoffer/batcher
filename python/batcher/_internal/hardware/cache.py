"""The CPU cache hierarchy this process runs on — the sizes every blocking decision needs.

Cache size is the physical quantity behind a whole family of thresholds that are usually
written as constants: how large a hash table may get before a broadcast join stops paying,
how many rows a morsel should carry so an operator's working set stays resident, how wide a
partition fan-out can go before the write buffers evict each other. A constant tuned on one
machine is wrong on the next by the ratio of the two caches, and that ratio spans an order of
magnitude across the hardware Batcher runs on: ~512 KiB of L2 and 1 MiB of L3 on a small ARM
core against 2 MiB of L2 and 32+ MiB of L3 per CCX on an EPYC.

Linux-only (`/sys/devices/system/cpu`); every probe reports `0` elsewhere so callers keep
whatever default they had.
"""

from __future__ import annotations

import functools
import glob
import os

__all__ = [
    "cache_hierarchy",
    "l3_cache_bytes",
]


def _parse_cache_size(raw: str) -> int:
    """Parse a `/sys` cache size like ``"16384K"`` / ``"32M"`` / ``"1G"`` into bytes."""
    raw = raw.strip()
    if not raw:
        return 0
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    mult = units.get(raw[-1].upper(), 1)
    try:
        return int(raw[:-1] if mult > 1 else raw) * mult
    except ValueError:
        return 0


def _read_int(path: str) -> int:
    """An integer from a `/sys` file, or `0` when absent or unparseable."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def _read_str(path: str) -> str:
    """A trimmed string from a `/sys` file, or `""` when absent."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


@functools.lru_cache(maxsize=1)
def cache_hierarchy() -> dict[str, int]:
    """Every cache level's size in bytes, plus the line size, for this core's cache domain.

    Reads ``cpu0``'s own hierarchy, so on a chiplet design it reports the caches *shared by
    the cores that probe one bucket* (per-CCX) rather than the socket total — which is exactly
    the residency that matters for a hash table probed from every core, and is what a
    socket-wide figure would overstate by the chiplet count.

    Keys are ``l1d``, ``l2``, ``l3``, and ``line`` (the cache-line size, the granularity every
    false-sharing and prefetch decision is expressed in). Instruction caches are omitted: no
    data-plane decision depends on them. A level this machine does not have, or does not
    report, is omitted rather than zeroed.

    Returns:
        Cache level name to size in bytes, plus ``line``.
    """
    out: dict[str, int] = {}
    for idx in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        level = _read_int(os.path.join(idx, "level"))
        kind = _read_str(os.path.join(idx, "type"))
        if level <= 0 or kind == "Instruction":
            continue
        size = _parse_cache_size(_read_str(os.path.join(idx, "size")))
        if size <= 0:
            continue
        # "Unified" at level 1 is rare but real (some ARM cores); treat it as the d-cache,
        # since a unified L1 is the cache a data working set actually contends for.
        key = f"l{level}d" if level == 1 else f"l{level}"
        out[key] = max(out.get(key, 0), size)
        line = _read_int(os.path.join(idx, "coherency_line_size"))
        if line > 0:
            out["line"] = max(out.get("line", 0), line)
    return out


def l3_cache_bytes() -> int:
    """Bytes of last-level (L3) cache in this core's cache domain, or `0` if undetectable.

    The physical quantity a broadcast-join threshold actually depends on: a broadcast builds
    one hash table probed from every core, so the strategy wins only while that table stays
    L3-resident. A fixed byte threshold is therefore wrong by the ratio of the real cache to
    the assumed one — ~1 MiB on a small ARM core to 32+ MiB per CCX on an EPYC, an 8x spread
    in both directions. Reading it lets the optimizer size the threshold to the machine.

    Returns:
        L3 cache size in bytes, or `0` when the platform cannot report it.
    """
    return cache_hierarchy().get("l3", 0)
