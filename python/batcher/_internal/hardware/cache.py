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

from batcher._internal.hardware.sysfs import read_int, read_text
from batcher._internal.hardware.topology import affinity_cpu_ids, parse_cpu_list

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


def _usable_cpus() -> list[int]:
    """CPU ids this process may run on, sorted; `[0]` when the mask cannot be read.

    `cpu0` is the fallback rather than "every CPU in `/sys`" because `/sys` is host-wide even
    inside a container: enumerating it would measure cores the process can never be scheduled
    on, which is the failure this whole module is being narrowed to avoid.
    """
    return sorted(affinity_cpu_ids() or {0})


def _cpu_cache_domain(cpu_id: int) -> tuple[dict[str, int], set[int]]:
    """One CPU's data-cache sizes, and the CPUs sharing its last level.

    The sharing set is what lets [`cache_hierarchy`] read one CPU per cache domain instead of
    all of them: every core in it reports the same hierarchy by construction.
    """
    sizes: dict[str, int] = {}
    domain: set[int] = set()
    deepest = 0
    for idx in sorted(glob.glob(f"/sys/devices/system/cpu/cpu{cpu_id}/cache/index*")):
        level = read_int(os.path.join(idx, "level"))
        kind = read_text(os.path.join(idx, "type"))
        if level <= 0 or kind == "Instruction":
            continue
        size = _parse_cache_size(read_text(os.path.join(idx, "size")))
        if size <= 0:
            continue
        # "Unified" at level 1 is rare but real (some ARM cores); treat it as the d-cache,
        # since a unified L1 is the cache a data working set actually contends for.
        key = f"l{level}d" if level == 1 else f"l{level}"
        sizes[key] = max(sizes.get(key, 0), size)
        line = read_int(os.path.join(idx, "coherency_line_size"))
        if line > 0:
            sizes["line"] = max(sizes.get("line", 0), line)
        if level >= deepest:
            deepest = level
            domain = parse_cpu_list(read_text(os.path.join(idx, "shared_cpu_list")))
    return sizes, domain


@functools.lru_cache(maxsize=1)
def cache_hierarchy() -> dict[str, int]:
    """Every cache level's size in bytes, plus the line size, for the caches this process has.

    Reported per *cache domain*, so on a chiplet design this is the cache shared by the cores
    that probe one bucket (per-CCX) rather than the socket total — exactly the residency that
    matters for a hash table probed from every core, and what a socket-wide figure would
    overstate by the chiplet count.

    **Read from the CPUs this process may actually run on, and reported as the binding
    (smallest) domain among them.** Reading `cpu0` alone was wrong three ways, each on hardware
    Batcher is routinely deployed to:

    * a container pinned by cpuset to cores 64-127 still read `cpu0`, because `/sys` is
      host-wide and shows every core whether or not the process can be scheduled on one;
    * an Intel hybrid part (Alder Lake and later) has P-cores and E-cores with different L2,
      and E-cores share an L2 four ways, so which core answered decided the figure;
    * an AMD part with stacked cache (X3D) has CCDs with 96 MiB of L3 beside CCDs with 32 MiB,
      a threefold spread inside one socket.

    In each case the previous answer was not an average or an approximation, it was whichever
    core happened to be numbered zero — and a broadcast threshold sized from the large domain
    spills out of the small one, which is the direction that costs a query rather than a
    ranking. The minimum is taken for the same reason every binding figure in this package is:
    a table that stays resident on the weakest domain stays resident on all of them.

    `line` is the one field taken as a **maximum** instead. It sizes false-sharing padding and
    prefetch distance, where the larger value is the safe one — padding to 64 bytes on a core
    with 128-byte lines still false-shares.

    Cost is one `/sys` read per cache domain, not per core: each CPU's `shared_cpu_list` names
    the cores that answer identically, and they are skipped. A 128-core EPYC reads eight.

    Keys are ``l1d``, ``l2``, ``l3``, and ``line``. Instruction caches are omitted: no
    data-plane decision depends on them. A level this machine does not have, or does not
    report, is omitted rather than zeroed.

    Returns:
        Cache level name to size in bytes, plus ``line``.
    """
    usable = _usable_cpus()
    remaining = set(usable)
    out: dict[str, int] = {}
    for cpu_id in usable:
        if cpu_id not in remaining:
            continue  # a core in a domain already measured reports the same hierarchy
        sizes, domain = _cpu_cache_domain(cpu_id)
        remaining -= (domain & set(usable)) or {cpu_id}
        for key, size in sizes.items():
            if key == "line":
                out[key] = max(out.get(key, 0), size)
            else:
                out[key] = min(out[key], size) if key in out else size
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
