"""The machine-shaped multipliers the per-operator cost forms fold in.

`model.py` says what work an operator does in rows and bytes. This module says what those
rows and bytes cost *on a machine*: whether the hash table it builds fits in cache, how
many bytes overflow the memory budget onto disk, how many merge passes an out-of-core sort
then needs, and how many comparisons a full sort or a top-N really performs.

They live apart from the operator forms because they answer a different question — what the
hardware is, rather than what the plan does — and because every one of them is consumed by
more than one operator. A `Join`, an `Aggregate`, a `Distinct`, and a distinct `Union` all
build a hash table and all pay the same cache and spill terms; stating them once is what
keeps those four operators from drifting apart.
"""

from __future__ import annotations

import math

from batcher._internal.hardware import l3_cache_bytes
from batcher.config import active_config
from batcher.kyber.storage_cost import spill_device_factor

__all__ = [
    "cache_factor",
    "memory_budget",
    "merge_passes",
    "sort_comparisons",
    "spill_io",
]

# How much more a random hash-table access costs once the table no longer fits in the last
# level of cache, per octave of overflow.
#
# A cost model that charges one `hash_probe_row` per probe regardless of the table's size
# says a 1,000-row build side and a 100-million-row build side cost the same to probe. They
# do not: the first lives in L1 and the second takes a DRAM round trip on essentially every
# probe, which is one to two orders of magnitude slower. That difference is precisely what
# join *ordering* is choosing between — whether to build the small side or the large one, and
# whether an intermediate should be materialized at all — so leaving it out of the model
# leaves the enumerator ranking plans by a quantity that ignores the dominant term.
#
# The penalty is charged per doubling of `build_bytes / cache_bytes` and capped, which
# reproduces the measured shape: flat while resident, a steep knee at the cache boundary, then
# a plateau once every access already misses and there is nothing left to lose.
_CACHE_MISS_PENALTY_PER_OCTAVE = 0.35
_CACHE_MISS_MAX_FACTOR = 8.0

# Sequential bytes moved per unit of `io` cost relative to a row of CPU work. Spilling is
# charged in the same units as everything else so the axes stay comparable; the constant only
# has to place a spilled byte on the same scale as a row of compute.
_SPILL_WRITE_READ_PASSES = 2.0

# Buffer reserved per input run during an external merge. The merge fan-in is the memory
# budget divided by this, so it is what decides whether an over-budget sort needs one merge
# pass or several.
_EXTERNAL_MERGE_RUN_BUFFER_BYTES = 1 << 20


def memory_budget() -> float:
    """The working-set budget an operator has before it must spill, in bytes.

    The same static envelope the data plane is given, so the cost model's idea of "this
    will spill" is the engine's. `0` means the user opted out of bounded memory entirely,
    in which case nothing spills and the spill terms vanish.

    Returns:
        The per-operator spill budget in bytes.
    """
    return float(active_config().spill_budget_bytes())


def cache_factor(state_bytes: float) -> float:
    """The per-access slowdown of a hash table of `state_bytes`, from cache residency.

    `1.0` while the table fits in the last-level cache, then growing by a fixed penalty per
    doubling beyond it and flattening at a ceiling — the shape a random-access probe
    actually has. With no detectable cache size the factor is 1.0.

    Args:
        state_bytes: Resident size of the hash table being probed.

    Returns:
        A multiplier on the per-probe cost, at least 1.0.
    """
    cache = float(l3_cache_bytes())
    if cache <= 0.0 or state_bytes <= cache:
        return 1.0
    octaves = math.log2(state_bytes / cache)
    return min(_CACHE_MISS_MAX_FACTOR, 1.0 + _CACHE_MISS_PENALTY_PER_OCTAVE * octaves)


def spill_io(state_bytes: float) -> float:
    """Bytes of spill IO an operator whose state is `state_bytes` will move.

    Zero while the state fits the memory budget. Past it, everything that does not fit is
    written once and read back once. Charging nothing for it makes a plan that spills look
    exactly as cheap as one that does not, which is the single largest cost error a plan can
    contain — a spilled operator is disk-bound, and the optimizer's whole reason to prefer a
    smaller build side is to avoid it.

    Scaled by the measured class of the spill device, so the same overflow is costed as the
    cheap thing it is on local flash and the expensive thing it is on a network volume.

    Args:
        state_bytes: The operator's resident state size.

    Returns:
        Spill bytes on the `io` axis, `0.0` when the state fits.
    """
    budget = memory_budget()
    if budget <= 0.0 or state_bytes <= budget:
        return 0.0
    return _SPILL_WRITE_READ_PASSES * (state_bytes - budget) * spill_device_factor()


def merge_passes(state_bytes: float) -> float:
    """External-merge passes an out-of-core sort of `state_bytes` needs.

    A sort that does not fit runs `ceil(log_F(state/budget))` merge passes, each of which
    rewrites the whole run — so its IO grows with the *logarithm* of the overflow, not
    linearly. `F` (the merge fan-in) is the budget divided by one run buffer, and is large
    enough in practice that a single pass covers almost everything; the formula is here so
    that the one case where it does not (a sort many times the budget) is not costed as if
    it were.

    Args:
        state_bytes: The sort's total state size.

    Returns:
        The number of merge passes, `0.0` when the sort fits in memory.
    """
    budget = memory_budget()
    if budget <= 0.0 or state_bytes <= budget:
        return 0.0
    runs = state_bytes / budget
    fan_in = max(2.0, budget / _EXTERNAL_MERGE_RUN_BUFFER_BYTES)
    return max(1.0, math.ceil(math.log(runs, fan_in)))


def merge_io(state_bytes: float) -> float:
    """Bytes an out-of-core sort of `state_bytes` moves, across all its merge passes.

    A sort rewrites its runs once per merge pass, so unlike a hash operator's one-shot
    overflow (`spill_io`) its IO is the *whole* state times the pass count.

    Args:
        state_bytes: The sort's total state size.

    Returns:
        Spill bytes on the `io` axis, `0.0` when the sort fits in memory.
    """
    return (
        _SPILL_WRITE_READ_PASSES * state_bytes * merge_passes(state_bytes) * spill_device_factor()
    )


def sort_comparisons(n: float, heap: float) -> float:
    """Comparisons a sort of `n` rows keeping `heap` of them performs.

    A **full** sort is the textbook `n·log2(n)`.

    A **top-N** (a fused `Sort` + `Limit`) is not `n·log2(k)`, which is what charging every
    row a heap sift-down assumes. Every row is compared once against the heap's root, but only
    a row that beats it is inserted — and over a randomly ordered input the expected number of
    such rows is `k·(1 + ln(n/k))`, because the `i`-th row displaces the root only if it lands
    in the running top `k`, with probability `min(1, k/i)`. So the real cost is `n` root
    comparisons plus `k·(1 + ln(n/k))` sift-downs of `log2(k)` each.

    The difference is the whole reason to fuse a limit into a sort: for `n = 10^8, k = 10`,
    `n·log2(k)` charges 3.3x the input while the true cost is a little over one pass. Costing
    top-N as a discounted full sort made the optimizer nearly indifferent to the fusion, and
    over-charged a `LIMIT 10` over a large scan by more than three times.

    Args:
        n: Rows entering the sort.
        heap: Rows the sort retains (`n` for a full sort, the limit for a top-N).

    Returns:
        The estimated comparison count.
    """
    if heap >= n:
        return n * math.log2(max(2.0, n))
    insertions = heap * (1.0 + math.log(n / heap))
    return n + insertions * math.log2(max(2.0, heap))
