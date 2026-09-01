"""Time variants round-robin, so drift cannot be read as a difference between them.

The obvious way to compare `n` settings of one knob is a loop: fix the setting, take `k`
timings, move on. It is also the way to manufacture a result, because anything that drifts
over the life of the process — a fleet warming, a page cache filling, a co-tenant arriving on
a shared cluster — is charged to whichever setting held the floor while it drifted. The
ordering *is* the confound.

This is not hypothetical here. Sweeping an aggregate's reducer count in the fixed order
`[2, 4, 8, 16, 32, 64]`, three timings each, produced a clean interior optimum at 8 across
four cardinalities spanning five orders of magnitude — a strong enough pattern to be written
up, published, and used to derive a mechanism. Re-run round-robin, the same configuration at
100 groups read `r=1` 407 ms, `r=2` 418 ms, `r=8` 487 ms, `r=16` 709 ms: monotone, with the
"optimum" 20% *worse* than its neighbour rather than 5% better. The whole shape was the drift.
`docs/architecture/internals/distributed_scaling_audit.md` records what that cost.

The fix is to give every variant the same distribution of positions in time. One rep of each,
in order, then again — so a linear drift lands on all of them equally and shows up as spread
within a variant rather than as distance between variants.

## What this does not fix

Interleaving removes *ordering* bias, not noise. It cannot help a difference smaller than the
run-to-run spread, which is why `timings` are returned rather than a verdict: a caller that
wants to claim a winner should look at whether the ranges overlap, and this module
deliberately does not make that call for it.

It also cannot help when the variants perturb each other — a setting that leaves a warm cache
the next variant reads, say. Interleaving makes that *worse*, not better, by mixing them; the
guard there is a reset between reps, which belongs to the caller because only the caller knows
what state is shared.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TypeVar

__all__ = ["interleaved", "overlapping"]

V = TypeVar("V")


def interleaved(
    variants: Sequence[V],
    run: Callable[[V], object],
    *,
    rounds: int = 5,
    warmup: bool = True,
) -> dict[V, list[float]]:
    """Time `run(variant)` for every variant, round-robin, `rounds` times each.

    Every variant is warmed once before any timing is taken, so a first-call cost (a fleet
    spawn, a JIT compile, a cold read) is not charged to whichever variant happened to go
    first — which is the same ordering bias one level down.

    Args:
        variants: The settings to compare. Must be hashable; used as the result keys.
        run: Called with one variant, once per warmup and once per round. Its return is
            discarded, so a caller that needs to assert on the result should do it inside.
        rounds: Timed repetitions of each variant. Each round runs every variant once, in
            `variants` order.
        warmup: Run each variant once, untimed, before the rounds. Leave it on unless the
            caller has warmed things itself.

    Returns:
        One list of elapsed milliseconds per variant, in round order.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    timings: dict[V, list[float]] = {v: [] for v in variants}
    if warmup:
        for v in variants:
            run(v)
    for _ in range(rounds):
        for v in variants:
            start = time.perf_counter()
            run(v)
            timings[v].append((time.perf_counter() - start) * 1000.0)
    return timings


def overlapping(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two variants' timing ranges overlap, i.e. the difference is not resolved.

    The honest reading of `min(a) <= max(b) and min(b) <= max(a)` is "these runs do not
    separate these variants" — not "the variants are equal". It exists so a caller states
    that conclusion explicitly rather than reading a rank order off medians that a couple of
    milliseconds of noise would reverse.

    Args:
        a: One variant's timings.
        b: The other's.

    Returns:
        True when the two ranges intersect, so no ordering between them is supported.
    """
    if not a or not b:
        return True
    return min(a) <= max(b) and min(b) <= max(a)
