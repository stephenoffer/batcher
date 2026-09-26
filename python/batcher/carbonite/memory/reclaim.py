"""Handing the allocator's arena back when a query is about to go out of core.

The data plane allocates every morsel through mimalloc, which **retains freed pages by
design** — that retention is why the engine scales across cores at all, because glibc's malloc
returns buffers of that size with `munmap`, and each `munmap` broadcasts a TLB-shootdown
interrupt to every core. The cost is that the process's resident set keeps counting memory the
engine has already finished with, and the amount is not small: three 8M-row Parquet group-bys
whose results were dropped left a 1,397 MiB resident set of which 1,289 MiB was the engine's
arena, and one forced trim handed **408 MiB** of it back.

`_internal.hardware.engine.allocator.release_retained_memory` has been able to do that for a
while, and its own docstring says it "is the thing to try first when an envelope is about to
force a spill". Nothing called it — the only callers were its own tests — so the purge-delay
constant's justification, that the retention is safe *because* Carbonite pulls the valve under
pressure, was not true. This module is that call.

It is pulled from the **executor**, once a query has committed to the out-of-core path
(`api.orchestration.stages.spill_to_disk` via `ResourceManager.going_out_of_core`), rather than
from the gate that decides. The decision has three independent routes — admission's
counter-offer, the plan's estimated peak, and the resident size of the input — and only the
middle one reads live pressure, so a trim hung off that reading covered one spill in three and
missed the estimate, which is the ordinary way a large query goes out of core.

**It does not try to avoid the spill, and that is deliberate.** By the time it runs the decision
is made. Even taken earlier, the obvious design — trim, then re-read the pressure and maybe stay
in memory — does not survive contact with how the level is computed.
`PressureMonitor._engine_used_fraction` is the *maximum* of two buffer-pool
utilizations and the process footprint, and a trim moves only the footprint: a level driven by
reservation accounting cannot come down no matter how much arena is returned. What is left is
then smoothed by a de-escalation EWMA whose whole purpose is to *not* fall on one good reading.
So re-reading would pay a forced walk of every heap for an answer that mostly cannot change.

What the trim is worth is the 408 MiB itself. A query that is spilling is a query on a box
where memory is scarce, and holding a third of a gigabyte that nothing will use is a third of a
gigabyte closer to an OOM kill — fatal on a swapless node, which is the default on Kubernetes.
The unmapping it costs is tens of milliseconds against a spill measured in seconds.

Result-invariant. It changes the process's resident set, never what a query computes and never
whether it spills.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from batcher._internal.logging import get_logger, log_kv, note_suppressed

__all__ = [
    "RECLAIM_COOLDOWN_MAX_S",
    "RECLAIM_COOLDOWN_S",
    "RECLAIM_WORTHWHILE_BYTES",
    "RETAINED_CEILING_FRACTION",
    "reclaim_before_spill",
    "reclaim_if_retaining",
    "reclaim_stats",
    "reset_reclaim_state",
]

#: How long to wait between attempts before one has failed. Short, because what it guards
#: against is a *decision path* calling this per query rather than a hot loop, and a query that
#: has reached this branch is spilling — writing state to disk for as long as it takes, which
#: is orders of magnitude more than the trim.
RECLAIM_COOLDOWN_S = 5.0

#: Ceiling on the doubling. Past this the process has proved several times over that its arena
#: is live, and the trim is pure cost; a minute is long enough that a workload which has moved
#: on to a different shape gets another chance without the loop noticing.
RECLAIM_COOLDOWN_MAX_S = 60.0

#: Below this a release did not pay for the unmapping it cost. Sized as "a morsel's worth":
#: freeing less than one batch of memory is not worth a forced walk of every heap, so counting
#: it as a success would keep re-trying a trim the process has nothing left to give.
RECLAIM_WORTHWHILE_BYTES = 16 * 1024 * 1024

#: Share of the memory envelope the process may hold resident between queries before the
#: allocator is asked for its retained pages back (`reclaim_if_retaining`).
#:
#: The engine keeps freed pages on purpose (`bc-py/src/hardware.rs`, `PURGE_DELAY_MS`): a
#: 13-column, 9M-row join re-runs ~35% faster when its output buffers land on pages it already
#: has. The premise was that consecutive queries *reuse* those regions, and on repeats of one
#: large join they do not -- measured on H2O join q1 run back to back, the resident set climbed
#: by roughly one result per run, 6.5 -> 8.8 -> 10.4 -> 11.6 GiB with the Python side holding
#: nothing, and the benchmark harness drove it past 40 GiB before the run was stopped, on a
#: 60 GiB envelope. The ceiling is on the whole process's resident set -- mimalloc's own `commit`
#: figure reports its arena reservation (a flat 32 GiB through that run), so it cannot see the
#: retention -- which means other libraries' memory counts against it too. At a quarter, a
#: benchmark process holding every engine's copy of the data sat over the ceiling permanently
#: and so purged on every query; at two fifths (24 GiB of 60) the ceiling still leaves room for
#: one more large query's working set below the envelope, which is what it protects.
RETAINED_CEILING_FRACTION = 0.4

#: Once over the ceiling, the allocator purges immediately until the resident set falls below
#: this share of the ceiling, then goes back to retaining. The gap is the hysteresis that keeps
#: one query at the boundary from toggling the mode on every run.
RETAINED_RESTORE_FRACTION = 0.5


@dataclass
class _ReclaimState:
    """What the last attempts achieved, so the next one can decide whether to bother.

    Attributes:
        next_attempt_s: Monotonic time before which no attempt is made.
        cooldown_s: The current wait, doubled on an empty release and reset on a paying one.
        purging: Whether the allocator is in immediate-purge mode, over the ceiling.
        attempts: Attempts made in this process.
        released_bytes: Bytes the allocator reported handing back, in total.
    """

    next_attempt_s: float = 0.0
    cooldown_s: float = RECLAIM_COOLDOWN_S
    purging: bool = False
    attempts: int = 0
    released_bytes: int = 0


_STATE = _ReclaimState()


def reclaim_before_spill() -> int:
    """Hand the allocator's retained pages back, at the point a query goes out of core.

    Called when a spill is being *taken*, not to avoid it — see the module docstring for why
    re-reading the level would pay for an answer that mostly cannot change. Never call it from
    a hot path.

    Self-limiting. An attempt is skipped inside the cooldown, and the cooldown doubles each time
    a release comes back under `RECLAIM_WORTHWHILE_BYTES` — so a process whose arena is genuinely
    all live stops paying for the trim, while one that has just finished a large stage keeps
    getting it.

    The measured shape it is sized against: three 8M-row Parquet group-bys whose results were
    dropped left a 1,397 MiB resident set of which 1,289 MiB was the engine's own arena, and one
    forced trim handed 408 MiB of it back. A second trim immediately after freed nothing, which
    is the case the cooldown exists to stop repeating.

    Deliberately unlocked. Two queries deciding to spill at the same instant can both pass the
    cooldown check and both trim; the second finds nothing, which doubles the cooldown and
    costs one extra pass over the heaps. A lock here would serialize two threads on a syscall
    to avoid an outcome the backoff already absorbs.

    Returns:
        Bytes of resident memory the allocator reported releasing, `0` when the attempt was
        skipped, freed nothing, or the engine cannot report. The caller does not act on it —
        the figure exists so the backoff and the diagnostic can. Never raises: a failed trim
        must leave the caller free to take the spill it was already going to take.
    """
    return _attempt_reclaim("released retained allocator memory before spilling")


def reclaim_if_retaining(envelope_bytes: int) -> int:
    """Hold the process's resident set near its share of `envelope_bytes`, as a query finishes.

    The spill path never covers this: a process whose queries all fit in memory never spills,
    so before this nothing bounded the pages the allocator kept between them. Over the ceiling
    (`RETAINED_CEILING_FRACTION`) the allocator is switched to purging freed regions at once
    and a forced release is attempted; once the resident set is back under
    `RETAINED_RESTORE_FRACTION` of the ceiling, the fast retention is restored. The purge
    delay is the lever that works: a forced collect from this thread reached 0.5 GB of the
    ~15 GB a run of large joins had retained.

    The check is one RSS read. The release goes through the same backoff as
    `reclaim_before_spill`, so a process whose memory is genuinely live stops paying for it.
    Result-invariant, and never raises.

    Args:
        envelope_bytes: The memory envelope the ceiling is a fraction of. Zero or negative
            disables the check.

    Returns:
        Bytes released, `0` when under the ceiling, inside the cooldown, or unreadable.
    """
    from batcher._internal.hardware.engine.allocator import set_purge_delay
    from batcher.carbonite.memory.probe import process_rss_bytes

    rss = process_rss_bytes()
    if envelope_bytes <= 0 or rss is None:
        return 0
    ceiling = envelope_bytes * RETAINED_CEILING_FRACTION
    if rss > ceiling:
        if not _STATE.purging:
            _STATE.purging = set_purge_delay(0)
        return _attempt_reclaim("released retained allocator memory over the retention ceiling")
    if _STATE.purging and rss < ceiling * RETAINED_RESTORE_FRACTION:
        set_purge_delay(-1)
        _STATE.purging = False
    return 0


def _attempt_reclaim(message: str) -> int:
    """One forced release, gated and backed off by `_STATE`; the shared body of both triggers."""
    now = time.monotonic()
    if now < _STATE.next_attempt_s:
        return 0
    try:
        from batcher._internal.hardware.engine.allocator import release_retained_memory

        # `force`, without which this reaches nothing. A plain collect walks only the calling
        # thread's heap, and the engine allocates its operator state on rayon workers — so a
        # collect from the control plane's thread frees essentially none of what it came for.
        # Measured on three Parquet group-bys: 0 MiB unforced against 408 MiB forced, of a
        # 1,397 MiB resident set. The expense of the forced walk is what the cooldown is for.
        released = release_retained_memory(force=True)
    except Exception as exc:  # a trim must never fail a query
        note_suppressed("carbonite", "release the allocator's retained memory", exc)
        _STATE.next_attempt_s = now + _STATE.cooldown_s
        return 0
    _STATE.attempts += 1
    _STATE.released_bytes += released
    if released >= RECLAIM_WORTHWHILE_BYTES:
        _STATE.cooldown_s = RECLAIM_COOLDOWN_S
        log_kv(
            get_logger("carbonite.memory"),
            # Debug rather than info: this is a routine trade the engine makes on its own, and
            # the reader who wants it is diagnosing a spill that did or did not happen.
            10,
            message,
            released_bytes=released,
        )
    else:
        _STATE.cooldown_s = min(RECLAIM_COOLDOWN_MAX_S, _STATE.cooldown_s * 2)
    _STATE.next_attempt_s = now + _STATE.cooldown_s
    return released


def reclaim_stats() -> dict[str, int | float]:
    """What the trim has achieved in this process, for `ResourceManager.stats`.

    Returns:
        `attempts`, `released_bytes`, and the current `cooldown_s`. A rising attempt count with
        no released bytes is the signature of a process whose arena is genuinely live, which is
        what separates "the engine is holding memory it does not need" from "the box is full".
    """
    return {
        "attempts": _STATE.attempts,
        "released_bytes": _STATE.released_bytes,
        "cooldown_s": _STATE.cooldown_s,
    }


def reset_reclaim_state() -> None:
    """Forget the cooldown and the counters, so the next call attempts a release.

    For tests, which otherwise inherit whatever backoff an earlier one left behind — the same
    reason `probe.reset_memory_sampling` exists.
    """
    global _STATE
    _STATE = _ReclaimState()
