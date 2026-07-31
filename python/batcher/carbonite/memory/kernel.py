"""The kernel's own view of how close this process is to being OOM-killed.

`probe` answers *how much memory is there* and *how much is in use*. This answers a different
and strictly more urgent question: **is the kernel already fighting to keep this cgroup
alive?** Those diverge, and the gap is exactly where a container dies without warning.

Four signals live here, none of which anything else in the engine read:

* **`memory.high`.** The cgroup v2 *throttle* threshold. Past it the kernel does not kill —
  it puts every allocating task to sleep in direct reclaim, so the workload keeps running at
  a fraction of its rate with no error anywhere. Kubernetes memory QoS sets `memory.high`
  well below `memory.max`, so a pod tuned against `memory.max` alone budgets for a ceiling
  it will never be allowed to reach. Reading only `memory.max` — which is what every sizing
  decision did — plans a query against memory the kernel has already decided to claw back.
* **`memory.events`.** Monotonic counters for how many times this cgroup hit `high` (was
  throttled), hit `max` (allocation failed), and was `oom_kill`ed. A non-zero `oom_kill` is
  not a prediction: it is proof this container has *already* been killed at this size. It is
  the single most actionable memory fact available, and nothing was reading it.
* **PSI `full`.** The share of a window in which **every** runnable task in the cgroup was
  stalled on memory. `some` (which `_internal.hardware.cgroup_pressure` already reports) is
  noisy — a healthy process faulting pages in registers `some` constantly. `full` means the
  cgroup made no progress at all, and it climbs for seconds before a kill.
* **Swap.** A cgroup with swap headroom degrades to slow; one at `memory.swap.max` has no
  soft landing left and the next reclaim failure is a kill.

Everything degrades to "not reported" rather than raising, and every reading is `None` off
Linux or outside a cgroup. A caller that gets `None` keeps whatever it was already doing, so
the whole module is inert on a bare-metal or macOS host.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from batcher._internal.hardware.cgroup import (
    cgroup_v2_dirs,
    read_cgroup_bytes,
    read_cgroup_stat,
    read_psi,
)

__all__ = [
    "STALL_CRITICAL",
    "STALL_ELEVATED",
    "KernelMemoryState",
    "cgroup_high_bytes",
    "kernel_memory_state",
    "kernel_stats",
    "memory_stall_full",
    "oom_kill_count",
    "reset_kernel_sampling",
    "swap_headroom_bytes",
    "throttle_limit_bytes",
]

#: PSI `full` share above which the cgroup is losing meaningful time to reclaim. Below this a
#: memory-tight workload is merely working; above it the kernel is spending the process's own
#: time recovering pages, so shrinking the working set buys back throughput as well as safety.
STALL_ELEVATED = 0.10

#: PSI `full` share at which the cgroup is thrashing. Sustained readings here precede an OOM
#: kill by seconds, which is the whole reason to watch `full` rather than `memory.current`:
#: by the time the charge is at the limit the kernel is already deciding whom to kill.
STALL_CRITICAL = 0.40

#: How long one kernel snapshot is reused. These files are read wherever a pressure level is
#: wanted, and a snapshot costs four `open`+parse round trips per cgroup level. The underlying
#: values cannot move meaningfully inside the window: PSI is a 10-second rolling average and
#: the event counters are monotonic, so a 50 ms sample oversamples both by orders of magnitude.
#: Deliberately the same window `probe.SAMPLE_TTL_SECONDS` uses, so a decision that reads both
#: sees one consistent moment rather than two readings taken a syscall apart.
SAMPLE_TTL_SECONDS = 0.05

_state_cache: tuple[float, KernelMemoryState] | None = None


@dataclass(frozen=True, slots=True)
class KernelMemoryState:
    """One reading of what the kernel thinks of this cgroup's memory.

    Every field is `None` when the kernel did not report it, which is distinct from zero: a
    cgroup with no swap configured reports `0` for `swap_current_bytes`, while a host with no
    cgroups at all reports `None`, and a caller must not treat the second as "no swap in use".

    Attributes:
        limit_bytes: The hard cap (`memory.max`), past which an allocation fails.
        high_bytes: The throttle threshold (`memory.high`), past which allocating tasks are
            put to sleep in direct reclaim instead of failing.
        current_bytes: The cgroup's total charge (`memory.current`), cache included.
        anon_bytes: Anonymous memory — the part reclaim cannot give back without swap.
        file_bytes: Page cache charged here, which the kernel drops before killing anything.
        slab_bytes: Kernel slab charged here (reclaimable and unreclaimable together).
        swap_current_bytes: Bytes of this cgroup's memory currently swapped out.
        swap_limit_bytes: The cgroup's swap cap (`memory.swap.max`).
        oom_kills: Times a task in this cgroup was OOM-killed, since the cgroup was created.
        throttle_events: Times the charge exceeded `memory.high` and tasks were throttled.
        max_events: Times an allocation hit `memory.max` and had to reclaim or fail.
        stall_some: PSI share where at least one task stalled on memory (10 s average).
        stall_full: PSI share where *every* runnable task stalled on memory (10 s average).
    """

    limit_bytes: int | None = None
    high_bytes: int | None = None
    current_bytes: int | None = None
    anon_bytes: int | None = None
    file_bytes: int | None = None
    slab_bytes: int | None = None
    swap_current_bytes: int | None = None
    swap_limit_bytes: int | None = None
    oom_kills: int | None = None
    throttle_events: int | None = None
    max_events: int | None = None
    stall_some: float | None = None
    stall_full: float | None = None

    @property
    def unreclaimable_bytes(self) -> int | None:
        """Charge the kernel cannot recover without swapping or killing something.

        `memory.current` minus the page cache. This is the figure the OOM killer effectively
        acts on, and it is routinely half of `memory.current` on a box that has merely read
        files — which is why sizing against the raw charge reports a machine as nearly dead
        when it is idle.
        """
        if self.current_bytes is None:
            return None
        return max(0, self.current_bytes - (self.file_bytes or 0))

    @property
    def effective_limit_bytes(self) -> int | None:
        """The ceiling that actually binds: the lower of `memory.high` and `memory.max`.

        `memory.high` is not advisory. Past it every allocating task in the cgroup enters
        direct reclaim, so the workload's *effective* ceiling is the throttle point, not the
        kill point — a query planned to sit between the two runs at a fraction of its rate
        for its whole duration while every counter reports success.
        """
        limits = [v for v in (self.high_bytes, self.limit_bytes) if v is not None and v > 0]
        return min(limits) if limits else None

    @property
    def headroom_bytes(self) -> int | None:
        """Bytes still allocable before the *effective* ceiling, from unreclaimable charge.

        `None` when either term is unknown. Never negative: a cgroup already past its throttle
        point has zero headroom, not a debt, and a negative figure would flow into a
        subtraction somewhere and quietly grant memory back.
        """
        ceiling = self.effective_limit_bytes
        used = self.unreclaimable_bytes
        if ceiling is None or used is None:
            return None
        return max(0, ceiling - used)

    @property
    def swap_headroom_bytes(self) -> int | None:
        """Bytes of swap left before the cgroup's swap cap, or `None` when swap is unlimited.

        Swap is the soft landing under an over-committed query: with headroom the workload
        degrades to slow, without it the next failed reclaim is a kill. `None` means either
        no cap (headroom is the device's, not the cgroup's) or nothing reported.
        """
        if self.swap_limit_bytes is None or self.swap_current_bytes is None:
            return None
        return max(0, self.swap_limit_bytes - self.swap_current_bytes)

    @property
    def was_oom_killed(self) -> bool:
        """Whether a task in this cgroup has already been OOM-killed.

        Proof rather than prediction. A workload that resumes into a cgroup carrying a
        non-zero count is running at a size that has demonstrably not fit, and the correct
        response is to plan smaller *before* the first breaker rather than after the second
        kill.
        """
        return bool(self.oom_kills)

    @property
    def thrashing(self) -> bool:
        """Whether the cgroup is losing whole windows to reclaim (PSI `full` past critical)."""
        return self.stall_full is not None and self.stall_full >= STALL_CRITICAL

    def as_dict(self) -> dict[str, float | int | bool]:
        """The reported fields plus the derived verdicts, for telemetry and `explain`.

        Fields the kernel did not report are omitted rather than sent as zero, so a reader
        can tell "no swap in use" from "this host has no cgroups".

        Returns:
            A flat mapping of every non-`None` reading and derivation.
        """
        out: dict[str, float | int | bool] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        for name in ("unreclaimable_bytes", "effective_limit_bytes", "headroom_bytes"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.oom_kills is not None:
            out["was_oom_killed"] = self.was_oom_killed
        if self.stall_full is not None:
            out["thrashing"] = self.thrashing
        return out


def reset_kernel_sampling() -> None:
    """Drop the cached snapshot so the next read re-opens the cgroup files.

    For tests that fake `/sys/fs/cgroup`, which otherwise observe whichever reading the first
    test in the process happened to take.
    """
    global _state_cache
    _state_cache = None


def kernel_memory_state() -> KernelMemoryState:
    """One snapshot of the kernel's memory verdict for this cgroup, TTL-sampled.

    Returns:
        The state, with every unreported field left as `None`. An all-`None` state is the
        correct and expected answer off Linux, outside a cgroup, and on cgroup v1 hosts that
        publish neither PSI nor `memory.events`.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.memory.kernel import kernel_memory_state
            >>> state = kernel_memory_state()
            >>> state.was_oom_killed in (True, False)
            True
    """
    global _state_cache
    now = time.monotonic()
    cached = _state_cache
    if cached is not None and now < cached[0]:
        return cached[1]
    state = _read_kernel_state()
    _state_cache = (now + SAMPLE_TTL_SECONDS, state)
    return state


def _read_kernel_state() -> KernelMemoryState:
    """The uncached read behind `kernel_memory_state`."""
    stat = _leafmost_stat()
    events = _leafmost_events()
    psi = _memory_psi()
    return KernelMemoryState(
        limit_bytes=_tightest("memory.max"),
        high_bytes=cgroup_high_bytes(),
        current_bytes=_leafmost_bytes("memory.current"),
        anon_bytes=stat.get("anon"),
        file_bytes=stat.get("file"),
        slab_bytes=stat.get("slab"),
        swap_current_bytes=_leafmost_bytes("memory.swap.current"),
        swap_limit_bytes=_tightest("memory.swap.max"),
        oom_kills=events.get("oom_kill"),
        throttle_events=events.get("high"),
        max_events=events.get("max"),
        stall_some=psi.get("some_avg10"),
        stall_full=psi.get("full_avg10"),
    )


def kernel_stats() -> dict[str, object]:
    """The kernel snapshot as a telemetry block, or `{}` when the kernel published nothing.

    Shaped for splatting into a stats mapping, and empty rather than a dict of `None`s off
    Linux — a reader must be able to tell "this host has no cgroups" from "this cgroup is
    idle", and a block of nulls says neither.

    Returns:
        `{"kernel": {...}}`, or an empty mapping.
    """
    state = kernel_memory_state().as_dict()
    return {"kernel": state} if state else {}


def cgroup_high_bytes() -> int | None:
    """The tightest `memory.high` binding this process, or `None` when none is set.

    Like every other cgroup v2 limit this can be set at any level of the hierarchy — the
    process's own leaf, a parent slice, or the mount root — and the kernel enforces all of
    them, so the effective threshold is the smallest anywhere in the ancestry.

    Returns:
        The throttle threshold in bytes, or `None` when unlimited, unset, or not on cgroup v2.
    """
    return _tightest("memory.high")


def throttle_limit_bytes() -> int | None:
    """The ceiling this process is actually allowed to reach: `min(memory.high, memory.max)`.

    The figure every memory budget should be sized against, and the one nothing read. See
    `KernelMemoryState.effective_limit_bytes`.

    Returns:
        The binding ceiling in bytes, or `None` when neither limit is set.
    """
    return kernel_memory_state().effective_limit_bytes


def oom_kill_count() -> int | None:
    """Times a task in this cgroup has been OOM-killed, or `None` when unreported.

    Returns:
        The monotonic count from `memory.events`, `None` off cgroup v2.
    """
    return kernel_memory_state().oom_kills


def swap_headroom_bytes() -> int | None:
    """Bytes of swap left under this cgroup's swap cap, or `None` when uncapped/unreported."""
    return kernel_memory_state().swap_headroom_bytes


def memory_stall_full() -> float | None:
    """The share of the last 10 s in which *every* task in the cgroup stalled on memory.

    The early warning `memory.current` cannot give: it rises for seconds before a kill, while
    the charge is pinned at the limit the whole time by the reclaim that is defending it.

    Returns:
        A fraction in [0, 1], or `None` when PSI is unavailable.
    """
    return kernel_memory_state().stall_full


def _tightest(name: str) -> int | None:
    """The smallest value of a byte-valued cgroup v2 file across the whole ancestry."""
    values = [
        v for d in cgroup_v2_dirs() if (v := read_cgroup_bytes(os.path.join(d, name))) is not None
    ]
    return min(values) if values else None


def _usage_dirs() -> tuple[str, ...]:
    """Cgroups to read a *usage* or *event* figure from, nearest-first.

    Usage is read nearest-first rather than min'd the way a limit is: a limit binds from
    anywhere in the ancestry, but a parent slice's *charge* folds in every sibling container's
    memory, which this process neither caused nor can release.

    The mount root is used **only** when this process has no sub-path of its own — which is
    exactly the cgroup-*namespace* case, where a pod's own cgroup is mapped to the root and the
    root's figures really are the pod's. Falling back to it otherwise reads the whole machine:
    measured on the box this was developed on, the root's `memory.events` reported two
    `oom_kill`s from unrelated tenants, which would have shrunk this engine's envelope by 20%
    for a failure it had nothing to do with. The same distinction PSI needs, for the same
    reason — see `_memory_psi`.
    """
    own = _own_cgroup_dirs()
    return own if own else (cgroup_v2_dirs()[0],)


def _leafmost_bytes(name: str) -> int | None:
    """A byte-valued cgroup v2 file from the nearest cgroup that publishes it."""
    for base in _usage_dirs():
        value = read_cgroup_bytes(os.path.join(base, name))
        if value is not None:
            return value
    return None


def _leafmost_stat() -> dict[str, int]:
    """`memory.stat` from the nearest cgroup that publishes one, empty when none does."""
    for base in _usage_dirs():
        stat = read_cgroup_stat(base, "memory.stat")
        if stat:
            return stat
    return {}


def _leafmost_events() -> dict[str, int]:
    """`memory.events` from the nearest cgroup that publishes one.

    Deliberately not the ancestry minimum or sum. These counters are per-cgroup and the
    interesting one — `oom_kill` — means "a task *here* was killed"; a parent slice's count
    includes kills in sibling containers that say nothing about this workload's sizing.
    """
    for base in _usage_dirs():
        events = read_cgroup_stat(base, "memory.events")
        if events:
            return events
    return {}


def _memory_psi() -> dict[str, float]:
    """Memory PSI **attributable to this process**, or empty when none is.

    Attribution is the whole difficulty, and getting it wrong is worse than having no signal.
    A host-wide reading — `/proc/pressure/memory`, or the cgroup mount root, which is the same
    thing — sums every tenant on the node. Acting on it means one container halves its morsels
    and spills its operators because a *different* container is thrashing, which relieves
    nothing (the memory is not ours to give back) and costs throughput for as long as the
    neighbour misbehaves. It also makes the level nondeterministic: the same query classifies
    differently depending on what else the box happens to be running.

    So the host file is used only when this process is genuinely **not** in a delegated
    cgroup, where "the host" and "us" are the same set of tasks. Inside a container whose own
    slice publishes no `memory.pressure` — common, since PSI needs `CONFIG_PSI` *and* the
    memory controller enabled down the ancestry — the honest answer is that nothing measured
    our stalls, and callers keep the byte accounting they already had.

    Read leaf-first within our own slice: the leaf is this workload, while a parent slice
    folds in every sibling container's stalls, which is the host-wide problem again one level
    down.
    """
    own = _own_cgroup_dirs()
    for base in own:
        psi = read_psi(os.path.join(base, "memory.pressure"))
        if psi:
            return psi
    if own:
        return {}  # delegated cgroup that publishes no PSI — nothing measured *our* stalls
    return read_psi("/proc/pressure/memory")


def _own_cgroup_dirs() -> tuple[str, ...]:
    """This process's own cgroup slice, leaf first, excluding the mount root.

    `cgroup_v2_dirs` yields the mount root first and then the process's own path from leaf
    upward, so dropping the head leaves exactly the cgroups that describe *this* workload,
    already in leaf-first order. Empty when the process is not in a delegated cgroup at all,
    which is the signal that the host's own figures are legitimately ours.
    """
    return tuple(cgroup_v2_dirs()[1:])
