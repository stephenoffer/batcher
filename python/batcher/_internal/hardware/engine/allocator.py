"""What the engine's allocator is holding, and how to make it give some back.

The data plane allocates every morsel through mimalloc, which **retains freed pages by
design**. That retention is why the engine scales across cores at all: glibc's malloc serves
buffers of that size through `mmap`/`munmap`, and each `munmap` broadcasts a TLB-shootdown
interrupt to every core, which turns a parallel scan into a serialization point.

The cost is that process RSS keeps counting memory the engine has already finished with. A
memory envelope computed from `psutil` therefore over-counts the data plane and spills earlier
than it needs to. There is no Python-side view of the split — only the allocator knows which
of its pages are live and which are merely retained.

[`allocator_stats`] exposes that split, and [`release_retained_memory`] is what makes it
actionable: handing the retained arena back is far cheaper than writing a hash table to disk,
so it is the thing to try first when an envelope is about to force a spill.
"""

from __future__ import annotations

from batcher._internal.hardware.engine.detected import call_engine

__all__ = ["allocator_stats", "release_retained_memory"]


def allocator_stats() -> dict[str, int]:
    """mimalloc's own accounting for the engine process.

    Keys, bytes unless noted: `rss`, `peak_rss`, `commit`, `peak_commit`, `page_faults`,
    `elapsed_ms`, `user_ms`, `system_ms`.

    Deliberately not memoized. Unlike everything else in this package these are readings
    rather than facts about the machine, and the whole point is to watch them move.

    The gap between `commit` and `rss` is retained-but-free arena, which
    :func:`release_retained_memory` can hand back.

    Returns:
        Allocator counters, or an empty dict when the engine cannot report them.
    """
    return call_engine("allocator_stats") or {}


def release_retained_memory(force: bool = False) -> int:
    """Return the engine allocator's retained free pages to the operating system.

    Call this **before** spilling, never in a hot path. Forcing pages back re-imposes exactly
    the unmapping cost that retention exists to avoid, and that trade only inverts when the
    alternative is writing operator state to disk.

    Args:
        force: Also walk other threads' heaps. Thorough and considerably more expensive;
            leave it false for a routine trim.

    Returns:
        Bytes of resident memory released, by the allocator's own accounting. Zero means
        there was nothing retained to give back, or that the engine cannot report.
    """
    return int(call_engine("allocator_collect", force) or 0)
