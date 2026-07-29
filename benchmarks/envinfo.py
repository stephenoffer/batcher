"""What the box was, and whether it was fit to measure on.

Every trap recorded in ``BENCHMARK_RESULTS.md`` that produced a *wrong number* rather than
a failed run came from the environment rather than the query: a debug build timed against
release competitors (~8-60x), a box at load 25 on 15 cores that made DuckDB "scale"
backwards from 1 to 8 threads, and ratios quoted across four different machines as if they
were one series. None of those is visible in a table of milliseconds.

So this module does two things and they are deliberately separate:

- **Refuse**: `require_release_build` and `require_quiet_box` stop a run that cannot
  produce a measurement. Both are hard stops rather than warnings, because a warning
  printed above a table of numbers is a warning that gets copied into a document without
  it.
- **Record**: `machine_fingerprint` is what every result document must carry, so a number
  can never be compared against one from a different box without that being obvious.

`require_release_build` moved here from ``run.py`` when the concurrency and resilience
harnesses needed the same gate; it is the same check, unchanged.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

__all__ = [
    "QUIET_LOAD_PER_CORE",
    "load_per_core",
    "machine_fingerprint",
    "require_quiet_box",
    "require_release_build",
]


#: Load-per-core above which a timing is not a measurement. One runnable task per core is
#: exact saturation, so 0.4 leaves the box comfortably idle while tolerating the couple of
#: background threads any Python process carries. The number that motivated a threshold at
#: all was 25 on 15 cores (`BENCHMARK_RESULTS.md`), roughly 1.7 — this is far below it, and
#: deliberately so: the concurrency harness saturates the box *by design*, so a neighbour
#: that would be merely annoying elsewhere silently decides the answer here.
QUIET_LOAD_PER_CORE = 0.4


def load_per_core() -> float | None:
    """The 1-minute run-queue length per available core, or None where unmeasurable.

    Delegates to the engine's own contention probe, which is cgroup-aware — the core count
    that matters is the one the cgroup grants, not the one the kernel advertises. A 96-core
    host under a 15-core quota reads as idle by the naive division and as saturated by this
    one.

    Returns:
        Load per available core, or None on a platform that cannot report it.
    """
    from batcher._internal.hardware import cpu_contention

    return cpu_contention().get("load_per_core")


def require_release_build(*, allow_debug: bool = False) -> None:
    """Refuse to let a debug-build timing pass for a measurement.

    ``just build`` (``maturin develop``) builds the engine with the *dev* profile: no
    ``opt-level``, ``debug_assertions`` on, and every third-party crate unoptimized with
    it. Every comparator here is an installed release wheel, so timing a dev build against
    them measures the profile rather than the engine.

    This used to be a *timing heuristic* — sum 4M integers and warn if it took over 60 ms.
    That guard silently passed on a real dev-build benchmark run: the sum came in under the
    threshold, the suite printed a clean table, and the ratios in it were meaningless. A
    heuristic cannot see a build profile; the engine can, so it now reports one
    (``_native.__engine_profile__``) and this reads it.

    A dev build is a hard stop rather than a warning. A warning above a table of numbers is
    a warning that gets copied into a document without its warning.

    Args:
        allow_debug: Skip the check, for deliberately timing a debug build.

    Raises:
        SystemExit: If the installed engine is not a release build.
    """
    if allow_debug:
        return
    from batcher._internal.native import engine

    profile = getattr(engine(), "__engine_profile__", None)
    if profile == "release":
        return
    if profile is None:
        # An engine too old to report its profile. Say so rather than assume either way.
        print(
            "WARNING: this engine does not report its build profile; if it was built with"
            " `just build` the timings below are not measurements.\n"
        )
        return
    raise SystemExit(
        "\n"
        + "!" * 78
        + f"\nThe installed engine is a {profile.upper()} build, so any timing here would"
        "\ncompare an unoptimized Batcher against release competitors.\n"
        "\nRebuild first:   just build-release\n"
        "\nTo time a debug build deliberately, pass --allow-debug-build.\n" + "!" * 78
    )


def require_quiet_box(*, threshold: float = QUIET_LOAD_PER_CORE, allow_busy: bool = False) -> None:
    """Refuse to time anything on a box someone else is already using.

    On a contended box the numbers do not merely get worse, they stop meaning anything:
    at load 25 on 15 cores DuckDB measured *slower* at 8 threads (13.42 s) than at 1
    (6.94 s), which is not a fact about DuckDB. Several deltas in ``BENCHMARK_RESULTS.md``
    are explicitly disavowed for exactly this.

    Args:
        threshold: Maximum tolerated load per core.
        allow_busy: Warn instead of exiting, for a run whose purpose is the contention.

    Raises:
        SystemExit: If the box is busier than `threshold` and `allow_busy` is False.
    """
    load = load_per_core()
    if load is None or load <= threshold:
        return
    message = (
        f"load average is {load:.2f} per core (threshold {threshold:.2f}). Another process "
        "is using this machine, so any timing here measures the neighbour."
    )
    if allow_busy:
        print(f"WARNING: {message}\n")
        return
    raise SystemExit(
        "\n" + "!" * 78 + f"\n{message}\n"
        "\nWait for the box to go idle, or pass --allow-busy-box to record a number that\n"
        "is explicitly not comparable to any other run.\n" + "!" * 78
    )


def _git_sha() -> str:
    """The working tree's commit, suffixed ``-dirty`` when it has uncommitted changes."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _cpu_model() -> str:
    """The CPU's marketing name from ``/proc/cpuinfo``, or the platform's processor string."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine_fingerprint() -> dict[str, object]:
    """Everything needed to know whether two numbers are comparable.

    This is the first record of every result document the benchmark harnesses write. The
    reason it exists is that ``BENCHMARK_RESULTS.md`` accumulated numbers from at least
    four machines — a 16-core box under a 15-core cgroup quota, a 92-core box, a 96-core
    box, and an 8-node cluster — and ratios move by an order of magnitude between them.
    Without the fingerprint attached, two rows of the same table are not a comparison.

    Both core counts are recorded on purpose: `cpu_count_available` is what the cgroup
    grants and what the engine will actually use, `cpu_count_logical` is what the kernel
    advertises. When they differ, that difference is usually the whole story.

    Returns:
        A JSON-serializable mapping of host, CPU, memory, kernel, engine, and git facts.
    """
    from batcher._internal.hardware import available_cpu_count, machine_memory_bytes
    from batcher.api.session.versions import versions

    vers = versions()
    return {
        "host": platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count() or 0,
        "cpu_count_available": available_cpu_count(),
        "memory_bytes": machine_memory_bytes(),
        "kernel": platform.platform(),
        "python": vers["python"],
        "batcher": vers["batcher"],
        "engine": vers["engine"],
        "engine_profile": vers["engine_profile"],
        "git_sha": _git_sha(),
        "load_per_core_at_start": load_per_core(),
        "argv": sys.argv[1:],
    }
