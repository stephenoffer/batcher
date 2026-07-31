"""What the container runtime took away, and the three limits that cost a GPU job silently.

A rented GPU node is almost never reached directly. The job arrives inside a container the
platform started, and a container runtime's defaults were chosen for a web service rather than
for a data engine feeding eight accelerators. Three of those defaults are actively harmful
here, and each one fails in a way that does not name itself:

* **`/dev/shm` defaults to 64 MiB.** Docker's default, inherited by a great many Kubernetes
  pods that never set `emptyDir.medium: Memory`. Batcher writes memory-mapped Arrow shards
  there to hand a UDF's input to worker processes without pickling it, and a parallel JSON
  write stages through it. At 64 MiB those writes fail with `ENOSPC` on any real batch, and
  the failure surfaces as a broken worker rather than as "your container has no shared
  memory". The same default is what breaks a PyTorch dataloader with more than zero workers,
  which is why it is a familiar symptom on this hardware and an unfamiliar cause.

* **`RLIMIT_MEMLOCK` defaults to 64 KiB.** Page-locked host memory is what makes a
  host-to-device copy asynchronous and what GPUDirect RDMA requires. At 64 KiB nothing can be
  pinned, so every staging buffer falls back to pageable memory and every transfer becomes a
  synchronous bounce through a driver-owned buffer. The job is correct and roughly half the
  speed, and nothing logs a reason. `--ulimit memlock=-1` is the fix and it is not the default.

* **`RLIMIT_NOFILE`** bounds how many files a scan can hold open at once. A partitioned
  lakehouse table and a wide shuffle both reach for a lot of descriptors, and the failure is
  an `EMFILE` deep inside a reader.

Everything here is a *fact*, never a verdict: this module reports what the limits are, and
whether they are worth acting on is the caller's. `container_findings` is the one exception and
it is deliberately shaped as advice rather than as policy — it names the condition and the flag
that fixes it, because an operator who cannot read this from inside the container is exactly
the person who needs to be told.

Off Linux, and anywhere a limit cannot be read, every figure is `0` and every finding is
absent. Unknown is not a finding.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import functools
import os
import sys

__all__ = [
    "DOCKER_DEFAULT_SHM_BYTES",
    "MIN_USEFUL_SHM_BYTES",
    "container_findings",
    "in_container",
    "memlock_limit_bytes",
    "open_files_limit",
    "shm_bytes",
    "shm_root",
    "usable_shm",
]

#: What Docker gives a container that does not ask for more. Recognized exactly, because a
#: `/dev/shm` at precisely this size is a default nobody chose rather than a deliberate budget.
DOCKER_DEFAULT_SHM_BYTES = 64 * (1 << 20)

#: Below this, `/dev/shm` is not worth using for staging and the system temp directory is the
#: better answer. One gibibyte is roughly the working set of a single morselized batch group
#: across a handful of worker processes — enough that a shard write will not fail, and low
#: enough that a deliberately modest allocation still qualifies.
MIN_USEFUL_SHM_BYTES = 1 << 30


def _rlimit(name: str) -> int:
    """A soft resource limit by name, or `0` when it cannot be read.

    `RLIM_INFINITY` is reported as `0` alongside every other unreadable case, because an
    unlimited limit and an unknown one lead to the same decision: there is nothing to act on.
    """
    if sys.platform == "win32":
        return 0
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return 0
    which = getattr(resource, name, None)
    if which is None:
        return 0
    try:
        soft, _hard = resource.getrlimit(which)
    except (OSError, ValueError):
        return 0
    return 0 if soft == resource.RLIM_INFINITY else max(0, soft)


def memlock_limit_bytes() -> int:
    """Page-locked memory this process may hold, in bytes.

    Returns:
        The soft `RLIMIT_MEMLOCK`, or `0` when it is unlimited or unreadable. Unlimited is the
        answer a caller wants and needs no action, so it shares the do-nothing sentinel.
    """
    return _rlimit("RLIMIT_MEMLOCK")


def open_files_limit() -> int:
    """File descriptors this process may hold open.

    Returns:
        The soft `RLIMIT_NOFILE`, or `0` when unlimited or unreadable.
    """
    return _rlimit("RLIMIT_NOFILE")


def _shm_stat() -> tuple[int, int]:
    """`(total, free)` bytes of `/dev/shm`, or `(0, 0)` when it cannot be stat'd.

    One reader for both figures, so a test fakes the filesystem here rather than by patching
    `os.statvfs` for the whole process.
    """
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return (0, 0)
    return (stat.f_blocks * stat.f_frsize, stat.f_bavail * stat.f_frsize)


def shm_bytes() -> int:
    """Size of `/dev/shm`, in bytes.

    Returns:
        The filesystem's total size, `0` when there is no `/dev/shm` or it cannot be stat'd.
        Total rather than free: the question this answers is what the runtime allocated, which
        is a property of the container, while free space is a property of the moment.
    """
    return _shm_stat()[0]


def usable_shm(needed_bytes: int = 0) -> bool:
    """Whether `/dev/shm` is worth staging through.

    Args:
        needed_bytes: Space this caller is about to want. Checked against *free* space, since
            an allocation that is large but already full helps nobody.

    Returns:
        True when `/dev/shm` exists, is at least `MIN_USEFUL_SHM_BYTES`, and has room for
        `needed_bytes`.
    """
    total, free = _shm_stat()
    if total < MIN_USEFUL_SHM_BYTES:
        return False
    return needed_bytes <= 0 or free >= needed_bytes


def shm_root(needed_bytes: int = 0) -> str:
    """Where to stage memory-mapped shards: `/dev/shm` when it is real, a temp dir otherwise.

    The one call every staging path should make. Testing `os.path.isdir("/dev/shm")` — which is
    what the call sites did — is true inside a container with a 64 MiB allocation, so the
    directory exists, the write starts, and it fails with `ENOSPC` partway through a batch
    group. A slower temp directory that works beats a fast one that does not.

    Args:
        needed_bytes: Space the caller is about to want, or `0` to ask only whether the
            allocation is a real one.

    Returns:
        A directory that exists and can be written to.
    """
    import tempfile

    return "/dev/shm" if usable_shm(needed_bytes) else tempfile.gettempdir()


@functools.lru_cache(maxsize=1)
def in_container() -> bool:
    """Whether this process is running inside a container.

    Three signals, because no one of them holds across runtimes: Docker's marker file, the
    cgroup v2 marker that a container runtime leaves in the namespace root, and a container
    engine's name in the process's own cgroup path.

    Returns:
        True when any of them is present. False off Linux, and on a host where none is.
    """
    if sys.platform != "linux":
        return False
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read()
    except OSError:
        return False
    return any(marker in path for marker in ("docker", "kubepods", "containerd", "lxc"))


def container_findings() -> tuple[str, ...]:
    """Container limits that will cost this job something, each named with its fix.

    Advice rather than policy: nothing here changes behavior, and a deployment that has made
    one of these choices deliberately is free to ignore the line. What it must not do is
    discover the limit from a broken worker three hours in.

    Returns:
        One sentence per finding, empty on a host with nothing to report and on one where the
        limits could not be read. Unknown is not a finding.
    """
    out: list[str] = []
    shm = shm_bytes()
    if 0 < shm < MIN_USEFUL_SHM_BYTES:
        default = " (the container runtime's default)" if shm == DOCKER_DEFAULT_SHM_BYTES else ""
        out.append(
            f"/dev/shm is {shm / (1 << 20):.0f} MiB{default}: too small to stage a worker's "
            "input through, so process UDFs and parallel JSON writes fall back to disk. "
            "Raise it with `--shm-size` or a Memory-medium emptyDir volume."
        )
    memlock = memlock_limit_bytes()
    if 0 < memlock < (1 << 20):
        out.append(
            f"memlock is limited to {memlock // 1024} KiB: host memory cannot be page-locked, "
            "so every host-to-device copy is synchronous and GPUDirect RDMA is unavailable. "
            "Raise it with `--ulimit memlock=-1`."
        )
    nofile = open_files_limit()
    if 0 < nofile < 4096:
        out.append(
            f"only {nofile} file descriptors: a partitioned scan or a wide shuffle will "
            "exhaust them. Raise it with `--ulimit nofile=65536`."
        )
    return tuple(out)
