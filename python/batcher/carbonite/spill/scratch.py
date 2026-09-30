"""Where Carbonite's out-of-core bytes go: resolving a scratch directory and its store.

Two callers need the same answer to "which local volume should this write to, and how
should the tiered store over it be configured": the distributed spill path
(`dist.spill`) and the result cache's disk tier (`carbonite.cache_disk`). `dist` may
import `carbonite`, but not the reverse, so the shared answer lives here — the lowest
layer both can see — rather than being pasted into each.

The directory choice is not a detail. On a GPU or container node a system tempdir is an
overlay on the container root, commonly under 100 GB and shared with the image and every
other tenant, while the several terabytes of local NVMe the node ships with are mounted
under a provider-specific name. Writing to the tempdir there fails with `ENOSPC` beside
unused storage, and the failure reads as an undersized query rather than a misplaced
directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from batcher.carbonite.spill.store import TieredSpillStore
from batcher.config import active_config

__all__ = ["make_store", "scratch_dir"]


def scratch_dir(spill_dir: str | None, prefix: str) -> tuple[str, bool]:
    """Resolve the local scratch directory to write under, and whether we own it.

    An explicit `spill_dir` is caller-owned and never removed. Otherwise, if the config
    sets `MemoryConfig.spill_dir`, a unique subdirectory is created *under* that root —
    so striping onto fast or large disks is honored and a later `rmtree` can only ever
    remove our own subdirectory, never a shared root. With neither, this falls back to
    the node's measured local scratch volume, and to a system tempdir only when there is
    none.

    Args:
        spill_dir: An explicit directory to use, or `None` to resolve one.
        prefix: Prefix for the created directory's name, so a stray one is attributable.

    Returns:
        The directory to write under, and whether the caller owns it (may remove it).
    """
    from batcher._internal.site import local_scratch_root

    if spill_dir is not None:
        return spill_dir, False
    root = active_config().memory.spill_dir or local_scratch_root() or tempfile.gettempdir()
    os.makedirs(root, exist_ok=True)
    _sweep_orphans(root, prefix)
    return tempfile.mkdtemp(prefix=f"{prefix}{os.getpid()}-", dir=root), True


#: `(root, prefix)` pairs this process has already swept, so a sweep costs one directory
#: listing per root rather than one per breaker.
_SWEPT: set[tuple[str, str]] = set()


def _sweep_orphans(root: str, prefix: str) -> None:
    """Remove scratch directories under `root` left by processes that no longer exist.

    `rmtree` in a `finally` and an `atexit` hook cover every ordinary end, and neither runs
    on `SIGKILL` -- which is how the OOM killer ends the process most likely to be spilling.
    Its scratch then outlives it on the spill volume, so the next query has less room and is
    likelier to be killed in turn. The engine's own grace stores already embed their pid and
    sweep the dead ones (`bc-runtime` `DiskSpillStore`); this is the same fence for the
    directories allocated here, which is what the pid in the name is for.

    Only a name this module creates -- `prefix`, a pid, `-` -- is ever considered, and only
    when that pid is not a live process, so a concurrently spilling sibling is never touched.
    A reused pid reads as alive and the directory is kept, which is the safe way to be wrong.
    Best-effort throughout: cleanup must never fail a query.

    Args:
        root: The scratch root the new directory is about to be created under.
        prefix: The name prefix of the directories to consider.
    """
    key = (os.path.abspath(root), prefix)
    if key in _SWEPT:
        return
    _SWEPT.add(key)
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        pid = _owner_pid(name, prefix)
        if pid is not None and pid != os.getpid() and not _alive(pid):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _owner_pid(name: str, prefix: str) -> int | None:
    """The pid in a `{prefix}{pid}-{suffix}` directory name, or `None` for any other name."""
    if not name.startswith(prefix):
        return None
    pid, sep, _ = name[len(prefix) :].partition("-")
    return int(pid) if sep and pid.isdigit() else None


def _alive(pid: int) -> bool:
    """Whether `pid` is a live process; anything that cannot answer says it is."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def make_store(work_dir: str) -> TieredSpillStore:
    """A tiered spill store rooted at `work_dir`, configured from the active `Config`.

    Local NVMe by default, overflowing to `MemoryConfig.spill_remote_uri` once the local
    budget is exhausted — so a write survives a full local disk. Batches are compressed
    with the configured codec.

    Args:
        work_dir: The local directory the store's local tier writes under.

    Returns:
        The configured store.
    """
    mem = active_config().memory
    return TieredSpillStore(
        work_dir,
        remote_uri=mem.spill_remote_uri,
        local_budget_bytes=mem.spill_local_budget_bytes,
        compression=mem.spill_compression,
    )
