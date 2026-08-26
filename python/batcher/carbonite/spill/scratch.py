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
    root = active_config().memory.spill_dir or local_scratch_root()
    if root:
        os.makedirs(root, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix, dir=root), True
    return tempfile.mkdtemp(prefix=prefix), True


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
