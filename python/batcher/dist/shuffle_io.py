"""Arrow IPC shuffle files — the object-store-bypassing data-plane transport.

Shuffle partitions are written as Arrow IPC stream files and passed between
stages by path. Only paths transit Ray; the bytes live on disk (local NVMe in
production), so the data plane never touches the Ray object store and is not
bounded by memory.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from typing import NamedTuple

import pyarrow as pa

from batcher._internal.errors import IOError
from batcher._internal.paths import open_private, private_dir

__all__ = [
    "distributed_work_dir",
    "read_ipc",
    "reduce_envelope",
    "shared_scratch_root",
    "shuffle_ipc_options",
    "write_ipc",
    "write_ipc_round_robin",
    "write_shuffle_buckets",
]

# Cluster-shared mounts, in scope order (narrowest first). On managed Ray clusters these are
# network filesystems reachable from every node; the disk shuffle passes only paths
# between tasks, so its scratch dir must live on one of these when the cluster spans
# more than one node. Node-local paths (``/tmp``, ``/mnt/local_storage``) are correct
# only on a single node or a genuinely shared filesystem.
_SHARED_MOUNTS = ("/mnt/cluster_storage", "/mnt/user_storage", "/mnt/shared_storage")


def shared_scratch_root() -> str | None:
    """The base directory for a shuffle scratch dir reachable from every worker node.

    The disk-transport shuffle hands only *paths* between Ray tasks; the bytes live on
    disk. Those paths must resolve on every node a task might run on, so the dir has to be
    on a shared filesystem the moment the cluster spans (or *might* span) more than one
    node. Resolution order:

    1. an explicit ``MemoryConfig.spill_dir`` (operator-configured shared scratch);
    2. otherwise an auto-detected cluster-shared mount (e.g. ``/mnt/cluster_storage``
       and friends), whenever one is present;
    3. otherwise ``None``, so the caller uses a node-local tempdir — correct only where no
       shared mount exists, i.e. a single-node laptop / CI box.

    We deliberately do NOT gate on the *current* node count: an autoscaling cluster can be
    single-node when the work_dir is created and multi-node moments later (the query's own
    ``request_autoscale`` brings spot workers up mid-flight), and a task landing on a
    freshly-joined node must still find the shuffle files. Whenever a shared mount exists
    (i.e. we are on a cluster that *can* scale out) we use it; the only cost is a network
    mount instead of local disk for a distributed run that happens to stay single-node —
    a small price for not losing the whole query to a `FileNotFoundError` when a node
    joins. A machine with no shared mount is a genuine single node, so local temp is right.
    """
    from batcher.config import active_config

    spill_dir = active_config().memory.spill_dir
    if spill_dir:
        return spill_dir
    for mount in _SHARED_MOUNTS:
        if os.path.isdir(mount):
            return os.path.join(mount, "batcher_shuffle")
    return None


def distributed_work_dir(prefix: str) -> str:
    """Create (and return) a unique shuffle scratch dir reachable from every node.

    Falls back to a node-local tempdir on a single-node/shared-filesystem cluster (see
    :func:`shared_scratch_root`). The caller owns the returned dir and removes it.
    """
    root = shared_scratch_root()
    if root:
        # The root is a *shared cluster mount*, so creating it 0755 publishes every
        # query's scratch listing to the whole node. `mkdtemp` already gives the inner
        # directory 0700; this closes the parent, which nothing else does.
        private_dir(root)
        return tempfile.mkdtemp(prefix=prefix, dir=root)
    # No shared mount: this is a genuine single node, so node-local scratch is correct — and
    # the node's measured local volume is a better one than the container root's overlay,
    # which is where a bare tempdir lands on a GPU node.
    #
    # Batcher gets its *own* subdirectory of that volume, and only that subdirectory is
    # tightened. The volume itself belongs to the node — Ray's object spill lives on the same
    # mount, and so does every other tenant's scratch — so chmod'ing it 0700 would take a
    # shared resource away from processes that have nothing to do with this query.
    from batcher._internal.site import local_scratch_root

    local = local_scratch_root()
    if local:
        root = os.path.join(local, "batcher_shuffle")
        private_dir(root)
        return tempfile.mkdtemp(prefix=prefix, dir=root)
    return tempfile.mkdtemp(prefix=prefix)


class ReduceEnvelope(NamedTuple):
    """The spill terms a shuffle reducer was shipped, read from its worker engine config.

    Four reduce paths need these and each read them for itself: the disk aggregate's
    interior combine and its reducer, the disk join's reducer, and the Flight worker's
    bounded reduce. Three spellings of one `json.loads` over a wire contract, on the two
    transports that are supposed to be the same algebra with a different sink — so a change
    to the shipped config's shape had four places to reach and no way to notice a miss.

    `budget` of `0` means unbounded: the reduce takes its in-memory fold. A positive budget
    is what routes it through the out-of-core one instead.
    """

    budget: int
    spill_dir: str | None
    compression: str | None


def reduce_envelope(engine_config: str | None) -> ReduceEnvelope:
    """Read the reducer's spill envelope out of a shipped engine config.

    Args:
        engine_config: The worker's engine config JSON, as shipped by
            `dist.executors.ray_runtime.engine_config_json`. Empty or `None` means no
            envelope was shipped.

    Returns:
        The envelope. `ReduceEnvelope(0, None, None)` when nothing was shipped, which every
        caller reads as "unbounded".

    Examples:
        .. doctest::

            >>> from batcher.dist.shuffle_io import reduce_envelope
            >>> reduce_envelope('{"memory_budget_bytes": 4096}').budget
            4096
            >>> reduce_envelope("").budget
            0
    """
    if not engine_config:
        return ReduceEnvelope(0, None, None)
    cfg = json.loads(engine_config)
    return ReduceEnvelope(
        int(cfg.get("memory_budget_bytes", 0) or 0),
        cfg.get("spill_dir"),
        cfg.get("spill_compression"),
    )


def shuffle_ipc_options(path: str) -> pa.ipc.IpcWriteOptions | None:
    """Arrow-IPC write options for a shuffle file at `path`, or `None` for uncompressed.

    The disk shuffle has the same two tiers the spill store has, and the same answer for
    each. A scratch dir under a **cluster-shared mount** is a network filesystem: every
    byte a mapper writes crosses the wire twice (once to the mount, once back to the
    reducer), so a cheap codec always pays there — the identical trade
    :func:`~batcher.carbonite.spill.disk.remote_ipc_options` makes for the remote spill
    tier, reused rather than restated. A node-local scratch dir is fast disk, so it honors
    the configured ``MemoryConfig.spill_compression`` and stays uncompressed under the
    ``"auto"`` default.

    The Flight transport has compressed its wire since it existed
    (``DistributedConfig.flight_compression``); the disk transport did not, so the one
    shuffle that runs over a network filesystem was the one shuffle sending raw bytes.

    Reads need no counterpart: an Arrow IPC message records its own codec, so
    :func:`read_ipc` decompresses whatever it is handed and a file written by an older
    build still reads.

    Args:
        path: Where the shuffle file will be written. Only its location is read, so the
            decision costs a string comparison and no I/O.

    Returns:
        The write options to pass to ``pa.ipc.new_stream``, or `None` to write
        uncompressed.
    """
    from batcher.carbonite.spill import disk

    # `active_config()` is the driver's only inside the driver: a Ray worker sees its own
    # process default (see `ray_runtime.lifecycle.engine_config_json`). That is why the
    # shared-mount branch reads no config at all — the branch where compression matters
    # decides from the path alone and so agrees on every node. The local branch's default
    # (`"auto"`) is uncompressed either way, so config drift can only change an explicitly
    # set codec on node-local scratch, where the codec is worth the least.
    if _on_shared_mount(path):
        return disk.remote_ipc_options(active_compression())
    return disk.ipc_options(active_compression())


def active_compression() -> str | None:
    """The configured spill/shuffle codec name, ``"auto"``, or `None`."""
    from batcher.config import active_config

    return active_config().memory.spill_compression


def _on_shared_mount(path: str) -> bool:
    """Whether `path` lives under a cluster-shared (network) mount.

    Compares against the same `_SHARED_MOUNTS` roots :func:`shared_scratch_root` picks
    from, plus an explicitly configured `MemoryConfig.spill_dir` — which is documented as
    *operator-configured shared scratch*, so a shuffle placed there is crossing a network
    for the same reason.
    """
    from batcher.config import active_config

    roots = [*_SHARED_MOUNTS]
    spill_dir = active_config().memory.spill_dir
    if spill_dir:
        roots.append(spill_dir)
    try:
        real = os.path.realpath(path)
    except OSError:  # pragma: no cover - realpath on a path we are about to create
        real = path
    return any(
        real == os.path.realpath(root) or real.startswith(os.path.realpath(root) + os.sep)
        for root in roots
    )


def write_ipc(batches: list[pa.RecordBatch], path: str) -> str:
    """Write record batches to an Arrow IPC stream file. Returns `path`."""
    if not batches:
        raise IOError(
            f"cannot write an Arrow IPC shuffle file to {path!r} with no batches: the "
            "stream needs at least one to take its schema from"
        )
    with (
        pa.PythonFile(open_private(path), mode="w") as sink,
        pa.ipc.new_stream(sink, batches[0].schema, options=shuffle_ipc_options(path)) as writer,
    ):
        for b in batches:
            writer.write_batch(b)
    return path


def write_shuffle_buckets(
    buckets: list[list[pa.RecordBatch]], work_dir: str, prefix: str, mapper_id: int
) -> list[str]:
    """Write one IPC file per reducer bucket and return their paths, in reducer order.

    The map half of every disk shuffle ends the same way: `partition_batches` hands back one
    batch list per reducer, and each is written to a file the reducer will later collect by
    name. The naming is the contract between the two halves — `<prefix><mapper>_r<reducer>` —
    so it belongs in one place rather than being spelled out in each operator's map task.

    Args:
        buckets: One batch list per reducer, in reducer order.
        work_dir: The shuffle scratch directory, shared across the cluster.
        prefix: Distinguishes concurrent shuffles in the same directory — `"m"` for the
            aggregate, `"wm"` for the window, `"<side>_m"` for each side of a join.
        mapper_id: This mapper's index, which makes the filename unique across mappers.

    Returns:
        The written paths, one per reducer, in reducer order.
    """
    paths = []
    for reducer, bucket in enumerate(buckets):
        path = os.path.join(work_dir, f"{prefix}{mapper_id}_r{reducer}.arrow")
        write_ipc(bucket, path)
        paths.append(path)
    return paths


def write_ipc_round_robin(
    batches: Iterable[pa.RecordBatch],
    fallback_schema: pa.Schema,
    paths: list[str],
) -> None:
    """Stream `batches` round-robin across per-partition IPC files.

    The driver holds **one batch at a time** — it never materializes the whole
    source — so a larger-than-RAM streaming input is partitioned in bounded memory.
    Each partition's IPC stream is seeded from the first batch's schema (a source
    yields a single consistent schema); a partition that receives no batch still
    gets one schema-only batch (from `fallback_schema` when the source was empty)
    so downstream map tasks always have a schema to operate over.

    Round-robin preserves the row multiset (each worker re-partitions by key before
    producing output, so the distributed result is unchanged); only which worker
    reads which batch differs.
    """
    n = len(paths)
    sinks: list[object | None] = [None] * n
    writers: list[object | None] = [None] * n
    schema: pa.Schema | None = None

    def _open(j: int, sch: pa.Schema) -> None:
        # Owner-only, like every other shuffle artifact: these files hold the query's rows
        # on a scratch path that may be a shared cluster mount.
        sinks[j] = pa.PythonFile(open_private(paths[j]), mode="w")
        writers[j] = pa.ipc.new_stream(sinks[j], sch, options=shuffle_ipc_options(paths[j]))

    try:
        for i, b in enumerate(batches):
            if schema is None:
                schema = b.schema
            j = i % n
            if writers[j] is None:
                _open(j, schema)
            writers[j].write_batch(b)  # type: ignore[union-attr]
        if schema is None:
            schema = fallback_schema
        empty = pa.RecordBatch.from_pylist([], schema=schema)
        for j in range(n):
            if writers[j] is None:
                _open(j, schema)
                writers[j].write_batch(empty)  # type: ignore[union-attr]
    finally:
        for w in writers:
            if w is not None:
                w.close()  # type: ignore[attr-defined]
        for s in sinks:
            if s is not None:
                s.close()


def read_ipc(path: str) -> list[pa.RecordBatch]:
    """Read all record batches from an Arrow IPC stream file."""
    with pa.OSFile(path, "rb") as src, pa.ipc.open_stream(src) as reader:
        return list(reader)
