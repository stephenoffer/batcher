"""Session-level administration: table maintenance and streaming-query control.

`compact` and `vacuum` are the two halves of the small-files story — one rewrites,
only the other deletes. `streams` and `await_any_termination` are the Spark-shaped
handles on the queries a streaming write started.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "add_streaming_listener",
    "await_any_termination",
    "compact",
    "remove_streaming_listener",
    "reset_terminated",
    "streaming_listeners",
    "streams",
    "vacuum",
]


def add_streaming_listener(listener: Any) -> None:
    """Register a `StreamingQueryListener` (Spark ``spark.streams.addListener``).

    Every streaming query in this process — already running or started later — reports
    its start, each completed micro-batch, and its termination to the listener. A
    listener that raises is logged and skipped, never allowed to fail the query.

    Args:
        listener: A `StreamingQueryListener` subclass instance. Registering the same
            one twice adds it once.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> listener = bt.StreamingQueryListener()
            >>> bt.add_streaming_listener(listener)
            >>> bt.remove_streaming_listener(listener)
            True
    """
    from batcher.plan.streaming import add_streaming_listener as _add

    _add(listener)


def remove_streaming_listener(listener: Any) -> bool:
    """Unregister a `StreamingQueryListener` (Spark ``spark.streams.removeListener``).

    Args:
        listener: The listener to remove.

    Returns:
        ``True`` if it was registered, ``False`` if it was not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.remove_streaming_listener(bt.StreamingQueryListener())
            False
    """
    from batcher.plan.streaming import remove_streaming_listener as _remove

    return _remove(listener)


def streaming_listeners() -> list[Any]:
    """Every registered `StreamingQueryListener`, in registration order.

    Returns:
        The registered listeners.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.streaming_listeners()
            []
    """
    from batcher.plan.streaming import streaming_listeners as _listeners

    return _listeners()


def streams() -> list[Any]:
    """List the currently-active streaming queries (Spark ``spark.streams.active``).

    Each entry is a handle to a query started by a streaming write that is still
    running, so you can track or stop it. Empty when no stream is active.

    Returns:
        The active streaming-query handles.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.streams()
            []
    """
    from batcher.api.streaming import active_streams

    return active_streams()


def await_any_termination(timeout: float | None = None) -> bool:
    """Block until any active streaming query stops (Spark ``awaitAnyTermination``).

    Waits for the first currently-running query to terminate, re-raising its exception
    if it failed. Returns immediately when no query is active.

    Args:
        timeout: Maximum seconds to wait; ``None`` waits indefinitely.

    Returns:
        ``True`` if a query stopped (or none were active), ``False`` on timeout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.await_any_termination(timeout=0.0)
            True
    """
    from batcher.api.streaming import await_any_termination as _await_any

    return _await_any(timeout)


def reset_terminated() -> None:
    """Forget terminations already reported by `await_any_termination` (Spark parity).

    Spark's `awaitAnyTermination` returns immediately once any query has terminated, and
    keeps doing so until `resetTerminated()` clears the record — so a supervisor that
    restarts a failed query and loops back in without resetting spins at full speed on a
    termination it has already handled.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.reset_terminated()
            >>> bt.await_any_termination(timeout=0.0)
            True
    """
    from batcher.api.streaming._query import reset_terminated as _reset

    _reset()


def compact(
    path: str,
    *,
    target_size_mb: float = 128.0,
    num_files: int | None = None,
    by: str | list[str] | None = None,
    z_order: list[str] | None = None,
    where: Any = None,
    format: str | None = None,
    **opts: Any,
) -> Any:
    """Compact a dataset in place — rewrite many small files into fewer, larger ones.

    The fix for the small-files problem (tiny part files from incremental writes). What it
    *does* depends on what `path` is, and the difference matters:

    * A **transactional table** (Delta) is compacted as a transaction. Small files are
      bin-packed and a new version retires the old ones *from the log*, leaving them on
      storage — so every existing version still reads and time travel survives. Nothing is
      deleted here; `vacuum` is what reclaims. Pass `z_order=[...]` to sort the rewritten
      rows along a Z-curve over those columns, which narrows each file's min/max bounds and
      so multiplies what the *next* query can skip from the log alone. `where` restricts
      the work to matching partitions.
    * A **plain file directory** (Parquet, CSV, ...) has no log, so it is read,
      repartitioned to ~`target_size_mb` files (or exactly `num_files`, optionally
      Hive-partitioned by `by`), written back, and the replaced part-files removed. Nothing
      references the old files, so removing them is safe. Single-writer only.

    Examples:
        .. doctest::

            >>> import tempfile, os, glob
            >>> import batcher as bt
            >>> d = tempfile.mkdtemp()
            >>> _ = bt.from_pydict({"x": [1, 2, 3, 4]}).repartition(num_files=2).write(
            ...     d, format="parquet"
            ... )
            >>> _ = bt.compact(d, num_files=1, format="parquet")
            >>> len(glob.glob(os.path.join(d, "*.parquet")))
            1

    Args:
        path: The dataset location to compact in place.
        target_size_mb: Approximate target size per output file (ignored if
            `num_files` is given).
        num_files: Exact number of output files to rewrite to (file directories only).
        by: Column(s) to Hive-partition the rewritten output by (file directories only).
        z_order: Columns to Z-order the rewritten rows by (transactional tables only).
        where: Partition filters limiting the scope (transactional tables only).
        format: The dataset format; inferred from `path` when omitted.
        **opts: Extra options forwarded to the writer / maintenance backend.

    Returns:
        The `WriteManifest` for a file directory; the backend's optimize metrics for a
        transactional table.
    """
    import os

    from batcher._internal.errors import PlanError
    from batcher.api.session.read import read
    from batcher.io.detect import detect_format
    from batcher.io.filesystem import resolve_filesystem
    from batcher.io.formats.base import SOURCES
    from batcher.io.formats.lakehouse.maintenance import table_maintenance

    fmt = detect_format(path, format)

    # A transactional table must be maintained transactionally. The file rewrite below
    # deletes what it replaces, and a table's older versions still *reference* those files
    # — doing it to a Delta table silently destroys time travel (and destroys it
    # invisibly, since `count()` keeps answering from the log after the data is gone).
    maintenance = table_maintenance(fmt)
    if maintenance is not None:
        target = None if num_files is not None else int(target_size_mb * 1024 * 1024)
        return maintenance.compact(
            path, target_size_bytes=target, z_order=z_order, where=where, **opts
        )
    if z_order is not None or where is not None:
        raise PlanError(
            f"compact(): z_order/where need a transactional table; {fmt!r} is a file "
            "directory with no transaction log. Use num_files/target_size_mb, or write "
            "to a Delta table."
        )

    fs = resolve_filesystem(path)
    suffix = getattr(SOURCES.get(fmt), "suffix", "")
    try:
        old_files = list(fs.expand(path, suffix=suffix))
    except OSError:
        old_files = []

    spec: dict[str, Any] = {"by": by} if by is not None else {}
    if num_files is not None:
        spec["num_files"] = num_files
    else:
        spec["target_size_mb"] = target_size_mb
    manifest = (
        read(path, format=fmt).repartition(**spec).write(path, format=fmt, mode="overwrite", **opts)
    )

    new_names = {os.path.basename(f.path) for f in manifest.files}
    for f in old_files:
        if os.path.basename(f) not in new_names:
            fs.remove(f)
    return manifest


def vacuum(
    path: str,
    *,
    retention_hours: float | None = None,
    dry_run: bool = True,
    format: str | None = None,
    **opts: Any,
) -> list[str]:
    """Reclaim the data files of a transactional table that no live version references.

    The counterpart to `compact`. Compaction never deletes — it rewrites small files and
    retires the old ones from the log, leaving them on storage so time travel still works.
    This is the operation that eventually removes them, and it is the only one allowed to.

    It **defaults to a dry run**, reporting what it would delete and deleting nothing,
    because the files it removes are precisely the ones older versions and any in-flight
    reader depend on. The retention window is the safety argument: a file is only removed
    once it has been unreferenced for longer than any reader could still be using it.
    Shortening the window below the table's configured minimum means an active reader can
    have its files deleted mid-scan, so the backend refuses unless you waive the check
    explicitly.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> would_delete = bt.vacuum("s3://lake/events")  # doctest: +SKIP
            >>> bt.vacuum("s3://lake/events", dry_run=False)  # doctest: +SKIP

    Args:
        path: The table root.
        retention_hours: How long an unreferenced file is kept before it can be
            reclaimed. Defaults to the format's own default (7 days for Delta).
        dry_run: When True (the default), report the files but delete nothing.
        format: The table format; inferred from `path` when omitted.
        **opts: Backend options (e.g. ``storage_options``).

    Returns:
        The files deleted — or, on a dry run, the files that would be.

    Raises:
        PlanError: If `path` is not a transactional table (a plain file directory has no
            log, so nothing is unreferenced and there is nothing to reclaim).
    """
    from batcher._internal.errors import PlanError
    from batcher.io.detect import detect_format
    from batcher.io.formats.lakehouse.maintenance import table_maintenance

    fmt = detect_format(path, format)
    maintenance = table_maintenance(fmt)
    if maintenance is None:
        raise PlanError(
            f"vacuum() needs a transactional table; {fmt!r} is a plain file directory "
            "with no transaction log, so no file is unreferenced and there is nothing "
            "to reclaim."
        )
    return maintenance.vacuum(path, retention_hours=retention_hours, dry_run=dry_run, **opts)
