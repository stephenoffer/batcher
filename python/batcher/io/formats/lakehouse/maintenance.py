"""Table maintenance — compaction, clustering, and reclamation, as transactions.

A lakehouse table cannot be maintained the way a directory of Parquet files can. The
file-level compactor rewrites a directory and then *deletes* what it replaced; do that to
a table and you delete data files the older versions still reference, which silently
destroys time travel — and destroys it invisibly, because a `count()` is answered from
the log and keeps reporting the rows whose files are gone. (That is not hypothetical: it
is what `bt.compact` did to a Delta table before this module existed.)

Maintaining a table is therefore a *transaction*, not a file operation, and it splits in
two:

* **Compaction rewrites, and never deletes.** It bin-packs small data files into larger
  ones and commits a new version that removes the old ones *from the log*. The old files
  stay on storage, so every existing version still reads. Z-ordering is the same
  transaction with the rows sorted along a space-filling curve first, which tightens each
  file's min/max bounds and so multiplies what the next query can skip
  (`io.stats.file_skipping`) — clustering and skipping are the same mechanism seen from
  the two ends.
* **Vacuum deletes, and never rewrites.** It removes files that *no live version*
  references, after a retention window long enough that no in-flight reader still needs
  them. This is the only operation allowed to delete, and the retention window is the
  whole safety argument.

Formats register a `TableMaintenance` here; `bt.compact` / `bt.vacuum` dispatch through
the registry, and a format with no entry (a plain Parquet directory) keeps the file-level
behavior. Backends are imported lazily, so this module costs nothing to import.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from batcher._internal.errors import BackendError
from batcher._internal.registry import Registry

__all__ = ["MAINTENANCE", "TableMaintenance", "table_maintenance"]


@runtime_checkable
class TableMaintenance(Protocol):
    """Transactional maintenance for one table format.

    Every method commits a new version and returns the backend's metrics. None of them
    may delete a file another version still references — that is `vacuum`'s job, and only
    after its retention window.
    """

    def compact(
        self,
        path: str,
        *,
        target_size_bytes: int | None = None,
        z_order: list[str] | None = None,
        where: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """Bin-pack the table's small files into larger ones, as one commit."""
        ...

    def vacuum(
        self,
        path: str,
        *,
        retention_hours: float | None = None,
        dry_run: bool = True,
        **opts: Any,
    ) -> list[str]:
        """Delete files no live version references, older than the retention window."""
        ...


MAINTENANCE: Registry[type[TableMaintenance]] = Registry("table maintenance")


def table_maintenance(fmt: str) -> TableMaintenance | None:
    """The maintenance backend for `fmt`, or None if the format is not transactional.

    None is the signal to fall back to file-level handling — a Parquet directory has no
    log, so there is no transaction to make and nothing that could reference an old file.

    Args:
        fmt: The registered format name (``"delta"``, ``"iceberg"``, ...).

    Returns:
        The format's maintenance backend, or None.
    """
    if fmt not in MAINTENANCE:
        return None
    return MAINTENANCE.get(fmt)()


def unsupported(operation: str, fmt: str, reason: str) -> BackendError:
    """The error a format raises for maintenance its client genuinely cannot perform.

    Kept honest on purpose: a maintenance call that silently does nothing is worse than
    one that fails, because the user believes their table was maintained.
    """
    return BackendError(f"{operation} is not supported for {fmt} tables: {reason}")


#: A write that appends a handful of rows leaves a file far below any sane target size, and
#: the *next* write cannot fix that — it just adds another. So the small-file problem is
#: cumulative by construction, and the only honest trigger is the table's standing count of
#: small files, never the count this one write produced.
AUTO_COMPACT_MIN_FILES = 50
AUTO_COMPACT_TARGET_BYTES = 128 * 1024 * 1024


def auto_compact(
    path: str,
    fmt: str,
    *,
    min_files: int = AUTO_COMPACT_MIN_FILES,
    target_size_bytes: int = AUTO_COMPACT_TARGET_BYTES,
    **opts: Any,
) -> dict[str, Any] | None:
    """Bin-pack the table after a write, if it has accumulated enough small files.

    The self-maintaining half of an incremental pipeline: a streaming or micro-batch writer
    leaves one small file per commit, and a table nobody compacts eventually costs more to
    *plan* than to read. This runs the same transactional compaction `bt.compact` does, so
    it never deletes a file an older version references.

    The trigger deliberately counts the table's *standing* small files rather than the
    ones this write produced — a write that adds a single small file to a table already
    holding a thousand of them is exactly the case that needs compacting, and a
    per-write trigger would never fire on it.

    Args:
        path: The table root.
        fmt: The table format.
        min_files: How many below-target files must exist before compacting.
        target_size_bytes: The size a file must reach to count as "not small".
        **opts: Forwarded to the maintenance backend.

    Returns:
        The backend's optimize metrics, or None if the table did not need compacting (or
        the format has no compaction).
    """
    backend = table_maintenance(fmt)
    if backend is None:
        return None
    counter = getattr(backend, "small_file_count", None)
    if counter is None:
        return None
    try:
        if counter(path, below_bytes=target_size_bytes, **opts) < min_files:
            return None
        return backend.compact(path, target_size_bytes=target_size_bytes, **opts)
    except BackendError:
        # Auto-compaction is an optimization running *after* a committed write. Failing it
        # must not fail the write — the data is already safely in the table.
        from batcher._internal.logging import get_logger

        get_logger("io").warning("auto-compaction of %s skipped: table maintenance failed", path)
        return None
