"""Delta table maintenance: OPTIMIZE, ZORDER, VACUUM, and log checkpointing.

delta-rs performs all of these as real transactions, which is the whole point: each one
commits a new version whose `remove` actions retire the old files *from the log* while
leaving them on storage, so every existing version still reads and time travel survives.
The file-level compactor cannot do that — it deletes what it replaces — which is why a
lakehouse table must never be maintained that way.

The three operations are different jobs and it matters not to confuse them:

* `compact` bin-packs small files. It reduces the file count a query must plan over, which
  is the fixed cost that dominates a scan of a table built by frequent small appends.
* `z_order` bin-packs *and* sorts the rows along a Z-curve over the given columns. Sorting
  is what makes each output file's min/max bounds narrow, and narrow bounds are exactly
  what lets the next query's predicate rule a file out from the log alone
  (`io.stats.file_skipping`). Clustering and skipping are one mechanism: z-ordering is how
  you *create* the skipping opportunity that the reader then exploits. A single sort key
  would do this for one column; the Z-curve interleaves several so that filters on *any*
  of them stay selective.
* `vacuum` is the only operation that deletes, and it deletes only files no live version
  references, older than a retention window. The window exists so an in-flight reader that
  planned against an older snapshot does not have its files pulled out from under it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.formats.lakehouse.maintenance import MAINTENANCE

__all__ = ["DeltaMaintenance"]

# delta-rs refuses a retention shorter than the table's configured
# `deletedFileRetentionDuration` unless the check is explicitly waived, because a short
# window can delete files a concurrent reader is still using. We surface that as an
# opt-in rather than waiving it silently.
_DEFAULT_RETENTION_HOURS = 168.0  # 7 days — Delta's own default


@MAINTENANCE.register("delta")
class DeltaMaintenance:
    """Transactional maintenance for a Delta table."""

    __slots__ = ()

    @staticmethod
    def _table(path: str, opts: dict[str, Any]) -> Any:
        deltalake = require_deltalake()
        storage_options = opts.get("storage_options")
        try:
            return deltalake.DeltaTable(path, storage_options=storage_options)
        except Exception as exc:
            raise BackendError(f"failed to open Delta table {path!r}: {exc}") from exc

    def compact(
        self,
        path: str,
        *,
        target_size_bytes: int | None = None,
        z_order: list[str] | None = None,
        where: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """Bin-pack (and optionally Z-order) the table's data files, as one commit.

        The old files are removed from the log, not from storage, so every prior version
        still reads — `vacuum` is what eventually reclaims them. `where` restricts the
        work to matching partitions (a daily job compacts today, not the whole table);
        `z_order` sorts the rewritten rows along a Z-curve over those columns, which
        narrows each file's bounds and is what makes the *next* query's file skipping
        effective.

        Args:
            path: The table root.
            target_size_bytes: Target size for each rewritten file.
            z_order: Columns to Z-order by; plain bin-packing when omitted.
            where: Partition filters, as ``[(col, op, value)]`` tuples, limiting the scope.
            **opts: ``storage_options`` and any delta-rs optimizer options.

        Returns:
            delta-rs's optimize metrics (files added/removed, partitions touched).

        Raises:
            BackendError: If the table cannot be opened or the rewrite fails.
        """
        table = self._table(path, opts)
        kwargs: dict[str, Any] = {}
        if target_size_bytes is not None:
            kwargs["target_size"] = target_size_bytes
        if where is not None:
            kwargs["partition_filters"] = where
        try:
            if z_order:
                return dict(table.optimize.z_order(z_order, **kwargs))
            return dict(table.optimize.compact(**kwargs))
        except Exception as exc:
            what = "z-order" if z_order else "compaction"
            raise BackendError(f"Delta {what} of {path!r} failed: {exc}") from exc

    def small_file_count(self, path: str, *, below_bytes: int, **opts: Any) -> int:
        """How many of the table's live data files are below `below_bytes`.

        Read straight from the add actions — the log already records every file's size, so
        deciding whether a table needs compacting costs one metadata read and never opens a
        data file. This is what makes an auto-compaction check cheap enough to run after
        every write.

        Args:
            path: The table root.
            below_bytes: The size under which a file counts as small.
            **opts: ``storage_options``.

        Returns:
            The number of live data files smaller than `below_bytes`.
        """
        import pyarrow.compute as pc

        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        snapshot = open_snapshot(path, storage_options=opts.get("storage_options"))
        manifest = snapshot.add_actions()
        if "size_bytes" not in manifest.column_names:
            return 0
        return int(pc.sum(pc.less(manifest.column("size_bytes"), below_bytes)).as_py() or 0)

    def vacuum(
        self,
        path: str,
        *,
        retention_hours: float | None = None,
        dry_run: bool = True,
        **opts: Any,
    ) -> list[str]:
        """Delete data files no live version references and older than the retention window.

        Defaults to a **dry run**: it reports what it would delete and deletes nothing.
        That default is deliberate — this is the one maintenance operation that destroys
        data, and the files it removes are exactly the ones time travel and any in-flight
        reader depend on. Pass ``dry_run=False`` to actually reclaim.

        The retention window is the safety argument: a file is only removed once it has
        been unreferenced for longer than any reader could plausibly still be using it.
        Shortening it below the table's configured minimum requires
        ``enforce_retention_duration=False``, which we make you pass explicitly rather
        than waive for you.

        Args:
            path: The table root.
            retention_hours: How long an unreferenced file is kept (default 7 days).
            dry_run: When True (the default), report but do not delete.
            **opts: ``storage_options``, ``enforce_retention_duration``.

        Returns:
            The files deleted (or, on a dry run, the files that would be).

        Raises:
            BackendError: If the table cannot be opened or the vacuum fails.
        """
        table = self._table(path, opts)
        hours = _DEFAULT_RETENTION_HOURS if retention_hours is None else retention_hours
        try:
            return list(
                table.vacuum(
                    retention_hours=int(hours),
                    dry_run=dry_run,
                    enforce_retention_duration=opts.get("enforce_retention_duration", True),
                )
            )
        except Exception as exc:
            raise BackendError(f"Delta vacuum of {path!r} failed: {exc}") from exc
