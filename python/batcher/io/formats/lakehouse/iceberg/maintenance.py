"""Iceberg table maintenance: snapshot expiry, and an honest refusal to compact.

Iceberg's maintenance story is not Delta's, and the difference is worth stating plainly
rather than papering over:

* **Snapshot expiry works, and is the important one.** Every write leaves a snapshot, and
  each one pins the data files it references. A table written by a streaming query
  accumulates them without bound, so expiring old snapshots is what actually lets storage
  be reclaimed — it is Iceberg's `vacuum`, and it is what `bt.vacuum` calls here.

* **Compaction does not, because the client cannot do it.** Rewriting data files is
  `rewrite_data_files`, a Spark procedure; pyiceberg 0.11 has no equivalent, and there is
  no way to synthesize one that is both correct and bounded. So `compact` raises and says
  so. That refusal is deliberate: a maintenance call that silently does nothing is worse
  than one that fails, because the user comes away believing the table was compacted and
  the small-file problem is still there. The `sort_by=` write option is the way to get
  well-clustered files out of Iceberg from here — cluster on the way in, since we cannot
  re-cluster after the fact.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.catalog import resolve_catalog
from batcher.io.formats.lakehouse.maintenance import MAINTENANCE, unsupported

__all__ = ["IcebergMaintenance"]

_DEFAULT_RETENTION_HOURS = 120.0  # 5 days — Iceberg's own default snapshot age


@MAINTENANCE.register("iceberg")
class IcebergMaintenance:
    """Transactional maintenance for an Iceberg table."""

    __slots__ = ()

    @staticmethod
    def _table(identifier: str, opts: dict[str, Any]) -> Any:
        catalog = resolve_catalog(opts.get("catalog") or "default")
        try:
            return catalog.load_table(identifier)
        except Exception as exc:
            raise BackendError(f"failed to load Iceberg table {identifier!r}: {exc}") from exc

    # The arguments are the shared `TableMaintenance` protocol's; this backend cannot
    # honor any of them, which is exactly what it raises to say.
    def compact(
        self,
        path: str,  # noqa: ARG002
        *,
        target_size_bytes: int | None = None,  # noqa: ARG002
        z_order: list[str] | None = None,  # noqa: ARG002
        where: str | None = None,  # noqa: ARG002
        **opts: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Not available: rewriting Iceberg data files needs Spark's `rewrite_data_files`.

        Raises:
            BackendError: Always. See the module docstring on why this refuses rather than
                pretending.
        """
        raise unsupported(
            "compaction",
            "iceberg",
            "rewriting data files is Spark's `rewrite_data_files` procedure, and pyiceberg "
            "does not implement it. Cluster the data on the way in with "
            "ds.write.iceberg(..., sort_by=[...]), or compact with Spark",
        )

    def vacuum(
        self,
        path: str,
        *,
        retention_hours: float | None = None,
        dry_run: bool = True,
        **opts: Any,
    ) -> list[str]:
        """Expire snapshots older than the retention window, releasing the files they pin.

        Iceberg's counterpart to Delta's vacuum. A snapshot keeps every data file it
        references alive, so expiring the old ones is what actually makes storage
        reclaimable. Defaults to a dry run, like every operation here that destroys
        history.

        Args:
            path: The table identifier (``namespace.table``).
            retention_hours: How old a snapshot must be to expire (default 5 days).
            dry_run: When True (the default), report the snapshots but expire nothing.
            **opts: ``catalog`` selects the catalog to resolve against.

        Returns:
            The snapshot ids expired — or, on a dry run, those that would be.

        Raises:
            BackendError: If the table cannot be loaded or the expiry fails.
        """
        import datetime as dt

        table = self._table(path, opts)
        hours = _DEFAULT_RETENTION_HOURS if retention_hours is None else retention_hours
        current = table.current_snapshot()
        current_id = current.snapshot_id if current is not None else None

        cutoff_ms = _cutoff_millis(hours, dt)
        expiring = [
            snapshot.snapshot_id
            for snapshot in table.snapshots()
            if snapshot.snapshot_id != current_id and snapshot.timestamp_ms < cutoff_ms
        ]
        if dry_run or not expiring:
            return [str(sid) for sid in expiring]
        try:
            table.maintenance.expire_snapshots().expire_snapshots_older_than(cutoff_ms).commit()
        except Exception as exc:
            raise BackendError(f"Iceberg snapshot expiry for {path!r} failed: {exc}") from exc
        return [str(sid) for sid in expiring]


def _cutoff_millis(hours: float, dt: Any) -> int:
    """The epoch-millis timestamp `hours` in the past — anything older may be expired."""
    moment = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    return int(moment.timestamp() * 1000)
