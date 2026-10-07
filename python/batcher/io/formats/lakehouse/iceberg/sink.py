"""Writing an Iceberg table: workers stage data files, the driver commits one snapshot."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.catalog import CatalogSpec, resolve_catalog
from batcher.io.formats.base import SINKS
from batcher.io.formats.lakehouse.iceberg._common import (
    _new_write_token,
    _require_pyiceberg,
    _staged_schema,
)
from batcher.io.manifest import WriteManifest, WrittenFile

__all__ = ["IcebergSink"]

#: Snapshot-summary keys recording a streaming micro-batch's transaction.
_APP_ID = "batcher.stream.app-id"
_TXN_VERSION = "batcher.stream.txn-version"


def _txn_property(app_id: str) -> str:
    """The table property holding `app_id`'s highest committed micro-batch.

    The snapshot summary alone is not durable enough to be the marker: snapshot expiry
    removes it, and after any later commit that does not carry it (a compaction, another
    writer) the snapshot that did becomes expirable. Table properties survive expiry.
    """
    return f"batcher.stream.{app_id}.txn-version"


def _record_txn(tx: Any, app_id: str, version: int) -> None:
    """Record a micro-batch in the table properties, inside the commit's transaction.

    The same transaction as the data, so the marker and the rows land or fail together.
    The recorded version only ever rises: a stale writer committing an older batch must
    not make a newer one look uncommitted.
    """
    key = _txn_property(app_id)
    try:
        recorded = int(tx.table_metadata.properties.get(key, "-1"))
    except ValueError:
        recorded = -1
    if version > recorded:
        tx.set_properties({key: str(version)})


@SINKS.register("iceberg")
class IcebergSink:
    """Scalable append/overwrite writer for an Iceberg table (one driver-side snapshot).

    Each worker writes its shard as a **real** Parquet file into a staging area under the
    catalog warehouse (parallel, shared-nothing, bounded per-worker memory) and returns
    only the file locator — no shard data flows through the driver. `commit` registers
    every staged file with the table in one snapshot via ``add_files`` (the data files are
    referenced in place, never re-read or re-written by the driver). This replaces the old
    buffer-everything design, which ``pa.concat_tables``-ed the whole result on the driver
    and — on the distributed path — silently wrote nothing (a worker's in-memory buffer
    never reached the committing driver sink).

    Staged file names carry a per-write `token` so a later write cannot clobber a file a
    prior snapshot still references; the name is otherwise deterministic in the shard index,
    so a preempted-and-rerun shard overwrites its own file (idempotent). Merge-on-read /
    equality-delete writes are not supported (pyiceberg's support is immature).

    Args:
        identifier: The table identifier (``namespace.table``).
        catalog: A catalog spec (name or property mapping; see `io.catalog`).
        mode: ``"append"`` (default) or ``"overwrite"``.
    """

    __slots__ = ("_app_txn", "_catalog", "_identifier", "_mode", "_replace_where", "_token")

    def __init__(
        self,
        identifier: str,
        *,
        catalog: CatalogSpec | str | None = None,
        mode: str = "append",
        replace_where: dict | None = None,
        write_token: str | None = None,
        app_id: str | None = None,
        txn_version: int | None = None,
    ) -> None:
        if mode not in ("append", "overwrite"):
            raise BackendError(f"unsupported Iceberg write mode {mode!r}; use append/overwrite")
        self._identifier = identifier
        self._catalog = catalog
        self._mode = mode
        self._replace_where = replace_where
        # Per-write token shared across the workers (injected via the sink kwargs) so all
        # shards of one write share it and it differs between writes; falls back to a
        # locally-derived token for a direct single-process construction.
        self._token = write_token or _new_write_token()
        # A streaming micro-batch's `(app_id, batch_id)`, recorded in the snapshot summary so
        # a replayed batch finds itself committed: Delta's `txn` action, in the one place an
        # Iceberg snapshot carries caller metadata.
        self._app_txn = (
            (app_id, txn_version) if app_id is not None and txn_version is not None else None
        )

    def _staging(self) -> str:
        """The staging directory for this write, under the catalog warehouse so every
        worker (any node) writes to shared storage the driver's commit can reference."""
        from batcher.io.formats.lakehouse._staging import staging_root

        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        warehouse = cat.properties.get("warehouse", "").rstrip("/")
        safe_id = self._identifier.replace("/", ".")
        return staging_root(f"{warehouse}/{safe_id}")

    def write(self, table: pa.Table, path: str, *, resume: bool = False) -> WrittenFile:  # noqa: ARG002
        # `resume` matches the common `FileSink.write` signature; ignored — an Iceberg
        # write is one atomic snapshot commit, not idempotent per-file shard writes.
        from batcher.io.formats.lakehouse._staging import stage_shard

        return stage_shard(table, self._staging(), file_index=0, token=self._token)

    def write_stream(
        self,
        batches: Any,
        path: str,  # noqa: ARG002
        *,
        schema: pa.Schema | None = None,
        resume: bool = False,  # noqa: ARG002
    ) -> WrittenFile:
        """Stream `batches` into one staged Parquet file (bounded memory) for the commit."""
        from batcher.io.formats.lakehouse._staging import stage_stream

        return stage_stream(
            batches, self._staging(), schema=schema, file_index=0, token=self._token
        )

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,  # noqa: ARG002
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write one shard as data file(s) laid out to the table's own partition spec.

        `partition_by` is ignored, and correctly so: an Iceberg table's partitioning is a
        property of the table, declared in the catalog's partition spec, not of the write.
        But it does not follow that the writer can ignore *partitioning* — and it used to.
        A shard was staged as one flat Parquet file, and the commit's ``add_files`` infers a
        file's partition from its column statistics, so any shard spanning more than one
        partition value was rejected outright::

            Cannot infer partition value ... more than one partition values
            for Partition Field: cat. lower_value='a', upper_value='b'

        A partitioned Iceberg table was therefore unwritable. The shard is now handed to
        pyiceberg's own writer, which splits it along the table's spec — applying each
        partition field's transform, assigning field ids, and collecting metrics — and emits
        one data file per partition. Each file then has a single partition value, which is
        exactly what the commit needs to place it.

        An **unpartitioned** table keeps the staging path: there is nothing to split, and
        staging keeps the deterministic per-shard file name a preempted worker overwrites.
        """
        if not self._partition_fields():
            from batcher.io.formats.lakehouse._staging import stage_shard

            return [stage_shard(table, self._staging(), file_index=file_index, token=self._token)]
        return self._write_partitioned_files(table)

    @property
    def partitions_itself(self) -> bool:
        """Whether this table owns its own partitioning, so a write must lay it out.

        An Iceberg table declares its partitioning in the catalog's spec, so it never
        arrives as a `partition_by` argument at the call site. The write path keys off that
        argument to decide whether to produce a directory of files or one flat file — so
        without this, a partitioned table took the flat-file branch and the commit rejected
        what it produced. The sink has to say what the write cannot know.
        """
        return bool(self._partition_fields())

    def _partition_fields(self) -> list[Any]:
        """The table's partition fields, or `[]` if it is unpartitioned (or absent)."""
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            return list(cat.load_table(self._identifier).spec().fields)
        except Exception:
            return []  # a table that does not exist yet is created unpartitioned

    def _write_partitioned_files(self, table: pa.Table) -> list[WrittenFile]:
        """Split `table` along the table's partition spec and write one data file each."""
        from pyiceberg.io.pyarrow import _dataframe_to_data_files

        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        target = cat.load_table(self._identifier)
        try:
            aligned = table.cast(target.schema().as_arrow())
            files = list(
                _dataframe_to_data_files(table_metadata=target.metadata, df=aligned, io=target.io)
            )
        except Exception as exc:
            raise BackendError(
                f"failed to write partitioned data files for Iceberg table "
                f"{self._identifier!r}: {exc}"
            ) from exc
        return [
            WrittenFile(
                path=f.file_path,
                rows=int(f.record_count),
                bytes=int(f.file_size_in_bytes),
            )
            for f in files
        ]

    def commit(self, manifest: WriteManifest, path: str) -> int | None:  # noqa: ARG002
        """Register all staged files with the table in ONE snapshot.

        Two properties, both of which were missing.

        **It is one transaction.** An overwrite used to call `delete(AlwaysTrue())` and then
        `add_files` as two separate commits, so between them the table was *committed* empty
        — a concurrent reader saw zero rows, and a driver that died in the gap left it that
        way permanently. Both now happen inside one transaction, so the table goes from its
        old contents to its new ones with nothing observable in between.

        **`replace_where` replaces only what it matches.** It used to be dropped entirely on
        the Iceberg path (the writer's fallback tests `exists(path)`, and an Iceberg "path"
        is a catalog identifier, not a file), so the write ran as a plain overwrite and
        *deleted the rest of the table*. Scoping the delete to the predicate is what makes a
        backfill a backfill instead of a wipe.

        `add_files` references the staged Parquet directly — the driver never re-reads or
        re-writes the data.

        **An empty write still deletes.** A write whose input produced no rows used to return
        before the transaction, so an overwrite or a `replace_where` of nothing left every
        old row in place: the table kept contents the write said to replace. Only an append
        of nothing is a no-op; a scoped write of nothing commits its delete. A table that
        does not exist has nothing to delete, and with no staged file there is no schema to
        create it from, so that case alone still commits nothing.

        Returns the snapshot id the commit created, which the writer reports as
        `WriteManifest.version`, or None when there was nothing to commit.
        """
        _require_pyiceberg()
        files = [f for f in manifest.files if f.rows]
        try:
            scope = self._delete_scope()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Iceberg commit to {self._identifier!r} failed: {exc}") from exc
        if not files and scope is None:
            return None
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            if files:
                schema = _staged_schema(files[0].path)
                table = cat.create_table_if_not_exists(self._identifier, schema=schema)
            elif cat.table_exists(self._identifier):
                table = cat.load_table(self._identifier)
            else:
                return None
            marker = {}
            if self._app_txn is not None:
                marker = {_APP_ID: self._app_txn[0], _TXN_VERSION: str(self._app_txn[1])}
            with table.transaction() as tx:
                if scope is not None:
                    tx.delete(scope, snapshot_properties=marker)
                if files:
                    tx.add_files([f.path for f in files], snapshot_properties=marker)
                if self._app_txn is not None:
                    _record_txn(tx, *self._app_txn)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Iceberg commit to {self._identifier!r} failed: {exc}") from exc
        # The committing handle's metadata, refreshed by its own transaction -- not a
        # catalog re-read, which could return a concurrent writer's later snapshot.
        snapshot = table.current_snapshot()
        return None if snapshot is None else snapshot.snapshot_id

    def is_committed(self, path: str) -> bool:  # noqa: ARG002 - the identifier is the path
        """Whether this write's ``(app_id, batch_id)`` is already recorded in the table.

        A streaming micro-batch is committed when the table records this app id with a
        transaction version at or past this batch's, the rule Delta's `txn` uses. The table
        property is consulted first, because it survives snapshot expiry; the snapshot
        summaries are still read so a table committed before the property existed keeps its
        markers. With no transaction configured there is nothing to find, so the write
        proceeds. A table that does not exist yet has committed nothing.

        Args:
            path: Unused: an Iceberg destination is its catalog identifier.

        Returns:
            True when this exact micro-batch was already committed.
        """
        if self._app_txn is None:
            return False
        app_id, version = self._app_txn
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            table = cat.load_table(self._identifier)
        except Exception:
            return False
        try:
            if int(table.properties.get(_txn_property(app_id), "-1")) >= version:
                return True
        except ValueError:
            pass  # an unreadable property proves nothing; fall through to the summaries
        for snapshot in table.snapshots():
            props = snapshot.summary.additional_properties if snapshot.summary else {}
            if props.get(_APP_ID) != app_id:
                continue
            try:
                if int(props.get(_TXN_VERSION, "-1")) >= version:
                    return True
            except ValueError:
                continue
        return False

    def _delete_scope(self) -> Any:
        """What this commit removes before adding its files: nothing, a predicate, or all.

        An append removes nothing. A `replace_where` removes exactly the rows its predicate
        matches. A plain overwrite removes everything — which is the mode's meaning, and
        precisely what `replace_where` must *not* be allowed to collapse into.
        """
        if self._replace_where is not None:
            from batcher.io.predicate import to_iceberg_expression

            expression = to_iceberg_expression(self._replace_where)
            if expression is None:
                raise BackendError(
                    "write(replace_where=...) on an Iceberg table needs a predicate the "
                    "table can express (comparisons, AND/OR, null tests over its columns). "
                    "Refusing rather than overwriting the whole table."
                )
            return expression
        if self._mode == "overwrite":
            from pyiceberg.expressions import AlwaysTrue

            return AlwaysTrue()
        return None
