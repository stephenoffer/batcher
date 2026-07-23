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

    __slots__ = ("_catalog", "_identifier", "_mode", "_replace_where", "_token")

    def __init__(
        self,
        identifier: str,
        *,
        catalog: CatalogSpec | str | None = None,
        mode: str = "append",
        replace_where: dict | None = None,
        write_token: str | None = None,
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

    def commit(self, manifest: WriteManifest, path: str) -> None:  # noqa: ARG002
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
        """
        _require_pyiceberg()
        files = [f for f in manifest.files if f.rows]
        if not files:
            return
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            schema = _staged_schema(files[0].path)
            table = cat.create_table_if_not_exists(self._identifier, schema=schema)
            scope = self._delete_scope()
            with table.transaction() as tx:
                if scope is not None:
                    tx.delete(scope)
                tx.add_files([f.path for f in files])
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Iceberg commit to {self._identifier!r} failed: {exc}") from exc

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
