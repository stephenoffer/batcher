"""The `ds.write` namespace — typed, per-format dataset sinks.

``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
is the explicit spelling. Methods are thin wrappers over `terminal._write` and the
merge helpers; sink implementations live in `io/formats/` and register into the
`SINKS` registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.api.session import read as _read

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["Writer"]


# Save modes (Spark `SaveMode` parity). `append` is only meaningful for the
# transactional lakehouse sinks, which consume `mode` as a constructor option; the
# file sinks always overwrite, so for them `mode` only drives the existence gate.
_SAVE_MODES = ("overwrite", "error", "ignore", "append")
_MODE_AWARE_SINKS = frozenset({"delta", "iceberg", "hudi"})
# Sinks with a native transactional MERGE; others fall back to a copy-on-write
# file merge (`api.merge.compose_file_merge`).
_MERGE_NATIVE_SINKS = frozenset({"delta"})


class Writer:
    """The `ds.write` namespace: callable for autodetect, typed methods per format.

    ``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
    is the explicit spelling. All methods accept `partition_by=` (Hive directory) and
    `distributed=`/`num_workers=` (parallel shard write + atomic commit), and return
    a `WriteManifest`.
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds

    def __call__(
        self,
        path: str,
        format: str | None = None,
        *,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
        distributed: bool = False,
        num_workers: int | None = None,
        resume: bool = False,
        max_rows_per_file: int | None = None,
        replace_where: Any = None,
        **opts: Any,
    ) -> WriteManifest:
        """Execute and write the result, inferring `format` from the path when omitted.

        `mode` is the save mode (Spark ``SaveMode`` parity):

        * ``"overwrite"`` (default) — write, replacing any existing output.
        * ``"error"`` — raise `PlanError` if `path` already exists.
        * ``"ignore"`` — skip the write (return an empty manifest) if `path` exists.
        * ``"append"`` — add to an existing table; only the transactional lakehouse
          sinks (`delta`/`iceberg`/`hudi`) support it (others raise).

        ``resume=True`` makes the write idempotent: output files already present
        (necessarily fully committed, since writes are atomic) are skipped, so a job
        re-run after a crash or spot preemption finishes only the unwritten shards —
        the resumability Ray Data lacks without external bookkeeping.

        **Correctness precondition (important):** resume identifies done work by file
        *position* (``part-NNNNN``), so it is exactly-once **only when the plan is
        deterministic** — the same input produces the same rows in the same order, so
        a given part file holds the same rows on every run. This holds for the
        read → ``map_batches``/``filter``/``select`` → write (ETL / batch-inference)
        path. It does **not** hold for a plan whose row→file assignment can vary
        between runs — a ``group_by``/``join``/``sort`` or distributed shuffle, where
        ordering is hash- and worker-count-dependent. Resuming such a plan could skip
        a file that now holds *different* rows, dropping or duplicating data. For
        those, write to a fresh path (no resume) or materialize a stable, keyed
        intermediate first.
        """
        from batcher._internal.errors import PlanError
        from batcher.api.terminal import _write
        from batcher.io.detect import detect_format
        from batcher.io.manifest import WriteManifest

        if mode not in _SAVE_MODES:
            raise PlanError(f"write(): unknown mode {mode!r}; use one of {list(_SAVE_MODES)}")
        fmt = detect_format(path, format)

        # `replace_where` = dynamic partition/range overwrite (Delta `replaceWhere` /
        # the backfill pattern): atomically replace only the rows matching the
        # predicate, preserving the rest. Copy-on-write: keep the existing rows
        # *outside* the range, union the new data, overwrite. Single-writer only.
        if replace_where is not None:
            from batcher.io.filesystem import resolve_filesystem

            if resolve_filesystem(path).exists(path):
                kept = _read(path, format=fmt).filter(~replace_where)
                combined = kept.union(self._ds)
                return combined.write(
                    path,
                    fmt,
                    mode="overwrite",
                    partition_by=partition_by,
                    max_rows_per_file=max_rows_per_file,
                    **opts,
                )
        if mode == "append" and fmt not in _MODE_AWARE_SINKS:
            raise PlanError(
                f"write(): mode='append' is only supported for {sorted(_MODE_AWARE_SINKS)}, "
                f"not {fmt!r} (use a fresh path, or 'overwrite')"
            )
        # error/ignore are a pre-write existence gate (resume has its own per-file
        # idempotence, so it is exempt).
        if mode in ("error", "ignore") and not resume:
            from batcher.io.filesystem import resolve_filesystem

            if resolve_filesystem(path).exists(path):
                if mode == "error":
                    raise PlanError(f"write(): path {path!r} already exists and mode='error'")
                return WriteManifest()  # ignore: leave the existing output untouched

        # The lakehouse sinks consume append/overwrite as a constructor option; the
        # file sinks always overwrite, so `mode` only drives the gate above for them.
        sink_kwargs = dict(opts)
        if fmt in _MODE_AWARE_SINKS:
            sink_kwargs["mode"] = mode if mode in ("append", "overwrite") else "overwrite"

        # A `repartition(...)` layout (set via ds.repartition) supplies write defaults:
        # `by` Hive-partitions, `num_files`/`target_size_mb` set the per-file row cap
        # (resolved post-materialization in `_write`).
        num_files: int | None = None
        target_bytes: int | None = None
        spec = self._ds._repartition
        if spec is not None:
            if spec.by and partition_by is None:
                partition_by = list(spec.by)
            num_files = spec.num_files
            if spec.target_size_mb is not None:
                target_bytes = int(spec.target_size_mb * 1024 * 1024)

        return _write(
            self._ds._plan,
            self._ds._sources,
            self._ds.columns,
            path,
            fmt,
            partition_by=partition_by,
            distributed=distributed,
            num_workers=num_workers,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
            num_files=num_files,
            target_bytes_per_file=target_bytes,
            sink_kwargs=sink_kwargs,
        )

    def parquet(self, path: str, *, compression: str = "zstd", **opts: Any) -> WriteManifest:
        """Write as Parquet (see `__call__` for `partition_by`/`distributed`)."""
        return self(path, "parquet", compression=compression, **opts)

    def csv(self, path: str, **opts: Any) -> WriteManifest:
        """Write as CSV."""
        return self(path, "csv", **opts)

    def json(self, path: str, **opts: Any) -> WriteManifest:
        """Write as newline-delimited JSON."""
        return self(path, "json", **opts)

    def orc(self, path: str, **opts: Any) -> WriteManifest:
        """Write as ORC."""
        return self(path, "orc", **opts)

    def arrow(self, path: str, **opts: Any) -> WriteManifest:
        """Write as Arrow/Feather IPC."""
        return self(path, "arrow", **opts)

    def avro(self, path: str, **opts: Any) -> WriteManifest:
        """Write as Avro (needs ``batcher-engine[avro]``)."""
        return self(path, "avro", **opts)

    def lance(self, path: str, **opts: Any) -> WriteManifest:
        """Write a Lance dataset (needs ``batcher-engine[lance]``)."""
        return self(path, "lance", **opts)

    def msgpack(self, path: str, **opts: Any) -> WriteManifest:
        """Write as MessagePack."""
        return self(path, "msgpack", **opts)

    def merge(
        self,
        target: str,
        *,
        on: str | list[str],
        when_matched: str = "update",
        when_not_matched: str = "insert",
        format: str | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Upsert (``MERGE INTO``) this dataset into an existing `target`, keyed on `on`.

        For a transactional sink (Delta) this delegates to the native ``MERGE``. For a
        plain file target it is a copy-on-write merge: read the current target,
        reconcile with this source (``when_matched`` ∈ ``update``/``delete``,
        ``when_not_matched`` ∈ ``insert``/``ignore``), and atomically overwrite. If the
        target does not exist yet, the source is written as-is (all inserts).

        File merge is read-modify-write over the whole path — single-writer only; use a
        Delta target for concurrent writers.
        """
        from batcher.api.merge import execute_merge

        return execute_merge(
            self,
            target,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
            format=format,
            native_sinks=_MERGE_NATIVE_SINKS,
            opts=opts,
        )

    # --- Lakehouse / catalog ----------------------------------------------
    def delta(
        self,
        uri: str,
        *,
        mode: str = "append",
        merge_on: str | list[str] | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Write to a Delta Lake table (one transactional commit).

        With `merge_on`, performs a ``MERGE INTO`` upsert keyed on those columns —
        matched rows are updated and new rows inserted (Spark/Delta ``MERGE``). The
        keys build the match predicate; pass `merge_predicate=` instead for a custom
        one. Otherwise `mode` is ``"append"`` (default) or ``"overwrite"``.
        """
        if merge_on is not None:
            from batcher.api.merge import merge_predicate_for

            opts["merge_predicate"] = merge_predicate_for(merge_on)
        return self(uri, "delta", mode=mode, **opts)

    def iceberg(self, identifier: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Iceberg table (``mode="append"|"overwrite"``)."""
        return self(identifier, "iceberg", mode=mode, **opts)

    def hudi(self, table_uri: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Apache Hudi table (``mode="append"|"overwrite"``)."""
        return self(table_uri, "hudi", mode=mode, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a database table via ADBC/FlightSQL."""
        return self(table, "adbc", **opts)

    def snowflake(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a Snowflake table."""
        return self(table, "snowflake", **opts)

    def mongo(self, collection: str, **opts: Any) -> WriteManifest:
        """Write to a MongoDB collection."""
        return self(collection, "mongo", **opts)
