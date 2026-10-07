"""BigQuery sink — one load job per shard, Parquet in, job identity out.

`google-cloud-bigquery`'s `Client.load_table_from_file` with
``source_format=PARQUET`` is the supported bulk-ingest route: the shard is serialized to
Parquet in memory, submitted as one load job, and waited on. Parquet is chosen over CSV or
JSON because it carries the nested shape: with ``ParquetOptions.enable_list_inference`` a
Parquet LIST becomes a BigQuery ``REPEATED`` field rather than a record wrapping a repeated
``list.element``, and a struct becomes a ``RECORD``.

Two Arrow shapes have no BigQuery spelling, and both are refused before a job is submitted,
naming the column, rather than failing inside the job:

* an array directly inside an array (``list<list<T>>``) — BigQuery has no ``ARRAY<ARRAY>``;
  wrap the inner list in a struct;
* a NULL *element* inside an array — BigQuery arrays cannot hold one (a NULL array is fine).

The returned `WrittenFile.job` carries the load job's ``job_id``, ``location`` and
destination, so a caller can find the job in the console or audit log.

Not yet verified against a live BigQuery; see ``tests/PENDING_VERIFICATION.md``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS
from batcher.io.formats.sql._common import require_module
from batcher.io.manifest import WrittenFile

__all__ = ["BigQuerySink", "check_bigquery_nesting"]

_EXTRA = "bigquery"
_MODULE = "google.cloud.bigquery"
_MODES = ("append", "overwrite")


def check_bigquery_nesting(table: pa.Table) -> None:
    """Refuse the nested shapes a BigQuery load job cannot store, naming the column.

    Args:
        table: The rows about to be loaded.

    Raises:
        BackendError: On a list directly inside a list, or a NULL element inside a list.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.vendors.bigquery_sink import check_bigquery_nesting
            >>> check_bigquery_nesting(pa.table({"tags": [["a", "b"], []]}))
    """
    for name, column in zip(table.column_names, table.columns, strict=True):
        for chunk in column.chunks:
            _check(name, chunk)


def _check(path: str, array: pa.Array) -> None:
    """Walk one array's nested children, refusing what BigQuery cannot represent."""
    kind = array.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        child = array.flatten()
        if pa.types.is_list(child.type) or pa.types.is_large_list(child.type):
            raise BackendError(
                f"column {path!r} is an array of arrays ({kind}), which BigQuery cannot "
                "store. Wrap the inner array in a struct, e.g. list<struct<values: list<T>>>."
            )
        if child.null_count:
            raise BackendError(
                f"column {path!r} holds {child.null_count} NULL element(s) inside arrays, "
                "and a BigQuery array cannot contain NULL. Drop them with "
                "col(...).list.drop_nulls() or replace them before writing."
            )
        _check(f"{path}[]", child)
    elif pa.types.is_struct(kind):
        for index in range(kind.num_fields):
            _check(f"{path}.{kind.field(index).name}", array.field(index))


def _parquet_bytes(table: pa.Table) -> io.BytesIO:
    """The shard as a Parquet file in memory, with standard three-level lists."""
    buffer = io.BytesIO()
    pq.write_table(table, buffer, use_compliant_nested_type=True)
    buffer.seek(0)
    return buffer


@SINKS.register("bigquery")
@dataclass(frozen=True, slots=True)
class BigQuerySink:
    """Load Arrow tables into a BigQuery table through a Parquet load job.

    Not yet verified against a live BigQuery; see tests/PENDING_VERIFICATION.md.

    Args:
        project: The project the load job runs and bills in. None uses the client's
            default from the ambient ``google.auth`` credentials.
        mode: ``"append"`` (``WRITE_APPEND``) or ``"overwrite"`` (``WRITE_TRUNCATE``,
            single-shard only).
        location: The job location (``"US"``, ``"europe-west1"``), when it is not the
            dataset's default.

    Raises:
        BackendError: If `mode` is not one of the two above.
    """

    project: str | None = None
    mode: str = "append"
    location: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise BackendError(
                f"BigQuery write mode {self.mode!r} is not supported; expected one of "
                f"{list(_MODES)}."
            )

    def _job_config(self, bq: Any) -> Any:
        options = bq.ParquetOptions()
        options.enable_list_inference = True
        disposition = (
            bq.WriteDisposition.WRITE_TRUNCATE
            if self.mode == "overwrite"
            else bq.WriteDisposition.WRITE_APPEND
        )
        return bq.LoadJobConfig(
            source_format=bq.SourceFormat.PARQUET,
            write_disposition=disposition,
            parquet_options=options,
        )

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Load `table` into the BigQuery table `path` (``project.dataset.table``).

        Args:
            table: The rows to load.
            path: The destination table, ``dataset.table`` or ``project.dataset.table``.

        Returns:
            The rows the job reports loaded, and the job's identity under `job`.
        """
        check_bigquery_nesting(table)
        bq = require_module(_MODULE, extra=_EXTRA)
        client = bq.Client(project=self.project, location=self.location)
        payload = _parquet_bytes(table)
        size = payload.getbuffer().nbytes
        job = client.load_table_from_file(payload, path, job_config=self._job_config(bq))
        try:
            job.result()
        except Exception as exc:
            raise BackendError(
                f"BigQuery load job {job.job_id} into {path!r} failed: {exc}"
            ) from exc
        loaded = getattr(job, "output_rows", None)
        return WrittenFile(
            path=path,
            rows=int(loaded) if loaded is not None else table.num_rows,
            bytes=size,
            job={
                "system": "bigquery",
                "job_id": job.job_id,
                "location": getattr(job, "location", None),
                "destination": str(getattr(job, "destination", path)),
            },
        )

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - a table, not a directory
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Load one shard as its own job; refuse an overwrite that more than one shard runs.

        ``WRITE_TRUNCATE`` replaces the table per job, so on a distributed write each shard
        would erase the ones before it and the table would hold only the last.

        Raises:
            BackendError: If ``mode="overwrite"`` meets a multi-shard write.
        """
        if file_index > 0 and self.mode == "overwrite":
            raise BackendError(
                f"mode='overwrite' cannot be used for a distributed write to BigQuery table "
                f"{path!r}: each shard's load job would truncate the table. Use "
                "mode='append' into an emptied table, or write single-node."
            )
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: each load job commits its own rows atomically."""
