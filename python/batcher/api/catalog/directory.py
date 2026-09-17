"""The directory catalog backend: a warehouse path of Delta tables, one directory per level.

The layout is ``<root>/<namespace>/<table>/``, where the table directory is a Delta table
(it holds a ``_delta_log``) and a one-part name lands in ``main``. It is not the Hive
warehouse layout (``<db>.db/<table>``, default-database tables at the root), so a Spark
warehouse is not read as-is. A namespace is a directory and nothing more, so there is
no catalog service to run and nothing to keep in sync with the files — listing the catalog
*is* listing the directory.

Every read and write goes through `bt.read.delta` / `ds.write.delta`, so a catalog table
gets the transactional commits, time travel and partition layout those already have. The
one thing added here is dropping: a table directory is removed recursively, and a namespace
directory only when nothing is left in it, so pointing a catalog at a directory that also
holds other files can never delete them through `drop_namespace`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.optional import require

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["DirectoryBackend"]

_LOG = "_delta_log"


def _deltalake() -> Any:
    return require("deltalake", feature="a directory catalog", provides="deltalake", extra="delta")


class DirectoryBackend:
    """Delta tables laid out as ``<root>/<namespace>/<table>``."""

    default_namespace = "main"

    def __init__(self, root: str) -> None:
        """Bind to a warehouse root, which need not exist until the first write.

        Args:
            root: A local path or a filesystem URI (``s3://bucket/warehouse``).
        """
        import pyarrow.fs as pafs

        uri = root.rstrip("/")
        if "://" not in uri:
            uri = os.path.abspath(uri)
        self._root = uri
        self._fs, self._base = pafs.FileSystem.from_uri(uri)

    # --- paths ---------------------------------------------------------------
    def _uri(self, *parts: str) -> str:
        return "/".join((self._root, *parts))

    def _path(self, *parts: str) -> str:
        return "/".join((self._base, *parts))

    def _subdirs(self, path: str) -> list[str]:
        import pyarrow.fs as pafs

        selector = pafs.FileSelector(path, allow_not_found=True)
        return sorted(
            info.base_name
            for info in self._fs.get_file_info(selector)
            if info.type == pafs.FileType.Directory and not info.base_name.startswith(("_", "."))
        )

    def _is_table(self, *parts: str) -> bool:
        import pyarrow.fs as pafs

        return self._fs.get_file_info(self._path(*parts, _LOG)).type == pafs.FileType.Directory

    # --- namespaces ----------------------------------------------------------
    def list_namespaces(self) -> list[str]:
        found = [d for d in self._subdirs(self._base) if not self._is_table(d)]
        return sorted({self.default_namespace, *found})

    def create_namespace(self, namespace: str) -> None:
        self._fs.create_dir(self._path(namespace), recursive=True)

    def drop_namespace(self, namespace: str) -> None:
        import pyarrow.fs as pafs

        path = self._path(namespace)
        if self._fs.get_file_info(path).type == pafs.FileType.NotFound:
            return
        leftover = self._fs.get_file_info(pafs.FileSelector(path))
        if leftover:
            raise PlanError(
                f"namespace directory {self._uri(namespace)!r} still holds "
                f"{len(leftover)} entr{'y' if len(leftover) == 1 else 'ies'} that are not "
                "catalog tables, so it was not deleted",
                hint="Move or remove those files, then drop the namespace again.",
            )
        self._fs.delete_dir(path)

    # --- tables --------------------------------------------------------------
    def list_tables(self, namespace: str) -> list[str]:
        return [t for t in self._subdirs(self._path(namespace)) if self._is_table(namespace, t)]

    def has_table(self, namespace: str, table: str) -> bool:
        return self._is_table(namespace, table)

    def read_table(self, namespace: str, table: str) -> Dataset:
        from batcher.api.io_namespace.reader import read

        return read.delta(self._uri(namespace, table))

    def write_table(
        self,
        namespace: str,
        table: str,
        data: Dataset,
        *,
        mode: str,
        partition_by: list[str] | None,
        properties: dict[str, str] | None,
        replace_where: Any,
    ) -> WriteManifest:
        uri = self._uri(namespace, table)
        if mode == "append":
            return data.write.delta(uri, mode="append")
        if mode == "replace_where":
            return data.write.delta(uri, replace_where=replace_where)
        if mode == "overwrite" and self.read_table(namespace, table).schema != data.schema:
            return self._replace_schema(uri, namespace, table, data, partition_by, properties)
        return data.write.delta(
            uri,
            mode="overwrite",
            partition_by=partition_by or None,
            table_properties=properties or None,
        )

    def _replace_schema(
        self,
        uri: str,
        namespace: str,
        table: str,
        data: Dataset,
        partition_by: list[str] | None,
        properties: dict[str, str] | None,
    ) -> WriteManifest:
        """Overwrite a table with rows of a different schema, as one more table version.

        A Delta overwrite keeps the table's schema and refuses a different one, which is
        right for a path write and wrong for a table *replace*, whose point is the new
        schema. Deleting and recreating the directory would restart the log at version 0,
        which every process that cached the old version 0 would then serve instead. So the
        schema is replaced by an empty ``schema_mode="overwrite"`` commit and the rows are
        appended after it: two versions, both newer than anything a reader has cached.
        """
        kept = self.partition_columns(namespace, table)
        if not partition_by and set(kept) <= set(data.columns):
            partition_by = kept
        _deltalake().write_deltalake(
            uri,
            data.schema.empty_table(),
            mode="overwrite",
            schema_mode="overwrite",
            partition_by=partition_by or [],
        )
        return data.write.delta(uri, mode="append", table_properties=properties or None)

    def drop_table(self, namespace: str, table: str) -> None:
        from batcher.io.formats.lakehouse.delta._snapshot import forget_table

        self._fs.delete_dir(self._path(namespace, table))
        forget_table(self._uri(namespace, table))

    def table_properties(self, namespace: str, table: str) -> dict[str, str]:
        metadata = _deltalake().DeltaTable(self._uri(namespace, table)).metadata()
        return dict(metadata.configuration)

    def partition_columns(self, namespace: str, table: str) -> list[str]:
        metadata = _deltalake().DeltaTable(self._uri(namespace, table)).metadata()
        return list(metadata.partition_columns)
