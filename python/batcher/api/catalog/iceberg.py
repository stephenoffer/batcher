"""The Iceberg catalog backend: any pyiceberg catalog, reached through its own API.

pyiceberg already abstracts the Iceberg catalog services behind one `Catalog` class — REST
(which is how Unity Catalog, Polaris, Gravitino and S3 Tables expose Iceberg), Glue, Hive,
SQL and DynamoDB — so this backend is written against that class and not against any one
service. Namespaces and table lifecycle are pyiceberg calls; reads and writes go through
`bt.read.iceberg` / `ds.write.iceberg` with the live catalog object, so a catalog table
gets exactly the snapshot commits and pushdown those already have.

Two properties follow from pyiceberg rather than from here. Dropping a table removes its
catalog entry and leaves the data files, which is Iceberg's ``DROP TABLE`` without
``PURGE``. And a table's partitioning is declared on the table, so a create that names
`partition_by` is refused rather than silently written unpartitioned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["IcebergBackend"]


class IcebergBackend:
    """Namespaces and tables of one live pyiceberg catalog."""

    default_namespace = "default"

    def __init__(self, catalog: Any) -> None:
        """Wrap a live pyiceberg catalog.

        Args:
            catalog: A `pyiceberg.catalog.Catalog`.
        """
        self._catalog = catalog

    def list_namespaces(self) -> list[str]:
        return [".".join(ns) for ns in self._catalog.list_namespaces()]

    def create_namespace(self, namespace: str) -> None:
        self._catalog.create_namespace(namespace)

    def drop_namespace(self, namespace: str) -> None:
        self._catalog.drop_namespace(namespace)

    def list_tables(self, namespace: str) -> list[str]:
        return [ident[-1] for ident in self._catalog.list_tables(namespace)]

    def has_table(self, namespace: str, table: str) -> bool:
        return bool(self._catalog.table_exists(f"{namespace}.{table}"))

    def read_table(self, namespace: str, table: str) -> Dataset:
        from batcher.api.io_namespace.reader import read

        return read.iceberg(f"{namespace}.{table}", catalog=self._catalog)

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
        identifier = f"{namespace}.{table}"
        if mode == "overwrite" and self.read_table(namespace, table).schema != data.schema:
            # Recreated rather than evolved: a replace takes the new schema whole. The old
            # properties are not carried, because pyiceberg records the old schema's field ids
            # in them (`schema.name-mapping.default`) and a new table read through that
            # mapping cannot find its own columns.
            self._catalog.drop_table(identifier)
            mode = "create"
        if mode == "create":
            if partition_by:
                raise PlanError(
                    f"creating Iceberg table {identifier!r} with partition_by={partition_by!r} "
                    "is not supported: Iceberg partitioning is a partition spec with "
                    "transforms, declared on the table",
                    hint="Create the table with pyiceberg's partition_spec=, then write to it.",
                )
            self._catalog.create_table(identifier, schema=data.schema, properties=properties or {})
            mode = "append"
        if mode == "replace_where":
            return data.write.iceberg(
                identifier, catalog=self._catalog, replace_where=replace_where
            )
        return data.write.iceberg(identifier, catalog=self._catalog, mode=mode)

    def drop_table(self, namespace: str, table: str) -> None:
        self._catalog.drop_table(f"{namespace}.{table}")

    def table_properties(self, namespace: str, table: str) -> dict[str, str]:
        return dict(self._catalog.load_table(f"{namespace}.{table}").properties)

    def partition_columns(self, namespace: str, table: str) -> list[str]:
        loaded = self._catalog.load_table(f"{namespace}.{table}")
        schema = loaded.schema()
        return [
            schema.find_field(field.source_id).name
            for field in loaded.spec().fields
            if str(field.transform) == "identity"
        ]
