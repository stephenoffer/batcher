"""The storage contract a `Catalog` is built over, and nothing about naming or modes.

A backend answers four questions about storage — which namespaces exist, which tables a
namespace holds, what a table's rows and properties are — and performs four primitive
writes. Everything a user sees on top of that (dotted-name resolution, the save modes,
``if_not_exists``, cascade, alignment of an incoming dataset to a table's schema) lives once
in `Catalog`, so the three backends cannot disagree about what ``mode="error"`` means.

The primitive write modes are deliberately fewer than the user-facing save modes:

``"create"``
    The table does not exist; write it with this schema, partitioning and properties.
``"append"``
    The table exists and the rows are already aligned to its schema; add them.
``"overwrite"``
    The table exists; replace its rows *and* its schema with the incoming dataset.
``"replace_where"``
    The table exists and the rows are aligned; atomically replace the rows matching
    `replace_where` and keep the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["CatalogBackend"]


class CatalogBackend(Protocol):
    """Storage primitives behind a `Catalog`: namespaces, tables, reads and writes."""

    #: The namespace a one-part table name resolves into, and one that cannot be dropped.
    default_namespace: str

    def list_namespaces(self) -> list[str]:
        """Every namespace, as a dotted name."""
        ...

    def create_namespace(self, namespace: str) -> None:
        """Create `namespace`, which does not exist yet."""
        ...

    def drop_namespace(self, namespace: str) -> None:
        """Remove `namespace`, which exists and holds no tables."""
        ...

    def list_tables(self, namespace: str) -> list[str]:
        """The table names in `namespace`, unqualified."""
        ...

    def has_table(self, namespace: str, table: str) -> bool:
        """Whether `namespace.table` exists."""
        ...

    def read_table(self, namespace: str, table: str) -> Dataset:
        """A lazy `Dataset` over the table's current snapshot."""
        ...

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
        """Perform one primitive write (see the module docstring for `mode`)."""
        ...

    def drop_table(self, namespace: str, table: str) -> None:
        """Remove the table, which exists."""
        ...

    def table_properties(self, namespace: str, table: str) -> dict[str, str]:
        """The table's key-value properties."""
        ...

    def partition_columns(self, namespace: str, table: str) -> list[str]:
        """The columns the table is partitioned by, in order (empty when unpartitioned)."""
        ...
