"""The in-memory catalog backend: tables held as Arrow in this process.

This is the catalog a fresh `Session` starts with (named ``memory``, namespace ``main``, as
DuckDB's are) and what `Catalog.from_pydict` builds. A write is a terminal operation here as
everywhere else, so a table holds the rows the write *computed* rather than the plan that
produced them: overwriting a source after ``save`` does not change the saved table. A table
seeded by `Catalog.from_pydict` from a lazy `Dataset` stays lazy until something writes to
it, which is what lets a catalog over large file reads be built without reading them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import SchemaError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["MemoryBackend"]


@dataclass
class _Entry:
    """One table: its rows, and the metadata a write declared."""

    data: Dataset
    properties: dict[str, str] = field(default_factory=dict)
    partition_by: list[str] = field(default_factory=list)


def _materialize(data: Dataset) -> Dataset:
    """Run `data` and hold the result, so the table no longer depends on its source."""
    from batcher.api.session.frames import from_arrow

    return from_arrow(data.collect())


def _manifest(rows: int) -> WriteManifest:
    """A manifest for an in-memory write: one logical file carrying the row count."""
    from batcher.io.manifest import WriteManifest, WrittenFile

    return WriteManifest((WrittenFile("memory", rows, 0),))


class MemoryBackend:
    """Namespaces of Arrow-backed tables living in this process."""

    default_namespace = "main"

    def __init__(self, tables: dict[tuple[str, str], Dataset] | None = None) -> None:
        """Start with the default namespace and any seeded tables.

        Args:
            tables: ``{(namespace, table): dataset}`` to seed, kept lazy.
        """
        self._namespaces: dict[str, dict[str, _Entry]] = {self.default_namespace: {}}
        for (namespace, table), data in (tables or {}).items():
            self._namespaces.setdefault(namespace, {})[table] = _Entry(data)

    def list_namespaces(self) -> list[str]:
        return list(self._namespaces)

    def create_namespace(self, namespace: str) -> None:
        self._namespaces[namespace] = {}

    def drop_namespace(self, namespace: str) -> None:
        del self._namespaces[namespace]

    def list_tables(self, namespace: str) -> list[str]:
        return list(self._namespaces.get(namespace, {}))

    def has_table(self, namespace: str, table: str) -> bool:
        return table in self._namespaces.get(namespace, {})

    def read_table(self, namespace: str, table: str) -> Dataset:
        return self._namespaces[namespace][table].data

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
        tables = self._namespaces[namespace]
        if mode in ("create", "overwrite"):
            previous = tables.get(table)
            entry = _Entry(_materialize(data))
            if previous is not None:
                entry.properties = previous.properties
                entry.partition_by = previous.partition_by
            entry.properties = {**entry.properties, **(properties or {})}
            entry.partition_by = list(partition_by or entry.partition_by)
            tables[table] = entry
            return _manifest(entry.data.count())
        entry = tables[table]
        incoming = data.collect()
        if mode == "replace_where":
            kept = entry.data.filter(~replace_where.fill_null(False)).collect()
        else:
            kept = entry.data.collect()
        entry.data = self._concat(table, kept, incoming)
        return _manifest(incoming.num_rows)

    @staticmethod
    def _concat(table: str, kept: pa.Table, incoming: pa.Table) -> Dataset:
        """Stack the kept rows and the incoming ones, which share one schema by construction."""
        from batcher.api.session.frames import from_arrow

        try:
            return from_arrow(pa.concat_tables([kept, incoming.cast(kept.schema)]))
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            raise SchemaError(
                f"rows written to table {table!r} do not match its schema {kept.schema}: {exc}"
            ) from exc

    def drop_table(self, namespace: str, table: str) -> None:
        del self._namespaces[namespace][table]

    def table_properties(self, namespace: str, table: str) -> dict[str, str]:
        return dict(self._namespaces[namespace][table].properties)

    def partition_columns(self, namespace: str, table: str) -> list[str]:
        return list(self._namespaces[namespace][table].partition_by)
