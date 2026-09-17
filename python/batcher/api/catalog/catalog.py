"""`Catalog`: named namespaces of tables, over one storage backend.

A catalog maps ``namespace.table`` names to tables and owns their lifecycle: create, list,
drop, truncate, and the save modes a write to a table goes through. The storage is a
`CatalogBackend` — Arrow in this process, a directory of Delta tables, or a pyiceberg
catalog — chosen by the constructor, and every rule that is about *names* or *modes*
rather than storage is implemented once here, so the three backends agree on what
``mode="error"``, ``if_not_exists`` and ``cascade`` mean.

This is `api`: it builds `Dataset`s and runs writes. A catalog holds no global state; a
`Session` attaches the catalogs it resolves names against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.catalog import _names
from batcher.api.catalog.table import Table

if TYPE_CHECKING:
    from batcher.api.catalog._backend import CatalogBackend
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["TABLE_WRITE_MODES", "Catalog"]

#: The save modes `ds.write.table` accepts. `replace` is the one a path write has no use
#: for: replace a table that must already exist (Spark V2 ``replace``), where ``overwrite``
#: creates it when missing.
TABLE_WRITE_MODES = ("error", "ignore", "append", "overwrite", "overwrite_partitions", "replace")


def _as_dataset(source: Any) -> Dataset:
    """Lift a pyarrow table/schema, a dict of columns, or any framework frame to a Dataset."""
    from batcher.api.dataset import Dataset
    from batcher.api.session.frames import from_arrow
    from batcher.api.session.frameworks import from_any

    if isinstance(source, Dataset):
        return source
    if isinstance(source, pa.Schema):
        return from_arrow(source.empty_table())
    return from_any(source)


class Catalog:
    """Namespaces of tables over one storage backend: in-memory, a Delta directory, or Iceberg.

    Build one with `from_pydict`, `from_directory` or `from_iceberg`, then attach it to a
    session with ``session.catalog.attach(catalog)`` so SQL and `Session.table` can name its
    tables. A one-part table name lands in the catalog's default namespace (``main``, or
    ``default`` for Iceberg).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> cat = bt.Catalog.from_pydict({"sales.orders": {"id": [1, 2, 3]}}, name="shop")
            >>> cat.list_tables()
            ['sales.orders']
            >>> cat.get_table("sales.orders").read().count()
            3
    """

    __slots__ = ("_backend", "_name")

    def __init__(self, backend: CatalogBackend, name: str) -> None:
        """Wrap a backend; use `from_pydict`, `from_directory` or `from_iceberg` instead."""
        _names.split(name, what="catalog")
        if "." in name:
            raise PlanError(f"a catalog name cannot contain a dot, got {name!r}")
        self._backend = backend
        self._name = name

    def __repr__(self) -> str:
        """Show the name and backend, e.g. ``Catalog('memory', MemoryBackend)``."""
        return f"Catalog({self._name!r}, {type(self._backend).__name__})"

    # --- construction ----------------------------------------------------------
    @classmethod
    def from_pydict(cls, tables: dict[str, Any], *, name: str = "memory") -> Catalog:
        """Build an in-memory catalog from ``{"namespace.table": data}``.

        Each value is anything a table can be built from: a `Dataset` (kept lazy until the
        table is written to), a pyarrow table, a ``{column: values}`` dict, or a pandas or
        Polars frame. A one-part key lands in the ``main`` namespace.

        Args:
            tables: The tables to seed, keyed by ``"table"`` or ``"namespace.table"``.
            name: The catalog's name, which qualifies its tables in a session.

        Returns:
            The catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1]}, "raw.events": {"e": ["a"]}})
                >>> cat.list_tables()
                ['main.t', 'raw.events']
        """
        from batcher.api.catalog.memory import MemoryBackend

        seeded: dict[tuple[str, str], Dataset] = {}
        for key, data in tables.items():
            parts = _names.split(key)
            namespace = (
                _names.join(*parts[:-1]) if len(parts) > 1 else MemoryBackend.default_namespace
            )
            seeded[(namespace, parts[-1])] = _as_dataset(data)
        return cls(MemoryBackend(seeded), name)

    @classmethod
    def from_directory(cls, path: str, *, name: str | None = None) -> Catalog:
        """Build a catalog over a warehouse directory of Delta tables.

        Tables live at ``<path>/<namespace>/<table>`` and are ordinary Delta tables, so the
        catalog is the directory listing and needs no service. The directory need not exist
        until the first table is written.

        Args:
            path: A local path or filesystem URI (``s3://bucket/warehouse``).
            name: The catalog's name; defaults to the directory's last path segment.

        Returns:
            The catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile
                >>> cat = bt.Catalog.from_directory(tempfile.mkdtemp(), name="lake")
                >>> _ = cat.create_table("orders", bt.from_pydict({"id": [1, 2]}))
                >>> cat.list_tables()
                ['main.orders']
        """
        from batcher.api.catalog.directory import DirectoryBackend

        backend = DirectoryBackend(path)
        return cls(backend, name or path.rstrip("/").rsplit("/", 1)[-1] or "directory")

    @classmethod
    def from_iceberg(cls, catalog: Any, *, name: str | None = None) -> Catalog:
        """Build a catalog over an Iceberg catalog service, through pyiceberg.

        `catalog` is a live `pyiceberg.catalog.Catalog`, or the spec `bt.read.iceberg`
        accepts: a catalog name configured for pyiceberg, or a property dict whose ``type``
        selects the service (``"rest"``, ``"glue"``, ``"hive"``, ``"sql"``, and the REST
        aliases ``"unity"``, ``"polaris"``, ``"snowflake"``).

        Args:
            catalog: A live pyiceberg catalog, a configured catalog name, or a property dict.
            name: The catalog's name; defaults to pyiceberg's name for it.

        Returns:
            The catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_iceberg(  # doctest: +SKIP
                ...     {"type": "glue", "warehouse": "s3://bucket/warehouse"}, name="glue"
                ... )
        """
        from batcher.api.catalog.iceberg import IcebergBackend
        from batcher.io.catalog import resolve_catalog

        live = resolve_catalog(catalog)
        return cls(IcebergBackend(live), name or str(getattr(live, "name", "iceberg")))

    @property
    def name(self) -> str:
        """The catalog's name, the first part of its tables' qualified names.

        Returns:
            The name given at construction.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Catalog.from_pydict({}, name="scratch").name
                'scratch'
        """
        return self._name

    # --- namespaces --------------------------------------------------------------
    def list_namespaces(self, pattern: str | None = None) -> list[str]:
        """The catalog's namespaces, sorted, optionally filtered by a glob.

        Args:
            pattern: A shell-style glob such as ``"raw*"``, or None for every namespace.

        Returns:
            The matching namespace names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({})
                >>> cat.create_namespace("raw")
                >>> cat.list_namespaces()
                ['main', 'raw']
        """
        return _names.filter_names(self._backend.list_namespaces(), pattern)

    def has_namespace(self, name: str) -> bool:
        """Whether the namespace `name` exists.

        Args:
            name: The namespace name.

        Returns:
            True if it exists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Catalog.from_pydict({}).has_namespace("main")
                True
        """
        return _names.join(*_names.split(name, what="namespace")) in self._backend.list_namespaces()

    def create_namespace(self, name: str, *, if_not_exists: bool = False) -> None:
        """Create the namespace `name`.

        Args:
            name: The namespace name.
            if_not_exists: Do nothing when it already exists, instead of raising.

        Raises:
            PlanError: The namespace exists and `if_not_exists` is False.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({})
                >>> cat.create_namespace("raw")
                >>> cat.create_namespace("raw", if_not_exists=True)
                >>> cat.has_namespace("raw")
                True
        """
        if self.has_namespace(name):
            if if_not_exists:
                return
            raise PlanError(f"namespace {name!r} already exists in catalog {self._name!r}")
        self._backend.create_namespace(name)

    def drop_namespace(self, name: str, *, if_exists: bool = False, cascade: bool = False) -> None:
        """Drop the namespace `name`, refusing one that still holds tables unless `cascade`.

        Args:
            name: The namespace name.
            if_exists: Do nothing when it does not exist, instead of raising.
            cascade: Drop the namespace's tables first (SQL ``DROP SCHEMA ... CASCADE``).

        Raises:
            PlanError: The namespace is missing, is the default one, or holds tables and
                `cascade` is False.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"raw.events": {"e": [1]}})
                >>> cat.drop_namespace("raw", cascade=True)
                >>> cat.list_namespaces()
                ['main']
        """
        if not self.has_namespace(name):
            if if_exists:
                return
            raise PlanError(f"no namespace {name!r} in catalog {self._name!r}")
        if name == self._backend.default_namespace:
            raise PlanError(
                f"the default namespace {name!r} of catalog {self._name!r} cannot be dropped"
            )
        tables = self._backend.list_tables(name)
        if tables and not cascade:
            raise PlanError(
                f"namespace {name!r} still holds {len(tables)} table(s): {sorted(tables)}",
                hint="Drop them first, or pass cascade=True.",
            )
        for table in tables:
            self._backend.drop_table(name, table)
        self._backend.drop_namespace(name)

    # --- tables ------------------------------------------------------------------
    def _locate(self, name: str) -> tuple[str, str]:
        """Split a catalog-relative table name into ``(namespace, table)``."""
        parts = _names.split(name)
        if len(parts) == 1:
            return self._backend.default_namespace, parts[0]
        return _names.join(*parts[:-1]), parts[-1]

    def list_tables(self, pattern: str | None = None) -> list[str]:
        """Every table as ``namespace.table``, sorted, optionally filtered by a glob.

        Args:
            pattern: A shell-style glob matched against ``namespace.table``, such as
                ``"sales.*"``, or None for every table.

        Returns:
            The matching table names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"a.t": {"x": [1]}, "b.u": {"x": [2]}})
                >>> cat.list_tables("a.*")
                ['a.t']
        """
        names = [
            _names.join(namespace, table)
            for namespace in self._backend.list_namespaces()
            for table in self._backend.list_tables(namespace)
        ]
        return _names.filter_names(names, pattern)

    def has_table(self, name: str) -> bool:
        """Whether the table `name` exists.

        Args:
            name: ``"table"`` (default namespace) or ``"namespace.table"``.

        Returns:
            True if it exists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Catalog.from_pydict({"t": {"x": [1]}}).has_table("main.t")
                True
        """
        return self._backend.has_table(*self._locate(name))

    def get_table(self, name: str) -> Table:
        """A handle on the table `name`.

        Args:
            name: ``"table"`` (default namespace) or ``"namespace.table"``.

        Returns:
            The table handle.

        Raises:
            PlanError: No such table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1]}})
                >>> cat.get_table("t").schema.names
                ['x']
        """
        namespace, table = self._locate(name)
        if not self._backend.has_table(namespace, table):
            raise PlanError(
                f"no table {name!r} in catalog {self._name!r}",
                available=self.list_tables(),
                available_label="Tables",
            )
        return Table(self, namespace, table)

    def create_table(
        self,
        name: str,
        source: Any,
        *,
        if_not_exists: bool = False,
        partition_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
    ) -> Table:
        """Create the table `name` from a dataset's rows, or empty from a schema.

        Args:
            name: ``"table"`` or ``"namespace.table"``; the namespace must exist.
            source: A `Dataset` (or anything `bt.from_any` takes) whose rows the table
                starts with, or a `pyarrow.Schema` for an empty table.
            if_not_exists: Return the existing table instead of raising when it exists.
            partition_by: Partition columns, for backends whose tables are partitioned.
            properties: Table properties (Spark ``TBLPROPERTIES``).

        Returns:
            A handle on the table.

        Raises:
            PlanError: The table exists and `if_not_exists` is False, or its namespace
                does not exist.

        Examples:
            .. doctest::

                >>> import batcher as bt, pyarrow as pa
                >>> cat = bt.Catalog.from_pydict({})
                >>> t = cat.create_table("t", pa.schema([("x", pa.int64())]))
                >>> t.read().count()
                0
        """
        if self.has_table(name):
            if if_not_exists:
                return self.get_table(name)
            raise PlanError(f"table {name!r} already exists in catalog {self._name!r}")
        self._write(
            name,
            _as_dataset(source),
            mode="error",
            partition_by=partition_by,
            properties=properties,
        )
        return self.get_table(name)

    def drop_table(self, name: str, *, if_exists: bool = False) -> None:
        """Drop the table `name`, including its data for the in-memory and directory backends.

        Args:
            name: ``"table"`` or ``"namespace.table"``.
            if_exists: Do nothing when it does not exist, instead of raising.

        Raises:
            PlanError: No such table and `if_exists` is False.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1]}})
                >>> cat.drop_table("t")
                >>> cat.list_tables()
                []
        """
        namespace, table = self._locate(name)
        if not self._backend.has_table(namespace, table):
            if if_exists:
                return
            raise PlanError(f"no table {name!r} to drop in catalog {self._name!r}")
        self._backend.drop_table(namespace, table)

    def truncate_table(self, name: str) -> None:
        """Remove every row from the table `name`, keeping its schema and properties.

        Args:
            name: ``"table"`` or ``"namespace.table"``.

        Raises:
            PlanError: No such table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1, 2]}})
                >>> cat.truncate_table("t")
                >>> cat.get_table("t").read().count()
                0
        """
        schema = self.get_table(name).schema
        self._write(name, _as_dataset(schema), mode="overwrite")

    # --- writes --------------------------------------------------------------------
    def _write(
        self,
        name: str,
        data: Dataset,
        *,
        mode: str,
        by_name: bool = True,
        partition_by: list[str] | None = None,
        properties: dict[str, str] | None = None,
        replace_where: Any = None,
    ) -> WriteManifest:
        """Write `data` to the table `name` under a save mode; `ds.write.table` is the surface."""
        from batcher.api.catalog.modes import plan_write

        return plan_write(
            self,
            name,
            data,
            mode=mode,
            by_name=by_name,
            partition_by=partition_by,
            properties=properties,
            replace_where=replace_where,
        )
