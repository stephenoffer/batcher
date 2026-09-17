"""`SessionCatalog`: the catalogs one `Session` has attached, and where names resolve.

``session.catalog`` is this object. It holds the attached `Catalog`s by name, a *current*
catalog and namespace, and the one resolution rule every entry point shares — `Session.table`,
SQL ``FROM``/``INSERT``/``DROP``, and `ds.write.table` all ask it what a dotted name means:

- ``"t"`` is table ``t`` in the current namespace of the current catalog;
- ``"ns.t"`` is table ``t`` in namespace ``ns`` of the current catalog, unless ``ns`` is an
  attached catalog's name and not a namespace of the current one, in which case it is table
  ``t`` in that catalog's default namespace;
- ``"cat.ns.t"`` (three or more parts) starting with an attached catalog is fully qualified.

State is per session. A fresh session starts with one in-memory catalog named ``memory`` and
the current namespace ``main``, which are DuckDB's names for the same two things, so
``current_catalog()`` and ``current_schema()`` answer the same in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.catalog import _names
from batcher.api.catalog.catalog import Catalog

if TYPE_CHECKING:
    from batcher.api.catalog.table import Table

__all__ = ["SessionCatalog"]


class SessionCatalog:
    """The catalogs a `Session` has attached, and the current catalog and namespace.

    Reached as ``session.catalog``; tables named without a catalog resolve against the
    current one, and without a namespace against the current namespace.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> s = bt.Session()
            >>> s.catalog.current_catalog(), s.catalog.current_namespace()
            ('memory', 'main')
            >>> _ = s.catalog.create_table("orders", bt.from_pydict({"id": [1, 2]}))
            >>> s.table("orders").count()
            2
    """

    __slots__ = ("_catalogs", "_current")

    def __init__(self) -> None:
        """Start with the in-memory ``memory`` catalog, current namespace ``main``."""
        default = Catalog.from_pydict({}, name="memory")
        self._catalogs: dict[str, Catalog] = {default.name: default}
        self._current: tuple[str, str] = (default.name, "main")

    def __repr__(self) -> str:
        """Show the attached catalog names and the current position."""
        catalog, namespace = self._current
        return f"SessionCatalog(catalogs={list(self._catalogs)!r}, current='{catalog}.{namespace}')"

    # --- attached catalogs ---------------------------------------------------------
    def attach(self, catalog: Catalog, alias: str | None = None) -> None:
        """Attach `catalog` to this session under `alias` (default: its own name).

        Args:
            catalog: The catalog to attach.
            alias: The name tables are qualified by in this session.

        Raises:
            PlanError: `catalog` is not a `Catalog`, or the name is already attached.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.catalog.attach(bt.Catalog.from_pydict({"t": {"x": [1]}}, name="scratch"))
                >>> s.catalog.list_catalogs()
                ['memory', 'scratch']
                >>> s.table("scratch.main.t").count()
                1
        """
        if not isinstance(catalog, Catalog):
            raise PlanError(
                f"attach() takes a bt.Catalog, got {type(catalog).__name__}",
                hint="Register a Dataset with Session.register(name, dataset) instead.",
            )
        name = catalog.name if alias is None else alias
        if "." in name or not name:
            raise PlanError(f"a catalog alias must be a non-empty name without dots, got {name!r}")
        if name in self._catalogs:
            raise PlanError(f"a catalog named {name!r} is already attached")
        if alias is not None and alias != catalog.name:
            catalog = Catalog(catalog._backend, alias)
        self._catalogs[name] = catalog

    def detach(self, name: str) -> None:
        """Detach the catalog `name`; the tables it holds are untouched.

        Args:
            name: The attached catalog's name.

        Raises:
            PlanError: No such catalog, or it is the current catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.catalog.attach(bt.Catalog.from_pydict({}, name="scratch"))
                >>> s.catalog.detach("scratch")
                >>> s.catalog.has_catalog("scratch")
                False
        """
        self.get_catalog(name)
        if name == self._current[0]:
            raise PlanError(
                f"catalog {name!r} is the current catalog and cannot be detached",
                hint="Switch away first with use('<other catalog>').",
            )
        del self._catalogs[name]

    def list_catalogs(self, pattern: str | None = None) -> list[str]:
        """The attached catalogs' names, sorted, optionally filtered by a glob.

        Args:
            pattern: A shell-style glob, or None for every catalog.

        Returns:
            The matching catalog names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.list_catalogs()
                ['memory']
        """
        return _names.filter_names(list(self._catalogs), pattern)

    def has_catalog(self, name: str) -> bool:
        """Whether a catalog is attached as `name`.

        Args:
            name: The catalog name.

        Returns:
            True if attached.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.has_catalog("memory")
                True
        """
        return name in self._catalogs

    def get_catalog(self, name: str) -> Catalog:
        """The catalog attached as `name`.

        Args:
            name: The catalog name.

        Returns:
            The attached `Catalog`.

        Raises:
            PlanError: No catalog is attached under `name`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.get_catalog("memory").list_namespaces()
                ['main']
        """
        if name not in self._catalogs:
            raise PlanError(
                f"no catalog {name!r} is attached",
                available=list(self._catalogs),
                available_label="Attached catalogs",
            )
        return self._catalogs[name]

    # --- current position ----------------------------------------------------------
    def current_catalog(self) -> str:
        """The name of the catalog unqualified names resolve against.

        Returns:
            The current catalog's name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.current_catalog()
                'memory'
        """
        return self._current[0]

    def current_namespace(self) -> str:
        """The namespace one-part table names resolve into.

        Returns:
            The current namespace's name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.current_namespace()
                'main'
        """
        return self._current[1]

    def use(self, name: str) -> None:
        """Make a catalog, a namespace, or ``catalog.namespace`` current (SQL ``USE``).

        A name that is an attached catalog switches to it and to its default namespace;
        otherwise it names a namespace, of the current catalog unless its first part is an
        attached catalog.

        Args:
            name: ``"catalog"``, ``"namespace"`` or ``"catalog.namespace"``.

        Raises:
            PlanError: Neither an attached catalog nor an existing namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.catalog.create_namespace("raw")
                >>> s.catalog.use("raw")
                >>> s.catalog.current_namespace()
                'raw'
        """
        parts = _names.split(name, what="catalog or namespace")
        if parts[0] in self._catalogs:
            catalog = self._catalogs[parts[0]]
            namespace = (
                _names.join(*parts[1:]) if len(parts) > 1 else catalog._backend.default_namespace
            )
        else:
            catalog, namespace = self._catalogs[self._current[0]], name
        if not catalog.has_namespace(namespace):
            raise PlanError(
                f"cannot use {name!r}: no namespace {namespace!r} in catalog {catalog.name!r} "
                "and no attached catalog of that name",
                available=[*self._catalogs, *catalog.list_namespaces()],
                available_label="Catalogs and namespaces",
            )
        self._current = (parts[0] if parts[0] in self._catalogs else self._current[0], namespace)

    # --- resolution ------------------------------------------------------------------
    def _current_catalog(self) -> Catalog:
        return self._catalogs[self._current[0]]

    def _resolve_table(self, name: str) -> tuple[Catalog, str]:
        """Resolve a session-level table name to its catalog and catalog-relative name."""
        parts = _names.split(name)
        current = self._current_catalog()
        if len(parts) == 1:
            return current, _names.join(self._current[1], parts[0])
        head = parts[0]
        if head in self._catalogs and (len(parts) > 2 or not current.has_namespace(head)):
            return self._catalogs[head], _names.join(*parts[1:])
        return current, name

    def _resolve_namespace(self, name: str) -> tuple[Catalog, str]:
        parts = _names.split(name, what="namespace")
        if len(parts) > 1 and parts[0] in self._catalogs:
            return self._catalogs[parts[0]], _names.join(*parts[1:])
        return self._current_catalog(), name

    # --- namespaces ------------------------------------------------------------------
    def list_namespaces(self, pattern: str | None = None) -> list[str]:
        """The current catalog's namespaces, sorted, optionally filtered by a glob.

        Args:
            pattern: A shell-style glob, or None for every namespace.

        Returns:
            The matching namespace names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.list_namespaces()
                ['main']
        """
        return self._current_catalog().list_namespaces(pattern)

    def has_namespace(self, name: str) -> bool:
        """Whether the namespace exists (``"ns"``, or ``"catalog.ns"``).

        Args:
            name: The namespace, optionally qualified by an attached catalog.

        Returns:
            True if it exists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.has_namespace("memory.main")
                True
        """
        catalog, namespace = self._resolve_namespace(name)
        return catalog.has_namespace(namespace)

    def create_namespace(self, name: str, *, if_not_exists: bool = False) -> None:
        """Create a namespace (``"ns"``, or ``"catalog.ns"``); see `Catalog.create_namespace`.

        Args:
            name: The namespace, optionally qualified by an attached catalog.
            if_not_exists: Do nothing when it already exists, instead of raising.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.catalog.create_namespace("raw")
                >>> s.catalog.list_namespaces()
                ['main', 'raw']
        """
        catalog, namespace = self._resolve_namespace(name)
        catalog.create_namespace(namespace, if_not_exists=if_not_exists)

    def drop_namespace(self, name: str, *, if_exists: bool = False, cascade: bool = False) -> None:
        """Drop a namespace (``"ns"``, or ``"catalog.ns"``); see `Catalog.drop_namespace`.

        Args:
            name: The namespace, optionally qualified by an attached catalog.
            if_exists: Do nothing when it does not exist, instead of raising.
            cascade: Drop its tables first.

        Raises:
            PlanError: It is the current namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.catalog.create_namespace("raw")
                >>> s.catalog.drop_namespace("raw")
                >>> s.catalog.has_namespace("raw")
                False
        """
        catalog, namespace = self._resolve_namespace(name)
        if (catalog.name, namespace) == self._current:
            raise PlanError(f"namespace {name!r} is the current namespace and cannot be dropped")
        catalog.drop_namespace(namespace, if_exists=if_exists, cascade=cascade)

    # --- tables ------------------------------------------------------------------------
    def list_tables(self, pattern: str | None = None) -> list[str]:
        """The current catalog's tables as ``namespace.table``, optionally filtered by a glob.

        Tables registered with `Session.register` are session views, not catalog tables, and
        are listed by `Session.list`.

        Args:
            pattern: A shell-style glob matched against ``namespace.table``.

        Returns:
            The matching table names, sorted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.catalog.create_table("orders", bt.from_pydict({"id": [1]}))
                >>> s.catalog.list_tables("main.*")
                ['main.orders']
        """
        return self._current_catalog().list_tables(pattern)

    def has_table(self, name: str) -> bool:
        """Whether a catalog table resolves from `name`.

        Args:
            name: A table name, resolved as the class docstring describes.

        Returns:
            True if the table exists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.Session().catalog.has_table("orders")
                False
        """
        catalog, relative = self._resolve_table(name)
        return catalog.has_table(relative)

    def get_table(self, name: str) -> Table:
        """A handle on the catalog table `name` resolves to.

        Args:
            name: A table name, resolved as the class docstring describes.

        Returns:
            The table handle.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.catalog.create_table("orders", bt.from_pydict({"id": [1]}))
                >>> s.catalog.get_table("orders").name
                'memory.main.orders'
        """
        catalog, relative = self._resolve_table(name)
        return catalog.get_table(relative)

    def create_table(self, name: str, source: Any, **options: Any) -> Table:
        """Create a catalog table; `options` are `Catalog.create_table`'s keywords.

        Args:
            name: A table name, resolved as the class docstring describes.
            source: The initial rows (a `Dataset` or anything `bt.from_any` takes) or a
                `pyarrow.Schema` for an empty table.
            **options: ``if_not_exists``, ``partition_by`` and ``properties``.

        Returns:
            A handle on the table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> t = s.catalog.create_table("orders", {"id": [1]}, properties={"owner": "ops"})
                >>> t.properties
                {'owner': 'ops'}
        """
        catalog, relative = self._resolve_table(name)
        return catalog.create_table(relative, source, **options)

    def drop_table(self, name: str, *, if_exists: bool = False) -> None:
        """Drop the catalog table `name` resolves to; see `Catalog.drop_table`.

        Args:
            name: A table name, resolved as the class docstring describes.
            if_exists: Do nothing when it does not exist, instead of raising.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.catalog.create_table("orders", {"id": [1]})
                >>> s.catalog.drop_table("orders")
                >>> s.catalog.has_table("orders")
                False
        """
        catalog, relative = self._resolve_table(name)
        catalog.drop_table(relative, if_exists=if_exists)

    def truncate_table(self, name: str) -> None:
        """Remove every row from the catalog table `name` resolves to, keeping its schema.

        Args:
            name: A table name, resolved as the class docstring describes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.catalog.create_table("orders", {"id": [1, 2]})
                >>> s.catalog.truncate_table("orders")
                >>> s.table("orders").count()
                0
        """
        catalog, relative = self._resolve_table(name)
        catalog.truncate_table(relative)
