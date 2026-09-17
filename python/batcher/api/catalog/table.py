"""`Table`: a handle on one catalog table — its name, schema, properties and rows.

A handle is metadata plus a way to read. It does not write: a write to a catalog table is
`ds.write.table(name, mode=...)`, the one spelling for every save mode, and a handle's
`name` is fully qualified precisely so it can be passed straight back to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.api.catalog.catalog import Catalog
    from batcher.api.dataset import Dataset

__all__ = ["Table"]


class Table:
    """A handle on one table in a `Catalog`, obtained from `Catalog.get_table`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> cat = bt.Catalog.from_pydict({"sales.orders": {"id": [1, 2]}}, name="shop")
            >>> t = cat.get_table("sales.orders")
            >>> t.name
            'shop.sales.orders'
            >>> t.read().count()
            2
    """

    __slots__ = ("_catalog", "_namespace", "_table")

    def __init__(self, catalog: Catalog, namespace: str, table: str) -> None:
        """Bind a handle; use `Catalog.get_table` rather than calling this directly."""
        self._catalog = catalog
        self._namespace = namespace
        self._table = table

    def __repr__(self) -> str:
        """Show the qualified name, e.g. ``Table('memory.main.orders')``."""
        return f"Table({self.name!r})"

    @property
    def name(self) -> str:
        """The fully qualified ``catalog.namespace.table`` name.

        Returns:
            The dotted name, which `Session.table` and `ds.write.table` both accept.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1]}})
                >>> cat.get_table("t").name
                'memory.main.t'
        """
        return f"{self._catalog.name}.{self._namespace}.{self._table}"

    @property
    def schema(self) -> pa.Schema:
        """The table's Arrow schema, read from metadata without scanning rows.

        Returns:
            The schema of the table's current snapshot.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1]}})
                >>> cat.get_table("t").schema.names
                ['x']
        """
        return self.read().schema

    @property
    def properties(self) -> dict[str, str]:
        """The table's key-value properties (Spark ``TBLPROPERTIES``).

        Returns:
            A copy of the properties; changing it does not change the table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({})
                >>> t = cat.create_table("t", {"x": [1]}, properties={"owner": "ops"})
                >>> t.properties
                {'owner': 'ops'}
        """
        return self._catalog._backend.table_properties(self._namespace, self._table)

    def read(self) -> Dataset:
        """A lazy `Dataset` over the table's current rows.

        Returns:
            The table's rows, resolved when this is called.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> cat = bt.Catalog.from_pydict({"t": {"x": [1, 2, 3]}})
                >>> cat.get_table("t").read().to_pydict()
                {'x': [1, 2, 3]}
        """
        return self._catalog._backend.read_table(self._namespace, self._table)
