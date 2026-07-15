"""The `ds.meta.schema` accessor — questions about types, which never touch data at all.

The cheapest shortcuts in the system: a plan knows its own output schema, so every answer
here is a dictionary lookup. They earn their place because the alternative people actually
write is a `collect()` followed by a pandas `dtypes` — a full scan to learn something the
plan already knew.

Nothing here ever executes, and nothing here can be wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.api.dataset.meta._facts import MetaBase

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["SchemaMeta"]

# The type families a user names in one breath, and the Arrow predicate for each.
_FAMILIES: dict[str, Callable[[pa.DataType], bool]] = {
    "numeric": lambda t: (
        pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)
    ),
    "integer": pa.types.is_integer,
    "float": pa.types.is_floating,
    "string": lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
    "boolean": pa.types.is_boolean,
    "temporal": lambda t: pa.types.is_temporal(t),
    "nested": lambda t: pa.types.is_nested(t),
}


class SchemaMeta(MetaBase):
    """Type-level shortcuts, reached as ``ds.meta.schema``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1], "name": ["a"]})
            >>> ds.meta.schema.numeric()
            ['x']
            >>> ds.meta.schema.is_string("name")
            True
    """

    __slots__ = ()

    def dtype(self, column: str) -> pa.DataType:
        """The Arrow type of `column`.

        Args:
            column: The column to look up.

        Returns:
            The column's Arrow data type.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.schema.dtype("x")
                DataType(int64)
        """
        return self._arrow().field(self.require_column(column)).type

    def num_columns(self) -> int:
        """How many columns the plan outputs.

        Returns:
            The number of output columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1], "b": [2]}).meta.schema.num_columns()
                2
        """
        return len(self._ds.columns)

    def has(self, column: str) -> bool:
        """Whether `column` is one of the plan's output columns.

        Args:
            column: The name to look for.

        Returns:
            ``True`` if the column exists.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1]}).meta.schema.has("b")
                False
        """
        return column in self._ds.columns

    def index(self, column: str) -> int:
        """The position of `column` in the output schema, counting from zero.

        Args:
            column: The column to locate.

        Returns:
            The column's zero-based position.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1], "b": [2]}).meta.schema.index("b")
                1
        """
        return self._ds.columns.index(self.require_column(column))

    def is_numeric(self, column: str) -> bool:
        """Whether `column` is an integer, float, or decimal column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a numeric column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.5]}).meta.schema.is_numeric("x")
                True
        """
        return _FAMILIES["numeric"](self.dtype(column))

    def is_integer(self, column: str) -> bool:
        """Whether `column` is an integer column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for an integer column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).meta.schema.is_integer("x")
                True
        """
        return _FAMILIES["integer"](self.dtype(column))

    def is_float(self, column: str) -> bool:
        """Whether `column` is a floating-point column — the type NaN lives in.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a float column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.5]}).meta.schema.is_float("x")
                True
        """
        return _FAMILIES["float"](self.dtype(column))

    def is_string(self, column: str) -> bool:
        """Whether `column` is a string column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a string column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"s": ["a"]}).meta.schema.is_string("s")
                True
        """
        return _FAMILIES["string"](self.dtype(column))

    def is_boolean(self, column: str) -> bool:
        """Whether `column` is a boolean column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a boolean column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"b": [True]}).meta.schema.is_boolean("b")
                True
        """
        return _FAMILIES["boolean"](self.dtype(column))

    def is_temporal(self, column: str) -> bool:
        """Whether `column` is a date, time, timestamp, or duration column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a temporal column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2024, 1, 1)]})
                >>> ds.meta.schema.is_temporal("d")
                True
        """
        return _FAMILIES["temporal"](self.dtype(column))

    def is_nested(self, column: str) -> bool:
        """Whether `column` is a list, struct, or map column.

        Args:
            column: The column to test.

        Returns:
            ``True`` for a nested column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"xs": [[1, 2]]}).meta.schema.is_nested("xs")
                True
        """
        return _FAMILIES["nested"](self.dtype(column))

    def numeric(self) -> list[str]:
        """Every numeric column, in schema order.

        Returns:
            The names of the integer, float, and decimal columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1], "s": ["a"]}).meta.schema.numeric()
                ['x']
        """
        return self._of_family("numeric")

    def strings(self) -> list[str]:
        """Every string column, in schema order.

        Returns:
            The names of the string columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1], "s": ["a"]}).meta.schema.strings()
                ['s']
        """
        return self._of_family("string")

    def booleans(self) -> list[str]:
        """Every boolean column, in schema order.

        Returns:
            The names of the boolean columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"b": [True], "x": [1]}).meta.schema.booleans()
                ['b']
        """
        return self._of_family("boolean")

    def temporal(self) -> list[str]:
        """Every date/time/timestamp/duration column, in schema order.

        Returns:
            The names of the temporal columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> bt.from_pydict({"d": [dt.date(2024, 1, 1)]}).meta.schema.temporal()
                ['d']
        """
        return self._of_family("temporal")

    def nested(self) -> list[str]:
        """Every list/struct/map column, in schema order.

        Returns:
            The names of the nested columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"xs": [[1]], "x": [1]}).meta.schema.nested()
                ['xs']
        """
        return self._of_family("nested")

    def select(self, family: str) -> Dataset:
        """Project every column of one type family (pandas ``select_dtypes``), lazily.

        Args:
            family: One of ``numeric``, ``integer``, ``float``, ``string``, ``boolean``,
                ``temporal``, ``nested``.

        Returns:
            A new dataset containing only the columns of that family.

        Raises:
            PlanError: If `family` is not a known type family.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1], "s": ["a"]})
                >>> ds.meta.schema.select("numeric").columns
                ['x']
        """
        return self._ds.select(*self._of_family(family))

    def _of_family(self, family: str) -> list[str]:
        """The columns whose type belongs to `family`, in schema order."""
        from batcher._internal.errors import PlanError

        predicate = _FAMILIES.get(family)
        if predicate is None:
            known = ", ".join(sorted(_FAMILIES))
            raise PlanError(f"meta.schema: unknown type family {family!r}; expected one of {known}")
        schema = self._arrow()
        return [f.name for f in schema if predicate(f.type)]

    def _arrow(self) -> pa.Schema:
        """The plan's output Arrow schema."""
        return self._ds.schema
