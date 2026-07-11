"""The `Selector` expression leaf and its `.name` rename accessor.

A `Selector` is an `Expr` that stands for *many* columns at plan time; it carries a
match test (name / pattern / dtype) and an optional per-column rename, and composes
with set algebra. The projection layer (`expand`) resolves it against a schema. The
public constructors live in `build`; this module is the type they return.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr

__all__ = ["Selector", "_Match", "_SelectorNameNamespace", "_by_name"]

# A selector's match test: given a column's name and its Arrow dtype (None when the
# plan cannot resolve a schema), decide whether the column is in the selection.
_Match = Callable[[str, "pa.DataType | None"], bool]


class _SelectorNameNamespace:
    """Rename accessors for a selector's expanded columns: ``numeric().name.prefix("n_")``.

    A selector expands to many columns, so `Expr.alias` (which names exactly one
    column) cannot name them. These methods derive each output name from the matched
    input column's name instead.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5], "s": ["x"]})
            >>> ds.select(bt.numeric().name.prefix("n_")).columns
            ['n_a', 'n_b']
    """

    __slots__ = ("_s",)

    def __init__(self, s: Selector) -> None:
        """Wrap the parent :class:`Selector` so its `.name` methods can build on it."""
        self._s = s

    def keep(self) -> Selector:
        """Keep each matched column's original name (the default).

        Returns:
            A selector whose expanded columns keep their input names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1.5]})
                >>> ds.select(bt.numeric().name.keep().round(0)).columns
                ['a']
        """
        return self._s._with_rename(lambda c: c, "name.keep()")

    def prefix(self, prefix: str) -> Selector:
        """Prepend `prefix` to each matched column's name.

        Args:
            prefix: The string to prepend to every expanded output name.

        Returns:
            A selector whose expanded columns are renamed with the prefix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "s": ["x"]})
                >>> ds.select(bt.numeric().name.prefix("n_")).columns
                ['n_a']
        """
        return self._s._with_rename(lambda c: f"{prefix}{c}", f"name.prefix({prefix!r})")

    def suffix(self, suffix: str) -> Selector:
        """Append `suffix` to each matched column's name.

        Args:
            suffix: The string to append to every expanded output name.

        Returns:
            A selector whose expanded columns are renamed with the suffix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "s": ["x"]})
                >>> ds.select(bt.numeric().name.suffix("_n")).columns
                ['a_n']
        """
        return self._s._with_rename(lambda c: f"{c}{suffix}", f"name.suffix({suffix!r})")

    def to_lowercase(self) -> Selector:
        """Lowercase each matched column's name.

        Returns:
            A selector whose expanded columns are renamed to lowercase.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"Aa": [1]})
                >>> ds.select(bt.all().name.to_lowercase()).columns
                ['aa']
        """
        return self._s._with_rename(str.lower, "name.to_lowercase()")

    def to_uppercase(self) -> Selector:
        """Uppercase each matched column's name.

        Returns:
            A selector whose expanded columns are renamed to uppercase.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"aa": [1]})
                >>> ds.select(bt.all().name.to_uppercase()).columns
                ['AA']
        """
        return self._s._with_rename(str.upper, "name.to_uppercase()")

    def map(self, fn: Callable[[str], str]) -> Selector:
        """Derive each output name by applying `fn` to the matched column's name.

        Args:
            fn: A function from the input column name to the output column name.

        Returns:
            A selector whose expanded columns are renamed by `fn`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a_raw": [1]})
                >>> ds.select(bt.all().name.map(lambda c: c.removesuffix("_raw"))).columns
                ['a']
        """
        return self._s._with_rename(fn, "name.map(...)")


class Selector(Expr):
    """An `Expr` leaf standing for every column that matches a predicate.

    Built by `all`, `exclude`, `matches`, `numeric`, `integer`, `floating`, `string`,
    `boolean`, and `temporal`; combined with ``|`` (union), ``&`` (intersection),
    ``-`` (difference), and ``~`` (complement). Use it wherever a projection is
    built — ``select``, ``with_columns``, ``drop`` — and compose the scalar algebra
    onto it to compute over every matched column at once.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1.234], "b": [5.678], "s": ["x"]})
            >>> ds.with_columns(bt.numeric().round(1)).to_pydict()
            {'a': [1.2], 'b': [5.7], 's': ['x']}
    """

    __slots__ = ("_desc", "_match", "_needs_dtype", "_rename")

    def __init__(
        self,
        match: _Match,
        desc: str,
        *,
        needs_dtype: bool = False,
        rename: Callable[[str], str] | None = None,
    ) -> None:
        """Build a selector from a match test; prefer the module-level constructors."""
        self._match = match
        self._desc = desc
        self._needs_dtype = needs_dtype
        self._rename = rename

    def __repr__(self) -> str:
        """Render the selector's construction, e.g. ``numeric() - matches('^_')``."""
        return self._desc

    def to_ir(self) -> dict[str, Any]:
        """Always raises: a selector is expanded by a projection, never lowered to IR.

        Reaching this means the selector sat somewhere expansion does not run — a
        `filter` predicate, a join key, an aggregate argument — so it would otherwise
        silently become an unresolvable column reference in the engine.

        Returns:
            Never returns; the signature matches `Expr.to_ir` so the guard triggers
            wherever an expression is lowered.

        Raises:
            PlanError: Always, naming the selector and the positions it is valid in.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> try:
                ...     bt.numeric().to_ir()
                ... except Exception as err:
                ...     print(type(err).__name__)
                PlanError
        """
        raise PlanError(
            f"the column selector {self._desc} cannot be used here; selectors expand to "
            "many columns and are only valid in select(), with_columns(), and drop() — "
            "in a filter/join/agg position name a single column with col(...)"
        )

    @property
    def name(self) -> _SelectorNameNamespace:
        """Rename accessors for the expanded columns (``.name.prefix``, ``.name.suffix``).

        Returns:
            The `.name` accessor namespace bound to this selector.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "s": ["x"]})
                >>> ds.select(bt.numeric().name.suffix("_n")).columns
                ['a_n']
        """
        return _SelectorNameNamespace(self)

    def _with_rename(self, fn: Callable[[str], str], what: str) -> Selector:
        return Selector(
            self._match, f"{self._desc}.{what}", needs_dtype=self._needs_dtype, rename=fn
        )

    def _combine(self, other: Selector, op: Callable[[bool, bool], bool], sym: str) -> Selector:
        if not isinstance(other, Selector):
            raise PlanError(
                f"a column selector can only be combined with another selector using "
                f"{sym!r}, got {type(other).__name__}"
            )
        return Selector(
            lambda n, d: op(self._match(n, d), other._match(n, d)),
            f"({self._desc} {sym} {other._desc})",
            needs_dtype=self._needs_dtype or other._needs_dtype,
            rename=self._rename or other._rename,
        )

    def __or__(self, other: Selector) -> Selector:  # type: ignore[override]
        """Union: columns matched by either selector."""
        return self._combine(other, lambda a, b: a or b, "|")

    def __and__(self, other: Selector) -> Selector:  # type: ignore[override]
        """Intersection: columns matched by both selectors."""
        return self._combine(other, lambda a, b: a and b, "&")

    def __sub__(self, other: Selector) -> Selector:  # type: ignore[override]
        """Difference: columns matched by this selector but not the other."""
        return self._combine(other, lambda a, b: a and not b, "-")

    def __invert__(self) -> Selector:  # type: ignore[override]
        """Complement: every column this selector does not match."""
        match, desc = self._match, self._desc
        return Selector(lambda n, d: not match(n, d), f"~{desc}", needs_dtype=self._needs_dtype)

    def exclude(self, *names: str) -> Selector:
        """Drop the named columns from this selection.

        Args:
            *names: Column names to remove from the selection.

        Returns:
            A selector matching this selection minus `names`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "b": [2], "id": [3]})
                >>> ds.select(bt.all().exclude("id")).columns
                ['a', 'b']
        """
        return self - _by_name(names)

    def matched_columns(self, columns: list[str], schema: Any | None) -> list[str]:
        """The input columns this selector matches, in the input's column order.

        Args:
            columns: The input plan's column names, in order.
            schema: The input plan's `SchemaRef`, or None when it cannot be resolved.

        Returns:
            The matched column names, in input order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.matches("^x").matched_columns(["xa", "b", "xc"], None)
                ['xa', 'xc']
        """
        if self._needs_dtype and schema is None:
            raise PlanError(
                f"the dtype-based selector {self._desc} needs the input schema, which is "
                "not known for this plan; select the columns by name instead"
            )
        arrow = schema.arrow if schema is not None else None
        out = []
        for c in columns:
            has_field = arrow is not None and arrow.get_field_index(c) >= 0
            dtype = arrow.field(c).type if has_field else None
            if self._match(c, dtype):
                out.append(c)
        return out

    def output_name(self, column: str) -> str:
        """The output name for a matched `column` under this selector's rename.

        Args:
            column: A matched input column name.

        Returns:
            The renamed output column name, or `column` when no rename is set.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.all().name.prefix("p_").output_name("x")
                'p_x'
        """
        return self._rename(column) if self._rename is not None else column


def _by_name(names: tuple[str, ...]) -> Selector:
    wanted = frozenset(names)
    return Selector(lambda n, _d: n in wanted, f"by_name({', '.join(map(repr, names))})")
