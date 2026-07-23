"""Ecosystem-compatible spellings on the typed accessor namespaces.

The same migration surface as `compat.names`, one level down: the names a user
types on ``.str`` / ``.dt`` / ``.list`` when arriving from Python's own ``str``,
from Polars, or from PySpark.

Every alias here was checked against the primary it delegates to, and the set is
deliberately narrow. Several obvious-looking candidates are **absent on purpose**,
because the ecosystem name means something different from Batcher's method and a
silently-wrong alias is worse than a missing one:

* ``str.find`` / ``str.index`` — `position` is 1-based and returns 0 when absent
  (SQL), where pandas' ``find`` is 0-based and returns -1.
* ``str.substring`` — `substr` is 1-based (SQL); pandas/Polars slicing is 0-based.
  Use the existing 0-based ``str.slice``.
* ``str.islower`` / ``str.isupper`` — `is_lower`/`is_upper` are true for a string
  with no cased characters (``"123"``), where Python's are false.
* ``str.count`` — pandas' ``count`` is regex; `count_matches` is literal. The
  regex spelling already exists as ``str.regexp_count``.
* ``str.casefold`` — Python's casefold is not `to_lowercase` for non-ASCII
  (``"ß"`` folds to ``"ss"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.expr_ir.core import Expr
    from batcher.plan.expr_ir.func_nodes import ListGet, StrFunc

__all__ = ["bind_namespace_compat"]


# --- .str: the Python `str` predicate names -----------------------------------------
def isdigit(self: object) -> Expr:
    """True where every character is a digit — the Python ``str.isdigit`` spelling.

    Matches Python exactly, including the empty-string and mixed-content cases.

    Returns:
        A boolean expression, true where the whole string is digits.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["123", "a1"]})
            >>> ds.select(r=bt.col("s").str.isdigit()).to_pydict()
            {'r': [True, False]}
    """
    return self.is_numeric()  # type: ignore[attr-defined]


def isalpha(self: object) -> Expr:
    """True where every character is a letter — the Python ``str.isalpha`` spelling.

    Returns:
        A boolean expression, true where the whole string is letters.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["abc", "a1"]})
            >>> ds.select(r=bt.col("s").str.isalpha()).to_pydict()
            {'r': [True, False]}
    """
    return self.is_alpha()  # type: ignore[attr-defined]


def isalnum(self: object) -> Expr:
    """True where every character is a letter or digit — the Python ``str.isalnum`` spelling.

    Returns:
        A boolean expression, true where the whole string is alphanumeric.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["a1", " "]})
            >>> ds.select(r=bt.col("s").str.isalnum()).to_pydict()
            {'r': [True, False]}
    """
    return self.is_alnum()  # type: ignore[attr-defined]


def isspace(self: object) -> Expr:
    """True where every character is whitespace — the Python ``str.isspace`` spelling.

    Returns:
        A boolean expression, true where the whole string is whitespace.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": [" ", "a"]})
            >>> ds.select(r=bt.col("s").str.isspace()).to_pydict()
            {'r': [True, False]}
    """
    return self.is_space()  # type: ignore[attr-defined]


def strip_prefix(self: object, prefix: str) -> StrFunc:
    """Remove `prefix` if present — the Polars ``strip_prefix`` spelling of `removeprefix`.

    Args:
        prefix: The leading substring to remove when present.

    Returns:
        A new expression with the prefix removed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["abcd", "xcd"]})
            >>> ds.select(r=bt.col("s").str.strip_prefix("ab")).to_pydict()
            {'r': ['cd', 'xcd']}
    """
    return self.removeprefix(prefix)  # type: ignore[attr-defined]


def strip_suffix(self: object, suffix: str) -> StrFunc:
    """Remove `suffix` if present — the Polars ``strip_suffix`` spelling of `removesuffix`.

    Args:
        suffix: The trailing substring to remove when present.

    Returns:
        A new expression with the suffix removed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"s": ["abcd", "abx"]})
            >>> ds.select(r=bt.col("s").str.strip_suffix("cd")).to_pydict()
            {'r': ['ab', 'abx']}
    """
    return self.removesuffix(suffix)  # type: ignore[attr-defined]


# --- .dt: the snake_case spellings ---------------------------------------------------
def day_of_week(self: object) -> Expr:
    """ISO day of week, Monday=1 to Sunday=7 — the snake_case spelling of `dayofweek`.

    Returns:
        An Int64 expression of the ISO weekday.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
            >>> ds.select(r=bt.col("d").dt.day_of_week()).to_pydict()
            {'r': [4]}
    """
    return self.dayofweek()  # type: ignore[attr-defined]


def day_of_year(self: object) -> Expr:
    """Day of year, 1-366 — the snake_case spelling of `dayofyear`.

    Returns:
        An Int64 expression of the day of year.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
            >>> ds.select(r=bt.col("d").dt.day_of_year()).to_pydict()
            {'r': [46]}
    """
    return self.dayofyear()  # type: ignore[attr-defined]


def week_of_year(self: object) -> Expr:
    """ISO week number, 1-53 — the snake_case spelling of `weekofyear`.

    Returns:
        An Int64 expression of the ISO week number.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
            >>> ds.select(r=bt.col("d").dt.week_of_year()).to_pydict()
            {'r': [7]}
    """
    return self.weekofyear()  # type: ignore[attr-defined]


# --- .list: the Polars/PySpark/numpy spellings ---------------------------------------
def lengths(self: object) -> Expr:
    """Element count per list — the legacy Polars ``lengths`` spelling of `len`.

    Returns:
        An Int64 expression of the list length.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"l": [[1, 2, 3], [4]]})
            >>> ds.select(r=bt.col("l").list.lengths()).to_pydict()
            {'r': [3, 1]}
    """
    return self.len()  # type: ignore[attr-defined]


def element_at(self: object, index: int) -> ListGet:
    """Element at `index` — the PySpark ``element_at`` spelling of `get`.

    Args:
        index: Zero-based position; negative counts from the end.

    Returns:
        An expression selecting the element.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"l": [[10, 20, 30]]})
            >>> ds.select(r=bt.col("l").list.element_at(1)).to_pydict()
            {'r': [20]}
    """
    return self.get(index)  # type: ignore[attr-defined]


def argmin(self: object) -> Expr:
    """Index of the smallest element — the numpy ``argmin`` spelling of `arg_min`.

    Returns:
        An Int64 expression of the minimum's index.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"l": [[3, 1, 2]]})
            >>> ds.select(r=bt.col("l").list.argmin()).to_pydict()
            {'r': [1]}
    """
    return self.arg_min()  # type: ignore[attr-defined]


def argmax(self: object) -> Expr:
    """Index of the largest element — the numpy ``argmax`` spelling of `arg_max`.

    Returns:
        An Int64 expression of the maximum's index.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"l": [[3, 1, 2]]})
            >>> ds.select(r=bt.col("l").list.argmax()).to_pydict()
            {'r': [0]}
    """
    return self.arg_max()  # type: ignore[attr-defined]


# Accessor class attribute on `Expr` → the aliases to bind onto that namespace.
_BINDINGS: dict[str, tuple[object, ...]] = {
    "str": (isdigit, isalpha, isalnum, isspace, strip_prefix, strip_suffix),
    "dt": (day_of_week, day_of_year, week_of_year),
    "list": (lengths, element_at, argmin, argmax),
}


def bind_namespace_compat() -> None:
    """Attach every namespace alias onto its typed accessor class.

    Returns:
        None. Each accessor namespace gains its compatibility spellings.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.col("s").str.isdigit().to_ir() == bt.col("s").str.is_numeric().to_ir()
            True
    """
    from batcher.plan.expr_ir.namespaces import _DtNamespace, _ListNamespace, _StrNamespace

    classes = {"str": _StrNamespace, "dt": _DtNamespace, "list": _ListNamespace}
    for accessor, funcs in _BINDINGS.items():
        cls = classes[accessor]
        for func in funcs:
            func.__qualname__ = f"{cls.__name__}.{func.__name__}"  # type: ignore[attr-defined]
            setattr(cls, func.__name__, func)
