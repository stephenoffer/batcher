"""The public selector constructors — ``bt.all()``, ``bt.numeric()``, ``bt.matches(...)``.

Each returns a `Selector` (from `core`); the dtype constructors defer their Arrow
type test through `_dtype_selector`. These are the names re-exported at ``bt.*``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pyarrow as pa

from batcher.plan.expr_ir.selectors.core import Selector

__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "contains",
    "ends_with",
    "exclude",
    "floating",
    "integer",
    "matches",
    "numeric",
    "starts_with",
    "string",
    "temporal",
]


def all() -> Selector:
    """Select every column of the input.

    Returns:
        A selector matching all columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2]})
            >>> ds.select(bt.all() * 10).to_pydict()
            {'a': [10], 'b': [20]}
    """
    return Selector(lambda _n, _d: True, "all()")


def exclude(*names: str) -> Selector:
    """Select every column except the named ones.

    Args:
        *names: The column names to leave out.

    Returns:
        A selector matching all columns but `names`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"id": [1], "a": [2], "b": [3]})
            >>> ds.select(bt.exclude("id")).columns
            ['a', 'b']
    """
    return all().exclude(*names)


def matches(pattern: str) -> Selector:
    """Select every column whose name matches a regular expression (via `re.search`).

    Args:
        pattern: A Python regular expression tested against each column name.

    Returns:
        A selector matching the columns whose names match `pattern`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"feat_a": [1], "feat_b": [2], "label": [3]})
            >>> ds.select(bt.matches("^feat_")).columns
            ['feat_a', 'feat_b']
    """
    compiled = re.compile(pattern)
    return Selector(lambda n, _d: compiled.search(n) is not None, f"matches({pattern!r})")


def starts_with(*prefixes: str) -> Selector:
    """Select every column whose name starts with any of the given prefixes.

    Args:
        *prefixes: One or more literal name prefixes to match.

    Returns:
        A selector matching the columns whose names start with a prefix.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x_a": [1], "x_b": [2], "y": [3]})
            >>> ds.select(bt.starts_with("x_")).columns
            ['x_a', 'x_b']
    """
    return Selector(lambda n, _d: n.startswith(prefixes), f"starts_with{prefixes!r}")


def ends_with(*suffixes: str) -> Selector:
    """Select every column whose name ends with any of the given suffixes.

    Args:
        *suffixes: One or more literal name suffixes to match.

    Returns:
        A selector matching the columns whose names end with a suffix.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a_id": [1], "b_id": [2], "name": [3]})
            >>> ds.select(bt.ends_with("_id")).columns
            ['a_id', 'b_id']
    """
    return Selector(lambda n, _d: n.endswith(suffixes), f"ends_with{suffixes!r}")


def contains(*substrings: str) -> Selector:
    """Select every column whose name contains any of the given substrings.

    Args:
        *substrings: One or more literal substrings to look for in the name.

    Returns:
        A selector matching the columns whose names contain a substring.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"user_id": [1], "order_id": [2], "ts": [3]})
            >>> ds.select(bt.contains("_id")).columns
            ['user_id', 'order_id']
    """
    return Selector(lambda n, _d: any(s in n for s in substrings), f"contains{substrings!r}")


def by_dtype(*dtypes: pa.DataType) -> Selector:
    """Select every column whose Arrow type is one of `dtypes`.

    The precise counterpart of the category selectors (`numeric`, `string`, …): pass
    the exact `pyarrow` types to match, e.g. ``bt.by_dtype(pa.int32(), pa.int64())``.

    Args:
        *dtypes: The Arrow data types to match, as ``pyarrow`` type objects.

    Returns:
        A selector matching columns of exactly those types.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow as pa
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5], "s": ["x"]})
            >>> ds.select(bt.by_dtype(pa.int64(), pa.string())).columns
            ['a', 's']
    """
    wanted = tuple(dtypes)
    return _dtype_selector(lambda d: any(d.equals(t) for t in wanted), f"by_dtype{wanted!r}")


def _dtype_selector(test: Callable[[pa.DataType], bool], desc: str) -> Selector:
    return Selector(lambda _n, d: d is not None and test(d), desc, needs_dtype=True)


def numeric() -> Selector:
    """Select every integer, floating-point, or decimal column.

    Returns:
        A selector matching the numeric columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5], "s": ["x"]})
            >>> ds.select(bt.numeric()).columns
            ['a', 'b']
    """
    return _dtype_selector(
        lambda d: pa.types.is_integer(d) or pa.types.is_floating(d) or pa.types.is_decimal(d),
        "numeric()",
    )


def integer() -> Selector:
    """Select every integer column.

    Returns:
        A selector matching the integer columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5]})
            >>> ds.select(bt.integer()).columns
            ['a']
    """
    return _dtype_selector(pa.types.is_integer, "integer()")


def floating() -> Selector:
    """Select every floating-point column.

    Returns:
        A selector matching the float columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5]})
            >>> ds.select(bt.floating()).columns
            ['b']
    """
    return _dtype_selector(pa.types.is_floating, "floating()")


def string() -> Selector:
    """Select every string column.

    Returns:
        A selector matching the string (and large-string) columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "s": ["x"]})
            >>> ds.select(bt.string().str.upper()).to_pydict()
            {'s': ['X']}
    """
    return _dtype_selector(
        lambda d: pa.types.is_string(d) or pa.types.is_large_string(d), "string()"
    )


def boolean() -> Selector:
    """Select every boolean column.

    Returns:
        A selector matching the boolean columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "ok": [True]})
            >>> ds.select(bt.boolean()).columns
            ['ok']
    """
    return _dtype_selector(pa.types.is_boolean, "boolean()")


def temporal() -> Selector:
    """Select every date, time, timestamp, or duration column.

    Returns:
        A selector matching the temporal columns.

    Examples:
        .. doctest::

            >>> import datetime
            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "d": [datetime.date(2024, 1, 1)]})
            >>> ds.select(bt.temporal()).columns
            ['d']
    """
    return _dtype_selector(pa.types.is_temporal, "temporal()")
