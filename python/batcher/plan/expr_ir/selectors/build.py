"""The public selector constructors — ``bt.all()``, ``bt.numeric()``, ``bt.matches(...)``.

Each returns a `Selector` (from `core`); the dtype constructors defer their Arrow
type test through `_dtype_selector`. These are the names re-exported at ``bt.*``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.selectors.core import Selector
from batcher.plan.types.lattice import widen

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


def by_dtype(*dtypes: pa.DataType | str) -> Selector:
    """Select every column whose Arrow type is one of `dtypes`, as the engine stores it.

    The precise counterpart of the category selectors (`numeric`, `string`, ...). Batcher
    widens narrow types once, when data enters the engine: every integer width becomes
    ``int64``, ``float16``/``float32`` become ``float64``, ``large_string`` and a
    dictionary-encoded string become ``string``. So a requested type is widened the same
    way before matching, and ``bt.by_dtype(pa.int32())`` selects the ``int64`` columns,
    including those that were ``int32`` in the source. The selector cannot tell a source
    ``int32`` from a source ``int64``, because the engine cannot either.

    Args:
        *dtypes: The Arrow data types to match, as ``pyarrow`` type objects or their
            names (``"float64"``).

    Returns:
        A selector matching columns of those types after widening.

    Raises:
        PlanError: If an argument is neither a ``pyarrow.DataType`` nor a type name.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow as pa
            >>> ds = bt.from_pydict({"a": [1], "b": [2.5], "s": ["x"]})
            >>> ds.select(bt.by_dtype(pa.int64(), "string")).columns
            ['a', 's']
            >>> narrow = bt.from_arrow(pa.table({"i": pa.array([1], pa.int32()), "s": ["x"]}))
            >>> narrow.select(bt.by_dtype(pa.int32())).columns
            ['i']
    """
    wanted = tuple(widen(_as_type(t)) for t in dtypes)
    return _dtype_selector(
        lambda d: any(widen(d).equals(t) for t in wanted), f"by_dtype{tuple(dtypes)!r}"
    )


def _as_type(dtype: object) -> pa.DataType:
    """`dtype` as a pyarrow type: a type passes through, a name is looked up."""
    if isinstance(dtype, pa.DataType):
        return dtype
    if isinstance(dtype, str):
        try:
            return pa.type_for_alias(dtype)
        except ValueError:
            pass
    raise PlanError(
        f"by_dtype() takes pyarrow types such as pa.int64() or their names such as 'int64', "
        f"got {type(dtype).__name__} {dtype!r}"
    )


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
