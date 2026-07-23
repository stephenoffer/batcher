"""Frame combination: the polymorphic `concat`.

``concat([df1, df2])`` means *frame* concatenation everywhere in the Python data
ecosystem — pandas, Polars, and PyArrow all spell it that way — while the string
builder is ``concat_str``. This module owns the frame side and dispatches the
expression side to `batcher.plan.functions.concat`, so both idioms work under the
one name a user will reach for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset

__all__ = ["concat"]

_HOWS = ("vertical", "vertical_relaxed", "diagonal", "horizontal")

# The internal row key `how="horizontal"` positions rows on. Chosen to be unusable as
# a real column name so it can never collide with a user's schema.
_POSITION = "__bc_concat_position__"


def concat(*items: Any, how: str = "vertical", rechunk: bool = False) -> Any:
    """Concatenate datasets into one `Dataset`, or values into one string expression.

    ``bt.concat([ds1, ds2])`` stacks datasets, the way ``pd.concat`` and
    ``pl.concat`` do. ``bt.concat(bt.col("a"), bt.col("b"))`` builds a string
    expression, the way SQL ``concat`` does; `bt.concat_str` is the explicit
    spelling of that form. The arguments decide which one you get, and mixing the
    two kinds is an error rather than a guess.

    `how` selects the frame strategy, using Polars' vocabulary:

    * ``"vertical"`` — stack rows; every dataset must have the same columns.
    * ``"vertical_relaxed"`` — stack rows, deduplicating the result (SQL ``UNION``).
    * ``"diagonal"`` — stack rows over the *union* of the columns, filling a column
      a dataset does not have with nulls.
    * ``"horizontal"`` — place the datasets side by side, matching rows by position
      and padding the shorter ones with nulls. Column names must not collide.

    Args:
        *items: The datasets to stack (as one sequence or as separate arguments), or
            the expressions/strings to join into text.
        how: The frame concatenation strategy; ignored for the expression form.
        rechunk: Accepted for Polars signature compatibility; Batcher chooses its own
            morsel layout, so this has no effect.

    Returns:
        A lazy `Dataset` for the frame form, or an `Expr` for the string form.

    Raises:
        PlanError: If `items` is empty, mixes datasets with expressions, or `how` is
            not one of the four strategies.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> a = bt.from_pydict({"x": [1, 2]})
            >>> b = bt.from_pydict({"x": [3, 4]})
            >>> bt.concat([a, b]).to_pydict()
            {'x': [1, 2, 3, 4]}

            >>> left = bt.from_pydict({"x": [1]})
            >>> right = bt.from_pydict({"y": ["a"]})
            >>> bt.concat([left, right], how="diagonal").to_pydict()
            {'x': [1, None], 'y': [None, 'a']}

            >>> ds = bt.from_pydict({"a": ["x"], "b": ["y"]})
            >>> ds.select(c=bt.concat(bt.col("a"), bt.col("b"))).to_pydict()
            {'c': ['xy']}
    """
    del rechunk  # accepted for signature compatibility; the engine batches its own way
    flat = _flatten(items)
    if not flat:
        raise PlanError("concat() requires at least one dataset or value")
    datasets = [item for item in flat if isinstance(item, Dataset)]
    if not datasets:
        from batcher.plan.functions import concat as concat_expressions

        return concat_expressions(*flat)
    if len(datasets) != len(flat):
        raise PlanError(
            "concat() got a mix of Datasets and expressions. Concatenate frames with "
            "bt.concat([ds1, ds2]), and build text with bt.concat_str(...)."
        )
    if how not in _HOWS:
        raise PlanError(f"concat(): how must be one of {_HOWS}, got {how!r}")
    if len(datasets) == 1:
        return datasets[0]
    if how == "horizontal":
        return _horizontal(datasets)
    if how == "diagonal":
        datasets = _widen(datasets)
    return datasets[0].union(*datasets[1:], distinct=how == "vertical_relaxed")


def _flatten(items: tuple[Any, ...]) -> list[Any]:
    """Accept both ``concat([a, b])`` and ``concat(a, b)``, returning a flat list."""
    if (
        len(items) == 1
        and isinstance(items[0], Sequence)
        and not isinstance(items[0], (str, bytes))
    ):
        return list(items[0])
    return list(items)


def _typed_null(dtype: Any) -> Any:
    """A NULL expression of `dtype`, for a column a diagonal input does not have.

    The IR carries no null literal (adding one is a wire-contract change), so the null
    is built as ``nullif(0, 0)`` and cast to the target type — which the engine folds
    to a typed null column.
    """
    from batcher.plan.expr_ir import lit, nullif

    name = _CAST_NAMES.get(str(dtype))
    if name is None:
        raise PlanError(
            f"concat(how='diagonal'): cannot synthesize a null {dtype} column. Add the "
            "missing column explicitly with with_columns() before concatenating, or use "
            "how='vertical' once every input has the same schema."
        )
    return nullif(lit(0), lit(0)).cast(name)


# Arrow type name to the cast name that produces a null of that type. Only the types the
# engine can cast to are here; anything else is reported rather than silently retyped.
_CAST_NAMES = {
    "bool": "bool",
    "int8": "int64",
    "int16": "int64",
    "int32": "int64",
    "int64": "int64",
    "uint8": "int64",
    "uint16": "int64",
    "uint32": "int64",
    "uint64": "int64",
    "halffloat": "float64",
    "float": "float64",
    "double": "float64",
    "string": "string",
    "large_string": "string",
    "date32[day]": "date32",
    "date64[ms]": "date32",
    "timestamp[us]": "timestamp",
    "timestamp[ns]": "timestamp",
    "timestamp[ms]": "timestamp",
    "timestamp[s]": "timestamp",
}


def _widen(datasets: list[Dataset]) -> list[Dataset]:
    """Give every dataset the union of all columns, filling the missing ones with nulls."""
    order: list[str] = []
    types: dict[str, Any] = {}
    for ds in datasets:
        for name, dtype in zip(ds.columns, ds.dtypes, strict=True):
            if name not in types:
                order.append(name)
                types[name] = dtype

    widened: list[Dataset] = []
    for ds in datasets:
        present = set(ds.columns)
        missing = {n: _typed_null(types[n]) for n in order if n not in present}
        if missing:
            ds = ds.with_columns(**missing)
        widened.append(ds.select(*order))
    return widened


def _horizontal(datasets: list[Dataset]) -> Dataset:
    """Place datasets side by side, matching rows by position (Polars ``horizontal``)."""
    seen: set[str] = set()
    for ds in datasets:
        clash = seen & set(ds.columns)
        if clash:
            raise PlanError(
                f"concat(how='horizontal'): column name(s) {sorted(clash)} appear in more "
                "than one dataset. Rename them first — horizontal concatenation places "
                "columns side by side and cannot merge two columns of the same name."
            )
        seen |= set(ds.columns)

    out = datasets[0].with_row_index(_POSITION)
    for other in datasets[1:]:
        out = out.join(other.with_row_index(_POSITION), on=_POSITION, how="full")
    return out.sort(_POSITION).drop(_POSITION)
