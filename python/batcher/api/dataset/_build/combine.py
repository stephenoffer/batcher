"""Bodies of the `Dataset` verbs that combine two or more relations over existing operators.

`join_where` is a theta join, `update` a keyed overwrite, `zip` a positional pairing. None
adds an operator. `join_where` lowers to the cartesian join plus filter that Kyber's
`derive_range_join` already turns into a `RangeJoin` (or, for an equality, a hash join);
`update` is a join and a coalesce; `zip` numbers each input under an explicit order and
joins on the number. So each one inherits the mergeable, spillable and distributed forms of
the operators it is built from, and single-node equals distributed by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api._join_helpers import _resolve_join_keys
from batcher.plan.expr_ir import Col, Expr, coalesce, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["OrderSpec", "build_join_where", "build_update", "build_zip", "rank_rows", "unused_name"]

#: What an explicit row order may be given as: a column name or expression, or several.
OrderSpec = str | Expr | Sequence[str | Expr]

#: The join types `update` accepts, which are Polars' three.
_UPDATE_HOWS = ("left", "inner", "full")


def unused_name(stem: str, *datasets: Dataset) -> str:
    """`stem`, lengthened with underscores until no column of any of `datasets` has it.

    Every helper column these verbs add is dropped before the result is returned, and it
    must never shadow a real column in between: `with_columns` replaces a same-named column,
    so a clash would silently overwrite the user's data and then delete it.

    Args:
        stem: The preferred hidden-column name.
        *datasets: Every relation whose columns the name must avoid.

    Returns:
        A name absent from all of them.
    """
    taken = {c for ds in datasets for c in ds.columns}
    name = stem
    while name in taken:
        name += "_"
    return name


def rank_rows(
    ds: Dataset, order_by: OrderSpec, descending: bool | Sequence[bool], *, name: str, api: str
) -> Dataset:
    """`ds` with a 1-based ``row_number`` column `name` under the explicit order `order_by`.

    This is the one place a positional verb turns "row position" into something a
    relation can answer. A relation has no order of its own (a morselized or distributed
    scan fixes none), so every positional verb takes `order_by` and ranks under it; for
    data with no ordering column, `with_row_index` at the source numbers rows in source
    order.

    Args:
        ds: The relation to number.
        order_by: The ordering keys.
        descending: Order every key, or each one, largest first.
        name: The rank column to add.
        api: The public method name, quoted in errors.

    Returns:
        `ds` with the rank column appended.

    Raises:
        PlanError: If `order_by` is empty.
    """
    from batcher.plan.expr_ir.nodes import row_number

    keys = [order_by] if isinstance(order_by, (str, Expr)) else list(order_by)
    if not keys:
        raise PlanError(
            f"{api}(): order_by must name at least one ordering key -- a relation has no row "
            "order of its own; number rows at the source with with_row_index('i') and pass "
            "order_by='i'"
        )
    rank = row_number().over(order_by=keys, descending=descending)
    return ds.with_columns(**{name: rank})


def build_join_where(
    left: Dataset, right: Dataset, predicates: tuple[Expr, ...], suffix: str
) -> Dataset:
    """Inner-join `left` and `right` on the conjunction of `predicates` (see `join_where`).

    Args:
        left: The left relation.
        right: The right relation.
        predicates: Boolean expressions over both sides' columns, ANDed together.
        suffix: Appended to a right column whose name collides with a left one.

    Returns:
        The rows of the cartesian product for which every predicate is true.

    Raises:
        PlanError: If no predicate is given, or one is not an expression.
    """
    flat: list[Any] = []
    for p in predicates:
        flat.extend(p if isinstance(p, (list, tuple)) else [p])
    if not flat:
        raise PlanError(
            "join_where() requires at least one predicate, e.g. "
            "a.join_where(b, bt.col('start') <= bt.col('t'), bt.col('t') < bt.col('end')); "
            "for an unconditional pairing use cross_join()"
        )
    for p in flat:
        if not isinstance(p, Expr):
            raise PlanError(
                f"join_where() predicates must be expressions such as bt.col('a') < "
                f"bt.col('b'), got {type(p).__name__}"
            )
    condition = reduce(lambda acc, p: acc & p, flat[1:], flat[0])
    return left.cross_join(right, suffix=suffix).filter(condition)


def build_update(
    ds: Dataset,
    other: Dataset,
    on: str | list[str] | None,
    how: str,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
    include_nulls: bool,
) -> Dataset:
    """Overwrite `ds`'s values from `other` where the keys match (see `Dataset.update`).

    Args:
        ds: The relation being updated.
        other: The relation supplying new values.
        on: Shared key column(s).
        how: ``"left"``, ``"inner"`` or ``"full"``.
        left_on: `ds`'s key column(s), when the names differ.
        right_on: `other`'s key column(s), when the names differ.
        include_nulls: Let a null in `other` overwrite a value.

    Returns:
        `ds`'s columns, in `ds`'s order, with matched values replaced.

    Raises:
        PlanError: If `how` is unknown or no key is given.
    """
    if how not in _UPDATE_HOWS:
        raise PlanError(f"update(): how must be one of {list(_UPDATE_HOWS)}, got {how!r}")
    if on is None and left_on is None and right_on is None:
        raise PlanError(
            "update() needs a key: on=... (or left_on=/right_on=). Pairing rows by position "
            "needs an explicit order -- number both sides with with_row_index('i') at their "
            "sources and pass on='i'"
        )
    left_keys, right_keys = _resolve_join_keys(on, left_on, right_on)
    # A key is never overwritten, on either side's spelling of it: the join already decides
    # which rows the key values belong to.
    keys = {*left_keys, *right_keys}
    mine = set(ds.columns)
    shared = [c for c in other.columns if c in mine and c not in keys]
    matched = unused_name("__bc_update_matched", ds, other)
    hidden = {c: unused_name(f"__bc_update_{i}", ds, other) for i, c in enumerate(shared)}
    source = other.select(
        *(Col(k) for k in right_keys),
        *(Col(c).alias(h) for c, h in hidden.items()),
        lit(True).alias(matched),
    )
    joined = ds.join(source, left_on=left_keys, right_on=right_keys, how=how)
    new_values = {}
    for c, h in hidden.items():
        if include_nulls:
            new_values[c] = when(Col(matched).is_not_null()).then(Col(h)).otherwise(Col(c))
        else:
            new_values[c] = coalesce(Col(h), Col(c))
    return joined.with_columns(**new_values).select(*ds.columns)


def build_zip(
    ds: Dataset,
    others: tuple[Dataset, ...],
    order_by: OrderSpec,
    descending: bool | Sequence[bool],
) -> Dataset:
    """Pair the rows of `ds` and `others` by position under `order_by` (see `Dataset.zip`).

    Args:
        ds: The first relation.
        others: The relations zipped onto it, left to right.
        order_by: The ordering keys every input is ranked by.
        descending: Order every key, or each one, largest first.

    Returns:
        One row per position: `ds`'s columns, then each other's, in position order.

    Raises:
        PlanError: If no other dataset is given, or the row counts differ.
    """
    if not others:
        raise PlanError("zip() needs at least one other dataset to zip with")
    inputs = (ds, *others)
    counts = [d.count() for d in inputs]
    if len(set(counts)) != 1:
        raise PlanError(
            f"zip() pairs rows by position, so every dataset must have the same number of "
            f"rows; got {counts}"
        )
    pos = unused_name("__bc_zip_pos", *inputs)
    out = rank_rows(ds, order_by, descending, name=pos, api="zip")
    names = [*ds.columns]
    for other in others:
        ranked = rank_rows(other, order_by, descending, name=pos, api="zip")
        renamed = {}
        for c in other.columns:
            # Ray Data's rule: a repeated name takes the smallest free `_1`, `_2`, ... suffix.
            new, k = c, 0
            while new in names:
                k += 1
                new = f"{c}_{k}"
            names.append(new)
            renamed[c] = new
        ranked = ranked.select(Col(pos), *(Col(c).alias(n) for c, n in renamed.items()))
        out = out.join(ranked, on=pos, how="inner")
    return out.sort(pos).select(*names)
