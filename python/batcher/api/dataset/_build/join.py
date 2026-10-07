"""The body of `Dataset.join`: key resolution, the opt-in keywords, and the `Join` node.

Every keyword here is a rewrite onto the one equi-join operator, so none adds IR and each
inherits the operator's spillable, mergeable and distributed forms:

* an **expression key** is materialized as a hidden column on its side and joined by name;
* ``nulls_equal`` replaces a key ``k`` with the pair ``(k IS NULL, coalesce(k, zero))``, so
  a null pairs with a null while a real zero cannot, because its null flag differs;
* ``indicator`` adds a constant ``TRUE`` marker to each side, which is null exactly where
  the join null-extended that side, whatever the payload holds;
* ``coalesce=False`` emits the right key as its own column instead of merging it;
* ``validate`` runs the `dq` uniqueness check on the side(s) that must be unique before the
  join is built, which is the one keyword that executes eagerly.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import DataQualityError, PlanError
from batcher.api._join_helpers import (
    KeySpec,
    _broadcast,
    _join_output,
    _resolve_key_specs,
)
from batcher.api.dataset._build.combine import unused_name
from batcher.plan.expr_ir import Coalesce, Col, Expr, coalesce, count, lit, when
from batcher.plan.logical import (
    Join,
    JoinOutputCol,
    Project,
    Projection,
    align_join_key_types,
    remap_sources,
)
from batcher.plan.types import dtype_name

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["JOIN_HOWS", "build_join", "validate_join_cardinality"]

#: The join types `Dataset.join` accepts, after ``"outer"`` is read as ``"full"``.
JOIN_HOWS = ("inner", "left", "right", "full", "semi", "anti")

#: ``validate=`` mode -> (left must be unique, right must be unique), pandas' spelling.
_VALIDATE = {
    "1:1": (True, True),
    "1:m": (True, False),
    "m:1": (False, True),
    "m:m": (False, False),
}

#: How many offending keys a cardinality error quotes.
_SHOWN_KEYS = 5


def build_join(
    left: Dataset,
    right: Dataset,
    on: KeySpec | list[KeySpec] | None,
    left_on: KeySpec | list[KeySpec] | None,
    right_on: KeySpec | list[KeySpec] | None,
    *,
    how: str,
    suffix: str,
    validate: str,
    nulls_equal: bool | Sequence[bool],
    coalesce_keys: bool | None,
    indicator: str | None,
) -> Dataset:
    """Equi-join `left` and `right` (see `Dataset.join` for the keywords).

    Args:
        left: The left relation.
        right: The right relation.
        on: Key(s) shared by both sides, names or expressions.
        left_on: The left key(s).
        right_on: The right key(s).
        how: inner/left/right/full/semi/anti (``"outer"`` and ``"cross"`` are resolved by
            the caller).
        suffix: Appended to a right column whose name a left column already has.
        validate: ``"1:1"``, ``"1:m"``, ``"m:1"`` or ``"m:m"``.
        nulls_equal: Whether a null key matches a null key, for every key or per key.
        coalesce_keys: Merge each named key pair into one column (`None` or `True`), or
            keep the right key as its own column (`False`).
        indicator: The name of a column saying which side(s) each row came from.

    Returns:
        The joined `Dataset`.

    Raises:
        PlanError: For an unknown `how` or `validate`, a keyword the join type cannot
            honour, or a key type ``nulls_equal`` cannot fill.
        DataQualityError: If `validate` finds a repeated key on a side that must be unique.
    """
    from batcher.api.dataset.frame import Dataset

    if how not in JOIN_HOWS:
        raise PlanError(
            f"unsupported join type {how!r} (inner|left|right|full|outer|cross|semi|anti)"
        )
    if validate not in _VALIDATE:
        raise PlanError(f"join(): validate must be one of {list(_VALIDATE)}, got {validate!r}")
    _check_how_keywords(how, coalesce_keys, indicator)
    lspecs, rspecs = _resolve_key_specs(on, left_on, right_on)
    null_safe = _broadcast(nulls_equal, len(lspecs), "nulls_equal")
    named = [
        (lk, rk)
        for lk, rk in zip(lspecs, rspecs, strict=True)
        if isinstance(lk, str) and isinstance(rk, str)
    ]
    # Widen each named key pair to its common type first, so the output key -- which a full
    # join coalesces from both sides -- has one type whichever side supplied it.
    lplan, rplan = align_join_key_types(
        left._plan, right._plan, tuple(k for k, _ in named), tuple(k for _, k in named)
    )
    lds, rds = Dataset(lplan, left._sources), Dataset(rplan, right._sources)
    lds, rds, lkeys, rkeys = _materialize_keys(lds, rds, lspecs, rspecs, left, right)
    labels = ([str(k) for k in lspecs], [str(k) for k in rspecs])
    validate_join_cardinality(lds, rds, (lkeys, rkeys), labels, validate, null_safe)
    lds, rds, lkeys, rkeys = _null_safe_keys(lds, rds, lkeys, rkeys, null_safe, left, right)
    markers = None
    if indicator is not None:
        markers = (
            unused_name("__bc_ind_l", left, right),
            unused_name("__bc_ind_r", left, right),
        )
        lds = lds.with_columns(**{markers[0]: lit(True)})
        rds = rds.with_columns(**{markers[1]: lit(True)})

    merge = coalesce_keys is not False
    output = _join_output(
        left.columns,
        right.columns,
        [k for k, _ in named],
        [k for _, k in named],
        how,
        suffix,
        coalesce=merge,
    )
    if markers is not None:
        output += [JoinOutputCol("left", markers[0], markers[0])]
        output += [JoinOutputCol("right", markers[1], markers[1])]
    lplan, rplan = align_join_key_types(
        lds._plan, remap_sources(rds._plan, len(left._sources)), tuple(lkeys), tuple(rkeys)
    )
    node = Join(lplan, rplan, tuple(lkeys), tuple(rkeys), how, tuple(output))
    sources = left._sources + right._sources
    if not ((how == "full" and merge) or markers is not None):
        return Dataset(node, sources)
    return Dataset(Project(node, _finish(node, named, how, merge, markers, indicator)), sources)


def _check_how_keywords(how: str, coalesce_keys: bool | None, indicator: str | None) -> None:
    """Refuse a keyword that means nothing for a semi or anti join, rather than ignore it."""
    if how not in {"semi", "anti"}:
        return
    if indicator is not None:
        raise PlanError(
            f"join(how={how!r}) cannot take indicator=: a {how} join returns only left rows, "
            "so every row would carry the same marker"
        )
    if coalesce_keys is False:
        raise PlanError(
            f"join(how={how!r}) cannot take coalesce=False: a {how} join returns only the "
            "left columns, so there is no right key to keep"
        )


def _materialize_keys(
    lds: Dataset,
    rds: Dataset,
    lspecs: list[KeySpec],
    rspecs: list[KeySpec],
    left: Dataset,
    right: Dataset,
) -> tuple[Dataset, Dataset, list[str], list[str]]:
    """Add each expression key as a hidden column on its own side; name every key.

    A pair with an expression on either side contributes no output key column: its hidden
    columns are never emitted, and the columns the expression reads stay as ordinary
    payload. So the output is the same whatever the expression computes.
    """
    lkeys: list[str] = []
    rkeys: list[str] = []
    ladd: dict[str, Expr] = {}
    radd: dict[str, Expr] = {}
    for i, (lk, rk) in enumerate(zip(lspecs, rspecs, strict=True)):
        if isinstance(lk, str) and isinstance(rk, str):
            lkeys.append(lk)
            rkeys.append(rk)
            continue
        lname = unused_name(f"__bc_jkey_l{i}", left, right)
        rname = unused_name(f"__bc_jkey_r{i}", left, right)
        ladd[lname] = Col(lk) if isinstance(lk, str) else lk
        radd[rname] = Col(rk) if isinstance(rk, str) else rk
        lkeys.append(lname)
        rkeys.append(rname)
    if ladd:
        lds, rds = lds.with_columns(**ladd), rds.with_columns(**radd)
    return lds, rds, lkeys, rkeys


def _null_safe_keys(
    lds: Dataset,
    rds: Dataset,
    lkeys: list[str],
    rkeys: list[str],
    null_safe: list[bool],
    left: Dataset,
    right: Dataset,
) -> tuple[Dataset, Dataset, list[str], list[str]]:
    """Replace each null-safe key with its ``(is null, filled value)`` pair of hidden keys.

    The equi-join drops a null key on every path (dense, radix, sort-merge, streaming), so
    a null-safe comparison is expressed in values the join does compare. The fill value is
    the type's zero; it cannot collide with a real zero because the null flags differ, and
    both sides fill with the same value because their key types were aligned first.
    """
    if not any(null_safe):
        return lds, rds, lkeys, rkeys
    lschema, rschema = lds.schema, rds.schema
    out_l: list[str] = []
    out_r: list[str] = []
    ladd: dict[str, Expr] = {}
    radd: dict[str, Expr] = {}
    for i, (lk, rk, safe) in enumerate(zip(lkeys, rkeys, null_safe, strict=True)):
        if not safe:
            out_l.append(lk)
            out_r.append(rk)
            continue
        names = [unused_name(f"__bc_nk_{s}{i}", left, right) for s in ("ln", "lv", "rn", "rv")]
        ladd[names[0]] = Col(lk).is_null()
        ladd[names[1]] = coalesce(Col(lk), _fill_value(lschema.field(lk).type, lk))
        radd[names[2]] = Col(rk).is_null()
        radd[names[3]] = coalesce(Col(rk), _fill_value(rschema.field(rk).type, rk))
        out_l += names[:2]
        out_r += names[2:]
    return lds.with_columns(**ladd), rds.with_columns(**radd), out_l, out_r


def _fill_value(dtype: pa.DataType, key: str) -> Expr:
    """The constant a null-safe key's nulls are filled with: the zero of `dtype`."""
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return lit("")
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return lit("").cast("binary")
    name = dtype_name(dtype)
    zero_castable = (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_boolean(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_temporal(dtype)
    )
    if name is None or not zero_castable:
        raise PlanError(
            f"join(nulls_equal=True) cannot compare key {key!r} of type {dtype} null-safely; "
            "it supports numeric, boolean, decimal, string, binary and temporal keys. Cast "
            "the key, or join with nulls_equal=False"
        )
    return lit(0).cast(name)


def _finish(
    node: Join,
    named: list[tuple[str, str]],
    how: str,
    merge: bool,
    markers: tuple[str, str] | None,
    indicator: str | None,
) -> tuple[Projection, ...]:
    """The projection over a join that coalesces full-join keys and computes `indicator`."""
    temps = {*(markers or ())}
    items: list[Projection] = []
    if how == "full" and merge:
        for i, (lk, _) in enumerate(named):
            items.append(Projection(lk, Coalesce([Col(f"__fk_l_{i}"), Col(f"__fk_r_{i}")])))
            temps |= {f"__fk_l_{i}", f"__fk_r_{i}"}
    items += [Projection(c, Col(c)) for c in node.available_columns() if c not in temps]
    if markers is not None and indicator is not None:
        if indicator in {p.alias for p in items}:
            raise PlanError(
                f"join(indicator={indicator!r}) names a column the join already outputs; "
                "pick another name"
            )
        lm, rm = Col(markers[0]).is_not_null(), Col(markers[1]).is_not_null()
        side = (
            when(lm & rm)
            .then(lit("both"))
            .when(lm)
            .then(lit("left_only"))
            .otherwise(lit("right_only"))
        )
        items.append(Projection(indicator, side))
    return tuple(items)


def validate_join_cardinality(
    left: Dataset,
    right: Dataset,
    keys: tuple[list[str], list[str]],
    labels: tuple[list[str], list[str]],
    validate: str,
    null_safe: list[bool],
) -> None:
    """Raise if a side `validate` requires to be unique repeats a key.

    The check is the `dq` uniqueness constraint (``ds.dq.unique(keys)``), run eagerly. A
    row with a null in a key that is not null-safe is left out first: it matches nothing,
    so it cannot multiply a row of the other side.

    Args:
        left: The left relation, expression keys already materialized.
        right: The right relation, likewise.
        keys: The left and right key column names.
        labels: The keys as the caller wrote them, for the message (an expression key's
            column is a hidden one).
        validate: ``"1:1"``, ``"1:m"``, ``"m:1"`` or ``"m:m"``.
        null_safe: Per key, whether nulls compare equal.

    Raises:
        DataQualityError: Naming the side and up to five repeated keys.
    """
    for i, (side, ds) in enumerate((("left", left), ("right", right))):
        if not _VALIDATE[validate][i]:
            continue
        cols = keys[i]
        checked = ds.select(*cols)
        matchable = [
            Col(k).is_not_null() for k, safe in zip(cols, null_safe, strict=True) if not safe
        ]
        if matchable:
            checked = checked.filter(reduce(operator.and_, matchable))
        report = checked.dq.unique(cols).validate()
        if report.ok:
            continue
        repeated = (
            checked.group_by(*cols)
            .agg(__bc_n=count())
            .filter(Col("__bc_n") > 1)
            .limit(_SHOWN_KEYS)
            .to_pydict()
        )
        values = [repeated[k] for k in cols]
        shown = values[0] if len(cols) == 1 else list(zip(*values, strict=True))
        raise DataQualityError(
            f"join(validate={validate!r}): the {side} side must have unique keys "
            f"{labels[i]}, but {report.total_violations} of its rows share a key, e.g. "
            f"{shown}. Deduplicate that side (distinct(..., keep=...)) or relax validate=",
            violations=report.violations,
        )
