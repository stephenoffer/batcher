"""Predicate translation for source-side pushdown.

Kyber records the `Filter` sitting directly above a `Scan` as that source's
*pushed predicate* (`PhysicalPlan.source_predicates`). A pushdown-capable source
translates the **pushable subset** of that predicate IR into its backend filter
(a pyarrow `Expression`, a SQL `WHERE`, …) to skip I/O at the reader. The engine
keeps the `Filter` operator regardless, so a partial or absent translation is
always safe — it just reads more rows. This module owns the IR→backend mapping.

Pushable subset: comparisons (`= != < <= > >=`) between a column and a literal,
`IS NULL` / `IS NOT NULL`, and `AND`/`OR` of pushable terms. Anything else makes
the whole expression unpushable for that backend (returns ``None``).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "to_iceberg_expression",
    "to_mongo_filter",
    "to_pyarrow_expression",
    "to_sql_where",
]

_CMP = {"eq", "ne", "lt", "le", "gt", "ge"}
_SQL_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def _literal(ir: dict[str, Any]) -> Any:
    """Unwrap a literal IR ``{"e":"lit","value":{"int":5}}`` to its Python value."""
    return next(iter(ir["value"].values()))


def _col_and_literal(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, Any, bool] | None:
    """Return ``(column, value, flipped)`` for a column-vs-literal comparison."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], _literal(right), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], _literal(left), True
    return None


def to_pyarrow_expression(ir: dict[str, Any]) -> Any | None:
    """Translate the pushable subset of `ir` to a `pyarrow.dataset.Expression`.

    Returns ``None`` if the predicate is not (fully) pushable.
    """
    import pyarrow.dataset as ds

    return _to_pa(ir, ds)


def _to_pa(ir: dict[str, Any], ds: Any) -> Any | None:
    e = ir.get("e")
    if e == "is_null":
        inner = ir["input"]
        return ds.field(inner["name"]).is_null() if inner.get("e") == "col" else None
    if e == "is_not_null":
        inner = ir["input"]
        return ds.field(inner["name"]).is_valid() if inner.get("e") == "col" else None
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = _to_pa(ir["left"], ds)
        right = _to_pa(ir["right"], ds)
        if left is None or right is None:
            return None
        return (left & right) if op == "and" else (left | right)
    if op in _CMP:
        parsed = _col_and_literal(ir["left"], ir["right"])
        if parsed is None:
            return None
        col, value, flipped = parsed
        effective = _FLIP[op] if flipped else op
        field = ds.field(col)
        return {
            "eq": field == value,
            "ne": field != value,
            "lt": field < value,
            "le": field <= value,
            "gt": field > value,
            "ge": field >= value,
        }[effective]
    return None


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is None:
        return "NULL"
    return str(value)


def to_sql_where(ir: dict[str, Any]) -> str | None:
    """Translate the pushable subset of `ir` to a SQL ``WHERE`` fragment, or None."""
    e = ir.get("e")
    if e == "is_null" and ir["input"].get("e") == "col":
        return f"{ir['input']['name']} IS NULL"
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return f"{ir['input']['name']} IS NOT NULL"
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = to_sql_where(ir["left"])
        right = to_sql_where(ir["right"])
        if left is None or right is None:
            return None
        return f"({left} {op.upper()} {right})"
    if op in _CMP:
        parsed = _col_and_literal(ir["left"], ir["right"])
        if parsed is None:
            return None
        col, value, flipped = parsed
        effective = _FLIP[op] if flipped else op
        return f"{col} {_SQL_OP[effective]} {_sql_literal(value)}"
    return None


def to_iceberg_expression(ir: dict[str, Any]) -> Any | None:
    """Translate the pushable subset of `ir` to a `pyiceberg` row filter, or None."""
    from pyiceberg import expressions as ie

    cmp_ctor = {
        "eq": ie.EqualTo,
        "ne": ie.NotEqualTo,
        "lt": ie.LessThan,
        "le": ie.LessThanOrEqual,
        "gt": ie.GreaterThan,
        "ge": ie.GreaterThanOrEqual,
    }

    def walk(node: dict[str, Any]) -> Any | None:
        e = node.get("e")
        if e == "is_null" and node["input"].get("e") == "col":
            return ie.IsNull(node["input"]["name"])
        if e == "is_not_null" and node["input"].get("e") == "col":
            return ie.NotNull(node["input"]["name"])
        if e != "binary":
            return None
        op = node["op"]
        if op in ("and", "or"):
            left = walk(node["left"])
            right = walk(node["right"])
            if left is None or right is None:
                return None
            return ie.And(left, right) if op == "and" else ie.Or(left, right)
        if op in _CMP:
            parsed = _col_and_literal(node["left"], node["right"])
            if parsed is None:
                return None
            col, value, flipped = parsed
            effective = _FLIP[op] if flipped else op
            return cmp_ctor[effective](col, value)
        return None

    return walk(ir)


_MONGO_OP = {"eq": "$eq", "ne": "$ne", "lt": "$lt", "le": "$lte", "gt": "$gt", "ge": "$gte"}


def to_mongo_filter(ir: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to a MongoDB filter document, or None."""
    e = ir.get("e")
    if e == "is_null" and ir["input"].get("e") == "col":
        return {ir["input"]["name"]: None}
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return {ir["input"]["name"]: {"$ne": None}}
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = to_mongo_filter(ir["left"])
        right = to_mongo_filter(ir["right"])
        if left is None or right is None:
            return None
        return {f"${op}": [left, right]}
    if op in _CMP:
        parsed = _col_and_literal(ir["left"], ir["right"])
        if parsed is None:
            return None
        col, value, flipped = parsed
        effective = _FLIP[op] if flipped else op
        return {col: {_MONGO_OP[effective]: value}}
    return None
