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
    "to_native_predicate",
    "to_pyarrow_expression",
    "to_sql_where",
]

_CMP = {"eq", "ne", "lt", "le", "gt", "ge"}
_SQL_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def _literal(ir: dict[str, Any]) -> Any:
    """Unwrap a literal IR ``{"e":"lit","value":{"int":5}}`` to its Python value.

    Temporal kinds (``date`` days, ``timestamp`` micros, ``time`` micros) unwrap to a
    plain Python ``date``/``datetime`` so a backend that types its own scalars (SQL,
    iceberg, mongo) gets a real temporal value, not a raw epoch offset.
    """
    ((kind, value),) = ir["value"].items()
    if kind == "date":
        import datetime as _dt

        return _dt.date(1970, 1, 1) + _dt.timedelta(days=value)
    if kind == "timestamp":
        import datetime as _dt

        return _dt.datetime(1970, 1, 1) + _dt.timedelta(microseconds=value)
    return value


def _pa_literal(ir: dict[str, Any]) -> Any:
    """A pyarrow scalar for a literal IR, typed for temporal kinds.

    A bare ``date``/``timestamp`` literal is an epoch offset (days / micros); handed to
    pyarrow as a Python ``int`` it infers ``int16``/``int64`` and the comparison kernel
    against a ``date32``/``timestamp`` column has no match (``greater_equal(date32,
    int16)``). Building an explicitly-typed ``date32``/``timestamp[us]`` scalar makes the
    column-vs-literal comparison type-check and enables row-group/page pruning on date
    columns (the common TPC-H shipdate/orderdate filters).
    """
    import pyarrow as pa

    ((kind, value),) = ir["value"].items()
    if kind == "date":
        return pa.scalar(value, pa.date32())
    if kind == "timestamp":
        return pa.scalar(value, pa.timestamp("us"))
    if kind == "time":
        return pa.scalar(value, pa.time64("us"))
    return value


def _col_and_literal(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, Any, bool] | None:
    """Return ``(column, value, flipped)`` for a column-vs-literal comparison."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], _literal(right), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], _literal(left), True
    return None


def _col_and_pa_literal(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[str, Any, bool] | None:
    """Like :func:`_col_and_literal`, but the value is a typed pyarrow scalar."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], _pa_literal(right), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], _pa_literal(left), True
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
        parsed = _col_and_pa_literal(ir["left"], ir["right"])
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


def _native_scalar(ir: dict[str, Any]) -> tuple[Any, bool]:
    """A literal for the native reader: ``(value, ok)``.

    Only plain ``int``/``float``/``str``/``bool`` literals push to the native reader's
    zone-map pruning. Temporal kinds (``date``/``timestamp``/``time``) are epoch offsets
    whose parquet physical unit the reader cannot verify without risking an unsound prune,
    so they mark the term non-pushable (``ok=False``) and the pyarrow path handles them.
    """
    ((kind, value),) = ir["value"].items()
    if kind in ("int", "float", "str", "bool"):
        return value, True
    return None, False


def to_native_predicate(ir: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to the native reader's compact predicate.

    The shape `bc_io`'s `predicate` module deserializes: ``{"node":"cmp","col":..,"op":..,
    "lit":..}`` / ``{"node":"and"/"or","left":..,"right":..}`` / ``{"node":"null","col":..,
    "negated":..}``. Comparisons are normalized so the column is on the left. Returns
    ``None`` if any term is not pushable (a non-column/literal comparison, a temporal
    literal, or an unsupported node) — the caller then reads without native pruning.
    """
    e = ir.get("e")
    if e in ("is_null", "is_not_null"):
        inner = ir["input"]
        if inner.get("e") != "col":
            return None
        return {"node": "null", "col": inner["name"], "negated": e == "is_not_null"}
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = to_native_predicate(ir["left"])
        right = to_native_predicate(ir["right"])
        if left is None or right is None:
            return None
        return {"node": op, "left": left, "right": right}
    if op in _CMP:
        left, right = ir["left"], ir["right"]
        if left.get("e") == "col" and right.get("e") == "lit":
            col, lit_ir, flipped = left["name"], right, False
        elif left.get("e") == "lit" and right.get("e") == "col":
            col, lit_ir, flipped = right["name"], left, True
        else:
            return None
        value, ok = _native_scalar(lit_ir)
        if not ok:
            return None
        return {"node": "cmp", "col": col, "op": _FLIP[op] if flipped else op, "lit": value}
    return None


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
