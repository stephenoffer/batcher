"""A readable ``repr`` for the scalar `Expr` tree.

``repr(col("x") + 1)`` should read like the code that built it — ``(col('x') + 1)``
— not leak node class names or memory addresses. `render_expr` walks the node tree
and reconstructs that source-like form; `Expr.__repr__` delegates here. It is a
display aid only (never parsed back), so an unusual node degrades to a clean generic
``Name(child, …)`` form rather than failing.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

# Binary/`op` tag → the Python operator that built it.
_BINOP_SYM = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floor_div": "//",
    "mod": "%",
    "pow": "**",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "&",
    "or": "|",
    "xor": "^",
    "shl": "<<",
    "shr": ">>",
}

# Accessor-function node class → (namespace, method name) it is reached through, so a
# `StrFunc('upper', col('x'))` renders as ``col('x').str.upper()``. A ``None`` method
# means the node carries its own `fn` field naming the call; the rest spell the method
# out because the node's shape does not name it (``ListGet`` is ``.list.get``).
_NS_PREFIX: dict[str, tuple[str, str | None]] = {
    "StrFunc": ("str", None),
    "DateFunc": ("dt", None),
    "DateTrunc": ("dt", "truncate"),
    "Strftime": ("dt", "strftime"),
    "Strptime": ("dt", "strptime"),
    "ConvertTimezone": ("dt", "convert_timezone"),
    "DateOffset": ("dt", "offset_by"),
    "ListFunc": ("list", None),
    "ListGet": ("list", "get"),
    "ListContains": ("list", "contains"),
    "ListPosition": ("list", "position"),
    "ListSlice": ("list", "slice"),
    "ListSimhash": ("list", "simhash"),
    "ListBinary": ("list", None),
    "ListSet": ("list", None),
    "ListZip": ("list", None),
    "ListTransform": ("list", "transform"),
    "ListFilter": ("list", "filter"),
    "StructField": ("struct", "field"),
    "MapFunc": ("map", None),
}

# Engine `fn` name → the accessor method that builds it, where the two differ. The
# node stores the engine's vocabulary (`element_at`, `array_union`); the repr should
# echo the method the user actually typed. Only mismatches are listed.
_FN_METHOD = {
    "element_at": "get",
    "map_keys": "keys",
    "map_values": "values",
    "array_union": "union",
    "array_intersect": "intersect",
    "array_except": "difference",
    "list_add": "add",
    "list_subtract": "subtract",
    "list_multiply": "multiply",
    "hamming": "hamming_distance",
}

_MAX_DEPTH = 6


def render_expr(expr: Any, depth: int = 0) -> str:
    """Render an `Expr` node as a source-like string (used by ``Expr.__repr__``)."""
    cls = type(expr).__name__
    if depth > _MAX_DEPTH:
        return "…"
    # A node that declares its own `__repr__` knows something this module does not —
    # `StrFunc` redacts an inline crypto key. Defer to it so a *nested* occurrence is
    # redacted too, rather than only a top-level one.
    own_repr = type(expr).__dict__.get("__repr__")
    if own_repr is not None:
        return own_repr(expr)
    r = _RENDERERS.get(cls)
    if r is not None:
        return r(expr, depth)
    if cls in _NS_PREFIX:
        return _render_accessor(expr, depth)
    return _render_generic(expr, depth)


# `core` imports this module for `Expr.__repr__`, so the base class cannot be imported at
# module level here — but re-importing it per rendered node meant the import machinery ran
# once per node of every expression printed. Bound on first use instead.
_EXPR_CLASS: type | None = None


def _expr_class() -> type:
    """The `Expr` base class, imported once (see the note above)."""
    global _EXPR_CLASS
    if _EXPR_CLASS is None:
        from batcher.plan.expr_ir.core import Expr

        _EXPR_CLASS = Expr
    return _EXPR_CLASS


def _kids(expr: Any) -> list[str]:
    """Rendered sub-expressions of a node, discovered by reflection."""
    Expr = _expr_class()

    out: list[str] = []
    if is_dataclass(expr):
        for f in fields(expr):
            v = getattr(expr, f.name)
            if isinstance(v, Expr):
                out.append(render_expr(v, 1))
            elif isinstance(v, (list, tuple)):
                out.extend(render_expr(e, 1) for e in v if isinstance(e, Expr))
    return out


def _render_generic(expr: Any, _depth: int) -> str:
    inner = ", ".join(_kids(expr))
    return f"{type(expr).__name__}({inner})"


def _render_accessor(expr: Any, depth: int) -> str:
    """Render an accessor node as ``<base>.<ns>.<method>(<args>)``.

    The receiver is the node's first expression-typed field — ``input`` for the
    unary nodes, ``left`` for the two-column ones (``ListBinary``/``ListSet``) —
    found by reflection rather than by name, so a node whose shape differs renders
    instead of raising. Every remaining set field becomes an argument, which is what
    makes the form round-trip visually: without them ``.str.contains('a')`` would
    render as ``.contains()`` and lose the very thing being searched for.
    """
    ns, method = _NS_PREFIX[type(expr).__name__]
    Expr = _expr_class()
    node_fields = list(fields(expr)) if is_dataclass(expr) else []
    base_field = next((f for f in node_fields if isinstance(getattr(expr, f.name), Expr)), None)
    if base_field is None:  # no receiver to hang the call off — degrade, don't raise
        return _render_generic(expr, depth)
    base = render_expr(getattr(expr, base_field.name), depth + 1)
    fn = getattr(expr, "fn", None)
    name = method or _FN_METHOD.get(fn, fn) or "?"
    args = [
        render_expr(v, depth + 1) if isinstance(v, Expr) else repr(v)
        for f in node_fields
        if f.name not in ("fn", base_field.name) and (v := getattr(expr, f.name)) is not None
    ]
    return f"{base}.{ns}.{name}({', '.join(args)})"


# --- precise per-node renderers -----------------------------------------------------


def _r_col(e: Any, _d: int) -> str:
    return f"col({e.name!r})"


def _r_lit(e: Any, _d: int) -> str:
    return f"lit({e.value!r})"


def _r_binary(e: Any, d: int) -> str:
    sym = _BINOP_SYM.get(e.op, e.op)
    return f"({render_expr(e.left, d + 1)} {sym} {render_expr(e.right, d + 1)})"


def _r_aliased(e: Any, d: int) -> str:
    return f"{render_expr(e.inner, d + 1)}.alias({e.name!r})"


def _r_not(e: Any, d: int) -> str:
    return f"~{render_expr(e.input, d + 1)}"


def _r_cast(e: Any, d: int) -> str:
    fn = "try_cast" if getattr(e, "try_cast", False) else "cast"
    return f"{render_expr(e.input, d + 1)}.{fn}({e.dtype!r})"


def _r_unary_method(name: str):
    def render(e: Any, d: int) -> str:
        return f"{render_expr(e.input, d + 1)}.{name}()"

    return render


def _r_agg(e: Any, d: int) -> str:
    if e.input is None:
        return f"{e.func}()"
    return f"{render_expr(e.input, d + 1)}.{e.func}()"


def _r_math(e: Any, d: int) -> str:
    return f"{render_expr(e.input, d + 1)}.{e.fn}()"


def _r_variadic(fn: str):
    def render(e: Any, d: int) -> str:
        parts = ", ".join(render_expr(x, d + 1) for x in e.inputs)
        return f"{fn}({parts})"

    return render


def _r_inlist(e: Any, d: int) -> str:
    vals = ", ".join(repr(v) for v in e.values)
    return f"{render_expr(e.input, d + 1)}.is_in([{vals}])"


def _r_case(e: Any, d: int) -> str:
    parts = "".join(
        f"when({render_expr(c, d + 1)}).then({render_expr(t, d + 1)})" for c, t in e.branches
    )
    return f"{parts}.otherwise({render_expr(e.otherwise, d + 1)})"


_RENDERERS = {
    "Col": _r_col,
    "Lit": _r_lit,
    "Binary": _r_binary,
    "Aliased": _r_aliased,
    "Not": _r_not,
    "Cast": _r_cast,
    "IsNull": _r_unary_method("is_null"),
    "IsNotNull": _r_unary_method("is_not_null"),
    "IsNan": _r_unary_method("is_nan"),
    "IsInf": _r_unary_method("is_infinite"),
    "AggExpr": _r_agg,
    "MathExpr": _r_math,
    "InList": _r_inlist,
    "Coalesce": _r_variadic("coalesce"),
    "Greatest": _r_variadic("greatest"),
    "Least": _r_variadic("least"),
    "Array": _r_variadic("array"),
    "Case": _r_case,
}
