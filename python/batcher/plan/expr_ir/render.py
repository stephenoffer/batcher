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
    "floordiv": "//",
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

# Accessor-function node class → the namespace it is reached through, so a
# `StrFunc('upper', col('x'))` renders as ``col('x').str.upper()``.
_NS_PREFIX = {
    "StrFunc": "str",
    "DateFunc": "dt",
    "DateTrunc": "dt",
    "Strftime": "dt",
    "Strptime": "dt",
    "ConvertTimezone": "dt",
    "DateOffset": "dt",
    "ListFunc": "list",
    "ListGet": "list",
    "ListContains": "list",
    "ListPosition": "list",
    "ListSlice": "list",
    "ListSimhash": "list",
    "ListBinary": "list",
    "ListSet": "list",
    "ListTransform": "list",
    "ListFilter": "list",
    "StructField": "struct",
    "MapFunc": "map",
}

_MAX_DEPTH = 6


def render_expr(expr: Any, depth: int = 0) -> str:
    """Render an `Expr` node as a source-like string (used by ``Expr.__repr__``)."""
    cls = type(expr).__name__
    if depth > _MAX_DEPTH:
        return "…"
    r = _RENDERERS.get(cls)
    if r is not None:
        return r(expr, depth)
    if cls in _NS_PREFIX:
        return _render_accessor(expr, depth)
    return _render_generic(expr, depth)


def _kids(expr: Any) -> list[str]:
    """Rendered sub-expressions of a node, discovered by reflection."""
    from batcher.plan.expr_ir.core import Expr

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
    ns = _NS_PREFIX[type(expr).__name__]
    base = render_expr(expr.input, depth + 1)
    fn = getattr(expr, "fn", None) or getattr(expr, "field", None) or "?"
    return f"{base}.{ns}.{fn}()"


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
