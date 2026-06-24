"""String-building free functions (`concat`, `concat_ws`, `format_string`).

All three lower to existing IR — the `concat` binary op (SQL ``||``), `array` +
`list.join`, and casts — so they add public surface without touching the engine.
Null handling matches DuckDB: `concat`/`concat_ws` treat NULL as absent (the
differential oracle), not null-propagating like the raw ``||`` operator.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Binary, Coalesce, Expr, IntoExpr, Lit, _wrap
from batcher.plan.expr_ir.nodes import Array, ListJoin


def concat(*exprs: IntoExpr) -> Expr:
    """Concatenate values into one string (DuckDB/Spark ``concat``).

    Each argument is cast to text; NULLs are treated as the empty string (DuckDB
    semantics), so ``concat("a", lit(None), "b")`` is ``"ab"`` — unlike the raw
    ``a || b`` operator, which propagates NULL. Requires at least one argument.
    """
    if not exprs:
        raise PlanError("concat() requires at least one argument")
    # NULL → '' so a null contributes nothing (DuckDB concat, not `||`).
    parts = [Coalesce([_wrap(e).cast("string"), Lit("")]) for e in exprs]
    result = parts[0]
    for part in parts[1:]:
        result = Binary("concat", result, part)
    return result


def concat_ws(separator: str, *exprs: IntoExpr) -> Expr:
    """Concatenate values with `separator` between them (DuckDB/Spark ``concat_ws``).

    NULL arguments are skipped entirely — no doubled separator — matching DuckDB:
    ``concat_ws(",", "a", lit(None), "b")`` is ``"a,b"``. Each argument is cast to
    text. Requires at least one value argument.
    """
    if not exprs:
        raise PlanError("concat_ws() requires at least one value argument")
    # array(...).list.join skips nulls, which is exactly concat_ws's contract.
    elements = [_wrap(e).cast("string") for e in exprs]
    return ListJoin(Array(elements), separator)


def format_string(format: str, *exprs: IntoExpr) -> Expr:
    """Interpolate values into a template with ``{}`` placeholders (Polars ``format``).

    ``format_string("{} = {}", col("k"), col("v"))`` yields ``"k = v"`` per row. The
    number of ``{}`` placeholders must equal the number of arguments. Values are cast
    to text with the same NULL-as-empty rule as :func:`concat`. The placeholder is the
    literal two-character ``{}`` (no printf width/precision — keep formatting in SQL).
    """
    chunks = format.split("{}")
    if len(chunks) - 1 != len(exprs):
        raise PlanError(
            f"format_string: {len(exprs)} argument(s) but {len(chunks) - 1} '{{}}' placeholder(s)"
        )
    parts: list[IntoExpr] = []
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append(Lit(chunk))
        if i < len(exprs):
            parts.append(exprs[i])
    if not parts:
        return Lit("")
    return concat(*parts)
