"""SQL list/array functions — the Spark-shaped half, including the lambda forms.

`anonymous._LIST_PAIR` covers the DuckDB `list_*`/`array_*` names that map onto a binary
`.list` method. This module covers the rest: the typed nodes sqlglot promotes
(`ArrayAppend`, `ArrayCompact`, `Transform`, `ArrayFilter`, ...) and, with them, the
**higher-order** forms — `transform(xs, x -> x + 1)`, `filter`, `exists`, `forall` — whose
argument is a lambda rather than a value.

A lambda is translated by rewriting its body against `element()`, the engine's placeholder
for "the current element" inside `.list.transform` / `.list.filter`. That is the whole
mechanism: bind the lambda's single parameter name to `element()` and translate the body
with the ordinary scalar path, so every expression the engine already has is available
inside a lambda without a second evaluator.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr, array, lit, nullif, when
from batcher.plan.functions.collection import element, sequence

__all__ = ["collection_function"]

# `f(vector, other)` → a binary `.list` method. Spark's `vector_*` family is the same
# arithmetic as the engine's embedding methods under different names.
_LAMBDA_LIST = {
    "list_transform": "transform",
    "array_transform": "transform",
    "list_apply": "transform",
    "list_filter": "filter",
    "array_filter": "filter",
}

_VECTOR_PAIR = {
    "vector_cosine_similarity": "cosine_similarity",
    "vector_inner_product": "dot",
    "vector_l2_distance": "l2_distance",
}


def collection_function(tr, node) -> Expr | None:
    """Translate a list/array call, or None when the node is not one of them."""
    if isinstance(node, exp.GenerateSeries):
        return _generate_series(tr, node)
    if isinstance(node, exp.ArrayAppend):
        return tr._scalar(node.this).list.concat(array(tr._scalar(node.expression)))
    if isinstance(node, exp.ArrayPrepend):
        # sqlglot keeps Spark's argument order (`array_prepend(xs, elem)`), so the
        # element is `expression` here even though it lands on the left of the concat.
        return array(tr._scalar(node.expression)).list.concat(tr._scalar(node.this))
    if isinstance(node, exp.ArrayInsert):
        return _array_insert(tr, node)
    if isinstance(node, exp.ArrayCompact):
        return tr._scalar(node.this).list.filter(element().is_not_null())
    if isinstance(node, exp.ArrayExcept):
        return tr._scalar(node.this).list.difference(tr._scalar(node.expression))
    if isinstance(node, exp.ArrayRemove):
        # The null guard is load-bearing: `element() != v` is *null* for a null element,
        # which the filter drops, so `array_remove(array(1, null), 1)` lost the null that
        # Spark keeps.
        value = tr._scalar(node.expression)
        return tr._scalar(node.this).list.filter(element().is_null() | (element() != value))
    if isinstance(node, exp.Transform):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.transform(body)
    if isinstance(node, exp.ArrayFilter):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.filter(body)
    if isinstance(node, exp.Exists) and isinstance(node.expression, exp.Lambda):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.filter(body).list.len() > lit(0)

    if not isinstance(node, exp.Anonymous):
        return None
    name = node.name.lower()
    args = list(node.expressions)

    if name in _LAMBDA_LIST and len(args) == 2:
        # DuckDB's spellings of the same two higher-order functions.
        body = _lambda_body(tr, args[1])
        if body is None:
            return None
        return getattr(tr._scalar(args[0]).list, _LAMBDA_LIST[name])(body)
    if name == "forall" and len(args) == 2:
        body = _lambda_body(tr, args[1])
        if body is None:
            return None
        # Every element satisfies the predicate exactly when none fails it.
        return tr._scalar(args[0]).list.filter(~body).list.len() == lit(0)
    if name == "get" and len(args) == 2:
        # Spark's `get` is 0-based and yields null out of range, which is `.list.get`.
        from batcher._sql.parser.expressions.literals import _const_int_arg

        return tr._scalar(args[0]).list.get(_const_int_arg(args[1], "get(): index"))
    if name == "array_repeat" and len(args) == 2:
        # `array_repeat(v, n)` — a list of `n` copies. The count has to be constant: a
        # per-row length would need a kernel, and a literal is the shape Spark's own
        # examples and nearly every real call use.
        from batcher._sql.parser.expressions.literals import _int_literal

        count = _int_literal(args[1])
        if count is None:
            return None
        value = tr._scalar(args[0])
        return array(*([value] * count)) if count > 0 else None
    if name == "arrays_overlap" and len(args) == 2:
        return _arrays_overlap(tr._scalar(args[0]), tr._scalar(args[1]))
    if name in _VECTOR_PAIR and len(args) == 2:
        method = _VECTOR_PAIR[name]
        return getattr(tr._scalar(args[0]).list, method)(tr._scalar(args[1]))
    if name == "vector_norm" and len(args) == 2:
        return _vector_norm(tr, args)
    return None


def _lambda_body(tr, lam) -> Expr | None:
    """Translate a one-parameter lambda body with its parameter bound to `element()`.

    Returns None for a lambda the engine cannot express — more than one parameter (the
    `(acc, x)` folds and `zip_with`'s `(x, y)`), which needs a second placeholder the
    `.list` kernels do not have.
    """
    if not isinstance(lam, exp.Lambda):
        return None
    params = [p.name for p in lam.expressions]
    if len(params) != 1:
        return None
    body = lam.this.copy()
    param = params[0]
    placeholder = exp.Anonymous(this="element", expressions=[])
    if isinstance(body, (exp.Column, exp.Identifier)) and body.name == param:
        return element()
    # A lambda parameter reaches the body as a bare `Identifier` (`x -> x + 1`) or, when
    # it is qualified, as a `Column`. Both spellings have to be rewritten, or the body
    # translates as a reference to a column that does not exist.
    for ref in list(body.find_all(exp.Column, exp.Identifier)):
        if ref.name == param and ref.parent is not None:
            ref.replace(placeholder.copy())
    return tr._scalar(body)


def _arrays_overlap(left: Expr, right: Expr) -> Expr:
    """Spark `arrays_overlap`: true on a shared element, null when only nulls could hide one.

    The three-valued rule is the whole difficulty. True wins outright; otherwise, if
    either side contains a null, the answer is unknown rather than false, because the
    null *might* have been the shared element. Composed from the set intersection and a
    null count so each branch is exact rather than approximated by `has_any`, whose null
    rule is a different one (it is null when a whole *list* is null).
    """
    shared = left.list.intersect(right).list.len() > lit(0)
    has_null = (left.list.filter(element().is_null()).list.len() > lit(0)) | (
        right.list.filter(element().is_null()).list.len() > lit(0)
    )
    unknown = nullif(lit(True), lit(True))
    return when(shared).then(lit(True)).when(has_null).then(unknown).otherwise(lit(False))


def _vector_norm(tr, args) -> Expr | None:
    """Spark `vector_norm(v, p)` for the two norms the engine implements (p = 1 or 2)."""
    order = args[1]
    if not isinstance(order, exp.Literal) or order.is_string:
        return None
    p = float(order.this)
    if p == 1.0:
        return tr._scalar(args[0]).list.l1_norm()
    if p == 2.0:
        return tr._scalar(args[0]).list.l2_norm()
    return None


def _generate_series(tr, node) -> Expr | None:
    """`range(...)` / `generate_series(...)` / Spark `sequence(...)` → the engine's series.

    One node serves three functions whose *bounds* differ, and the difference is not
    cosmetic: `generate_series(1, 5)` and `sequence(1, 5)` are `[1,2,3,4,5]` where
    `range(1, 5)` is `[1,2,3,4]`, and `range(3)` is `[0,1,2]`. The engine's `sequence` is
    the inclusive form, so the exclusive one is expressed by pulling the stop in by a
    step — which is also the only reading that stays correct for a step other than 1.
    """
    start, stop = node.args.get("start"), node.args.get("end")
    if start is None:
        return None
    step = node.args.get("step")
    step_expr = tr._scalar(step) if step is not None else lit(1)
    if stop is None:
        # The one-argument `range(n)`: 0 .. n-1.
        return sequence(lit(0), tr._scalar(start) - lit(1), step_expr)
    exclusive = bool(node.args.get("is_end_exclusive"))
    stop_expr = tr._scalar(stop)
    if exclusive:
        stop_expr = stop_expr - step_expr
    return sequence(tr._scalar(start), stop_expr, step_expr)


def _array_insert(tr, node) -> Expr | None:
    """Spark `array_insert(xs, pos, value)` — `value` spliced in at 1-based `pos`.

    Only a *constant* position is served, because the splice is expressed as two slices
    around it and a per-row position would need a kernel. Spark's rule that a position
    past the end pads with nulls is not expressible that way either, so a position beyond
    the list's length is declined rather than answered with a shorter list.
    """
    from batcher._sql.parser.expressions.literals import _const_int_arg

    position = _const_int_arg(node.args.get("position"), "array_insert(): position")
    if position < 1:
        return None  # Spark counts a negative position from the end; not expressible here
    xs = tr._scalar(node.this)
    value = array(tr._scalar(node.args["expression"]))
    head = xs.list.slice(0, position - 1)
    tail = xs.list.slice(position - 1, 2**31 - 1)
    return head.list.concat(value).list.concat(tail)
