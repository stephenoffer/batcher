"""SQL → `.map` accessor dispatch.

The map family carries enough of its own dispatch to live in a module of its own, in the
same shape as `collections`, `strings` and `temporal`: one entry point returning None for
anything it does not serve, so the caller's "unknown function" error still names the
function.

It exists as a separate module rather than more branches in `functions.py` because a map
subscript and a list subscript are the *same* sqlglot node. Keeping the map reading of it
here is what stops `_list_function` growing a second, unrelated concern — and that concern
is where the bug was: every `exp.Bracket` was assumed to be a list index, so `m['a']` and
Spark's `element_at(m, 'a')` (which parses as the same node) died on `int('a')`.

A struct is served here too, not in a module of its own: `s['a']` and `struct_extract(s,'a')`
are the same lookup as a map's, and the engine resolves which container it is from the
array's own type. Splitting them would put one operation in two files.

The dot form `s.a` is **not** handled here and still fails. sqlglot parses it as a
`Column` qualified by table `s` rather than as a struct access, so it is rejected during
*column resolution*, before any expression dispatch runs — a different layer, and a
separate change.

`map_extract` is deliberately not served here. DuckDB returns a *list* for it — `[1]` for a
hit and `[]` for a miss — where the subscript returns the bare value, so mapping it to
`.map.get` would answer a plausible result that is not DuckDB's. It needs a kernel.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr

__all__ = ["map_function", "map_subscript"]


def map_function(tr, node) -> Expr | None:
    """Build the `.map` expression for `node`, or None if it is not a map function.

    Args:
        tr: The translator, used to lower child expressions.
        node: The sqlglot node to dispatch.

    Returns:
        The Batcher expression, or None when `node` is not served here.
    """
    if isinstance(node, exp.StructExtract):
        # `struct_extract(s, 'a')` is the subscript under another name, so it lands on
        # the same kernel as `s['a']`.
        name = node.expression
        if isinstance(name, (exp.Literal, exp.Identifier)):
            return tr._scalar(node.this).struct.get(name.name)
        return None
    if isinstance(node, exp.MapKeys):
        # `map_values` arrives as `exp.Anonymous` and the anonymous table serves it;
        # `map_keys` gets a typed node, so without this branch it raises "unsupported SQL
        # expression" beside its own working kernel.
        return tr._scalar(node.this).map.keys()
    return None


def map_subscript(tr, node) -> Expr | None:
    """Build `m[key]` when `node`'s subscript names a map key, else None.

    A string key is a map lookup; an integer is a list index. The key's type is the only
    signal available, because a translator has no schema — so an *integer-keyed* map still
    reads as a list index, which is the one ambiguous case. A string is unambiguous: no
    list is indexed by one.

    Args:
        tr: The translator, used to lower child expressions.
        node: The `exp.Bracket` node to inspect.

    Returns:
        The `.map.get` expression, or None to leave `node` on the list path.
    """
    keys = node.expressions
    if len(keys) != 1:
        return None
    key = keys[0]
    if isinstance(key, exp.Literal) and key.is_string:
        return tr._scalar(node.this).map.get(key.name)
    return None
