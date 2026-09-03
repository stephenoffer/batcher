"""The dispatches *derived* from the public expression surface, in the order they run.

Two of the SQL translator's handlers are not tables of names. `families` reaches the whole
public function library (`plan.functions`) and `accessors` reaches the typed `Expr`
namespaces, both by deriving the SQL name from the Python one -- which is what stops either
vocabulary from falling behind the surface it serves.

They share one rule, and it is the reason they are sequenced here rather than at the call
site: **they run last**, after every typed node and every curated family table, so a name
SQL already means keeps its SQL meaning and neither can shadow one. Stating that ordering
once means `scalar.py` asks a single question ("does anything derived serve this?") instead
of restating the precedence at each of the two places it dispatches.
"""

from __future__ import annotations

from batcher._sql.parser.expressions.lowering.accessors import accessor_function
from batcher._sql.parser.expressions.lowering.families import family_function
from batcher.plan.expr_ir import Expr

__all__ = ["derived_function"]


def derived_function(tr, node) -> Expr | None:
    """Build `node` from the derived vocabularies, or None if neither serves its name.

    The function library goes first: it is the older of the two and the narrower, and a
    name in both is a free function that the accessor namespace also exposes as a method,
    where the free function is the spelling SQL has always had.

    Args:
        tr: The translator, used to lower the arguments.
        node: The sqlglot node to dispatch.

    Returns:
        The Batcher expression, or None when neither vocabulary names `node`.
    """
    # Explicit `is not None`, never `or`: `Expr.__bool__` raises `PlanError` by design, so
    # `family_function(...) or accessor_function(...)` would fail on every geospatial call
    # the first one served.
    library = family_function(tr, node)
    if library is not None:
        return library
    return accessor_function(tr, node)
