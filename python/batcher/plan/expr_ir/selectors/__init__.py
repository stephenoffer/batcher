"""Column selectors — expressions that stand for *many* columns at plan time.

A `Selector` is an `Expr` leaf that matches a set of the input's columns, by name,
by name pattern, or by Arrow dtype. Because it is an `Expr`, the whole scalar algebra
composes onto it for free (``numeric() * 2``, ``string().str.upper()``), and the
projection layer expands it against the input schema when it builds a projection.

This package is a façade: `core` holds the `Selector` type and its `.name` accessor,
`build` holds the public constructors, and `expand` holds the schema-resolution used
by `select` / `with_columns` / `drop` / `group_by().agg()` and the name-taking verbs.
"""

from __future__ import annotations

from batcher.plan.expr_ir.selectors.build import (
    all,
    boolean,
    by_dtype,
    contains,
    ends_with,
    exclude,
    floating,
    integer,
    matches,
    numeric,
    starts_with,
    string,
    temporal,
)
from batcher.plan.expr_ir.selectors.core import Selector
from batcher.plan.expr_ir.selectors.expand import (
    expand_one,
    expand_selectors,
    has_selector,
    resolve_names,
    single_selector,
    substitute,
)

__all__ = [
    "Selector",
    "all",
    "boolean",
    "by_dtype",
    "contains",
    "ends_with",
    "exclude",
    "expand_one",
    "expand_selectors",
    "floating",
    "has_selector",
    "integer",
    "matches",
    "numeric",
    "resolve_names",
    "single_selector",
    "starts_with",
    "string",
    "substitute",
    "temporal",
]
