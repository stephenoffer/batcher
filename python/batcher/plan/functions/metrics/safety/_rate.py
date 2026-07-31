"""The corpus-rate shape and the pattern-match helper every safety monitor here shares.

A safety monitor is always the same two steps: decide per row whether a pattern is present,
then divide by the corpus size. Writing that out per module is how `_rate` came to exist three
times over in the text metrics; it lives here once for this package.
"""

from __future__ import annotations

from collections.abc import Iterable

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = ["matches_any", "rate"]


def rate(condition: Expr) -> Expr:
    """The fraction of rows where `condition` holds, as a corpus rate in ``[0, 1]``."""
    return count_if(condition) / count_if(lit(True))


def matches_any(text: IntoExpr, patterns: Iterable[str]) -> Expr:
    """True where the text matches any of the case-insensitive regexes.

    The patterns are joined into one alternation so the engine walks the string once rather
    than once per pattern, which matters when a monitor carries twenty markers.
    """
    joined = "|".join(f"(?:{p})" for p in patterns)
    return _as_column(text).str.regexp_matches(f"(?i)(?:{joined})")
