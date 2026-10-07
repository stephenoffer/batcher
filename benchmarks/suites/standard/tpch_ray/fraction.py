"""TPC-H's one scale-dependent substitution parameter: q11's ``FRACTION = 0.0001 / SF``.

Every other substitution parameter is a constant, so the query text is written once. q11's is
not: the spec divides the threshold by the scale factor because the total it is a fraction of
grows with SF while each part's value does not. Left at ``0.0001`` past sf1, no part clears the
bar and every engine agrees on zero rows -- a DEGENERATE case that times no work at all.

The case builder knows the scale (``Context.scale``) and records it here before any engine
runs, so the SQL text and the native Polars / Ray Data pipelines read the same number.
"""

from __future__ import annotations

__all__ = ["Q11_SQL_FRACTION", "q11_fraction", "scaled_sql", "set_scale"]

# The literal q11's SQL carries at sf1; `scaled_sql` replaces it with the scaled value.
Q11_SQL_FRACTION = "0.0001000000"

_scale = 1.0


def set_scale(scale: float) -> None:
    """Record the scale factor the current run's cases execute at."""
    global _scale
    _scale = float(scale)


def q11_fraction() -> float:
    """q11's FRACTION at the recorded scale factor."""
    return 0.0001 / _scale


def scaled_sql(name: str, query: str) -> str:
    """``query`` with q11's FRACTION scaled to the recorded scale factor; others unchanged."""
    if name != "tpch-q11":
        return query
    if Q11_SQL_FRACTION not in query:
        raise ValueError("tpch-q11 no longer carries the FRACTION literal it is scaled by")
    return query.replace(Q11_SQL_FRACTION, f"{q11_fraction():.12f}")
