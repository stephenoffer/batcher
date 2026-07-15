"""Approximate shortcuts — answers a sketch can give, named so they can never pretend otherwise.

Everything in this module reads a *measured* statistic: a KLL quantile grid, a Misra-Gries
top-values map, an HLL distinct count, a measured average width. None of them is exact, and
none of them may ever back an exact terminal — which is why every function here says `approx`
in its name, and why the exact families refuse the very facets this one is built on.

The trade they offer is the one that matters at scale: a median over a billion rows is a
sort, and a sort is a pipeline breaker; a median from a KLL grid is a dictionary lookup with
a bounded error. When the caller can spend the error, this is free. When it cannot, it asks
the exact family and pays for a run.

`None` here means "nothing has been measured yet" — not "not provable". These sketches are
recorded by Core on a previous execution, so a query that has run before answers from them
and a query that has not, does not. That is the learned-metadata moat: it gets cheaper the
more it runs.
"""

from __future__ import annotations

from typing import Any

from batcher.kyber.metadata_answer import _value_at_quantile
from batcher.kyber.shortcuts.facts import Facts

__all__ = [
    "approx_column_bytes",
    "approx_frequency",
    "approx_histogram",
    "approx_median",
    "approx_memory_bytes",
    "approx_percentile",
    "approx_quantile",
    "approx_row_bytes",
    "approx_top_k",
]

#: Assumed average width of a variable-width value (string, binary, list) when nothing has
#: measured it. Deliberately a single documented constant rather than a per-call guess — a
#: memory estimate that is wrong is at least wrong *consistently*, and once the column has
#: been read once, `avg_bytes` replaces it with the truth.
_ASSUMED_VARIABLE_WIDTH_BYTES = 16.0


def approx_quantile(facts: Facts, column: str, q: float) -> float | None:
    """The approximate value at quantile `q` (0..1) from a measured KLL grid, or None.

    Interpolated between the recorded quantile boundaries — the inverse of the range
    selectivity the cost model reads from the same grid. Returns None when no grid has been
    measured for this column, in which case an exact quantile (a sort, or a streaming sketch)
    is the only way to get one.
    """
    grid = facts.col(column).quantiles
    if not grid:
        return None
    return _value_at_quantile(float(q), list(grid.get("probs", [])), list(grid.get("values", [])))


def approx_median(facts: Facts, column: str) -> float | None:
    """The approximate median from a measured quantile grid, or None if none is recorded."""
    return approx_quantile(facts, column, 0.5)


def approx_percentile(facts: Facts, column: str, p: float) -> float | None:
    """The approximate value at percentile `p` (0..100), or None if no grid is recorded."""
    return approx_quantile(facts, column, float(p) / 100.0)


def approx_histogram(facts: Facts, column: str, bins: int = 10) -> list[tuple[float, float]] | None:
    """`bins` equal-probability bucket edges as `(low, high)` pairs, or None if unmeasured.

    Equal-*probability*, not equal-width: each bucket holds roughly the same number of rows,
    and its width tells you where the mass is. A long tail shows up as one very wide bucket
    rather than as nine empty ones, which is the shape a person actually wants to see.
    """
    if bins < 1:
        return None
    edges = [approx_quantile(facts, column, i / bins) for i in range(bins + 1)]
    if any(edge is None for edge in edges):
        return None
    return [(float(edges[i]), float(edges[i + 1])) for i in range(bins)]  # type: ignore[arg-type]


def approx_top_k(facts: Facts, column: str, k: int = 10) -> list[tuple[str, float]] | None:
    """The `k` most common values as `(value, share_of_rows)`, or None if unmeasured.

    From the Misra-Gries most-common-values map Core records. The share is a fraction of all
    rows in `[0, 1]`, and the values are their string forms (the map is keyed that way, so a
    value and its rendering cannot disagree). A skewed join key or a hot partition shows up
    here without a `GROUP BY`.
    """
    mcv = facts.col(column).mcv
    if not mcv:
        return None
    ranked = sorted(mcv.items(), key=lambda item: (-item[1], item[0]))
    return [(value, float(freq)) for value, freq in ranked[:k]]


def approx_frequency(facts: Facts, column: str, value: Any) -> float | None:
    """The approximate share of rows where `column` equals `value`, or None if unmeasured.

    Read from the most-common-values map, so it is only known for a value that *is* common —
    a rare value is absent from the map and returns None rather than a fabricated `1/ndv`.
    That refusal is deliberate: `1/ndv` is exactly the estimate this map exists to correct.
    """
    mcv = facts.col(column).mcv
    if not mcv:
        return None
    freq = mcv.get(str(value))
    return None if freq is None else float(freq)


def approx_column_bytes(facts: Facts, column: str) -> float | None:
    """The approximate in-memory size of `column`, in bytes, or None if its type is unknown.

    Measured average width when Core has recorded one, else the type's fixed width, else a
    documented assumption for variable-width data. An estimate, always — Arrow's actual
    footprint depends on buffer padding, dictionary encoding, and validity bitmaps.
    """
    width = _value_width(facts, column)
    return None if width is None else width * facts.estimated_rows


def approx_row_bytes(facts: Facts) -> float:
    """The approximate width of one row, in bytes — the sum over every column."""
    return sum(_value_width(facts, name) or 0.0 for name in facts.columns)


def approx_memory_bytes(facts: Facts) -> float:
    """The approximate size of the whole relation in memory, in bytes.

    Rows times row width, with both approximate: it is the number to size a buffer, a
    broadcast, or a spill threshold from — never the number to report as a fact.
    """
    return approx_row_bytes(facts) * facts.estimated_rows


def _value_width(facts: Facts, column: str) -> float | None:
    """One value's assumed width in bytes: measured, else fixed by type, else assumed."""
    col = facts.col(column)
    if col.avg_bytes is not None:
        return float(col.avg_bytes)
    if col.dtype is None:
        return None
    try:
        return col.dtype.bit_width / 8.0
    except (AttributeError, ValueError):
        # A variable-width type (string, binary, list, struct) has no bit width to ask for.
        return _ASSUMED_VARIABLE_WIDTH_BYTES
