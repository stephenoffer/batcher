"""A tiny benchmarking + correctness-checking framework.

Each query is expressed once per engine as a zero-argument callable that returns a
``pyarrow.Table``. ``compare`` runs all engines, verifies they produce the same result, and
records best-of-N wall-clock timings. ``print_table`` renders an aligned summary.

Three responsibilities, one per module:

``compare``
    Canonicalize each engine's result, reconcile column types across the lineup, compare as
    row multisets tolerant of float rounding, and time what agreed.
``order``
    The half of correctness a multiset comparison cannot see. Both sides get sorted before
    they are compared, so an engine that skipped an ``ORDER BY`` entirely still matches; a
    case that asked for an order is additionally checked for monotonicity in its own.
``report``
    The aligned table, the one-line result format an isolated child hands back, and the
    per-case subprocess isolation that keeps one engine's ``SIGKILL`` from taking the run.
"""

from __future__ import annotations

from .compare import (
    FLOAT_ATOL,
    FLOAT_RTOL,
    GRID_ATOL,
    ROUND_DECIMALS,
    CompareResult,
    EngineResult,
    RowSet,
    bench,
    column_classes,
    compare,
    results_match,
    rowsets_match,
    to_rowset,
)
from .divergences import KNOWN_DIVERGENCES, Divergence, explain
from .names import canonical_column_name, canonical_names
from .order import OrderKey, order_keys_of, order_violation
from .report import RESULT_PREFIX, emit_result, print_table, run_isolated
from .summary import Summary, format_repeats, format_summary, geomean, summarize

__all__ = [
    "FLOAT_ATOL",
    "FLOAT_RTOL",
    "GRID_ATOL",
    "KNOWN_DIVERGENCES",
    "RESULT_PREFIX",
    "ROUND_DECIMALS",
    "CompareResult",
    "Divergence",
    "EngineResult",
    "OrderKey",
    "RowSet",
    "Summary",
    "bench",
    "canonical_column_name",
    "canonical_names",
    "column_classes",
    "compare",
    "emit_result",
    "explain",
    "format_repeats",
    "format_summary",
    "geomean",
    "order_keys_of",
    "order_violation",
    "print_table",
    "results_match",
    "rowsets_match",
    "run_isolated",
    "summarize",
    "to_rowset",
]
