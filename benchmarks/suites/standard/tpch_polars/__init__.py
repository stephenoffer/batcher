"""Native Polars lazy-DataFrame pipelines for all 22 TPC-H queries.

Polars *has* a SQL frontend, so the standard suite used to fan the TPC-H text at it
like any other SQL engine. That comparison was effectively dead: ``pl.SQLContext``
rejects implicit ``FROM a, b WHERE ...`` joins, ``EXISTS``, and scalar-subquery
comparisons, which left 21 of 22 queries reporting ``n/a``, and on q6 its constant
folding of ``0.06 + 0.01`` produced a *wrong* answer (see ``queries_a.q6``).

Polars' own published TPC-H benchmark is written against the lazy DataFrame API, not
its SQL parser, so that is what this package supplies — the same 22 workloads, one
``LazyFrame`` pipeline each, giving Polars a real column on every query. The harness
still gates each timing on the result matching DuckDB, so a pipeline that drifted
from the SQL would be reported as ``FAILED``, never quietly timed.

The query bodies live in ``queries_a`` (q1-q11) and ``queries_b`` (q12-q22), the
registry they self-register into is ``base``, and ``runner`` adapts one to the shape
the suite calls. This module only re-exports the entry point.
"""

from __future__ import annotations

from .runner import polars_impl

__all__ = ["polars_impl"]
