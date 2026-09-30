"""The 22 TPC-H queries exactly as Ray Data's release benchmark states them, on `bt.Dataset`.

Ray's `release/nightly_tests/dataset/tpch/tpch_q*.py` are not standard TPC-H: q3, q10 and
q18 drop the `LIMIT` and materialize the whole sorted result, q18's threshold is 312, q11's
fraction is `0.0001 / SF`, q14 and q17 return the raw sums, q13 matches `special.*requests`
as a regex, and q16 counts distinct suppliers through two group-bys. A ratio against Ray's
published numbers is only meaningful for the queries Ray ran, so these mirror the Spark
port of those scripts statement for statement (column selections, casts to float64, join
types, sort keys) rather than the SQL in `suites/standard/tpch.py`.

What is deliberately *not* mirrored is how Ray gets there. Where Ray pulls a scalar to the
driver mid-query (q11's total, q15's max, q22's average) and pays a separate execution for
it, these express the same value as a one-row aggregate joined back in, which is the one
query plan Batcher optimizes as a whole. The answer is identical; the scalar is the same
reduction over the same rows.

Each builder takes ``{table -> bt.Dataset}`` and the scale factor, and returns a lazy
`bt.Dataset`; the caller times `collect`.
"""

from __future__ import annotations

from . import q01_11, q12_22  # noqa: F401  (importing registers the queries)
from .tables import LARGE_RESULT, QUERIES, TABLE_COLUMNS, load_tables

__all__ = ["LARGE_RESULT", "QUERIES", "TABLE_COLUMNS", "load_tables"]
