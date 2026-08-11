"""Benchmark suites, in four families that share one harness.

- ``standard`` — the SQL-first industry benchmarks (TPC-H, TPC-DS, ClickBench).
- ``h2o`` — the H2O.ai db-benchmark's groupby and join tasks, the standard dataframe-engine
  workload (one wide table, keys of very different cardinality, three RHS join sizes).
- ``operators`` — the operator-mix (single relational ops over real TPC-H tables),
  where the non-SQL engines (PyArrow, Ray Data) also compete.
- ``scan`` — one table in three parquet file layouts, isolating scan-planning cost.
- ``multimodal`` — unstructured-data ingest (images: read/decode/resize), across the
  engines with a multimodal path (Batcher, Ray Data, Daft, PyArrow).
- ``semistructured`` — JSON-document parsing + typed path extraction (Batcher, DuckDB,
  Polars, Daft, Spark).

Importing this package imports all six, which runs their registration decorators and
populates ``registry.REGISTRY``.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for registration side effects)
    h2o,
    multimodal,
    operators,
    scan,
    semistructured,
    standard,
)
