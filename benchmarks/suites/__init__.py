"""Benchmark suites, in two families that share one harness.

- ``standard`` — the SQL-first industry benchmarks (TPC-H, TPC-DS, ClickBench).
- ``operators`` — the operator-mix (single relational ops over real TPC-H tables),
  where the non-SQL engines (PyArrow, Ray Data) also compete.

Importing this package imports both, which runs their registration decorators and
populates ``registry.REGISTRY``.
"""

from __future__ import annotations

from . import operators, standard  # noqa: F401  (imported for registration side effects)
