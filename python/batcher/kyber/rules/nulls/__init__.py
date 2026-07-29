"""Null-reasoning rule families — three-valued logic and null-strictness.

SQL's three-valued logic is where "obvious" algebra goes wrong most often, so the
rules that reason about nulls are grouped here rather than scattered through the
arithmetic and string families. Two concerns live in this package:

* `strictness` pushes an `IS NULL` / `IS NOT NULL` test *through* the scalar
  functions that are null-strict and total, so the test lands on a bare column,
  where predicate pushdown, zonemap pruning, and join-key null rejection can all
  see it.
* `three_valued` collapses the tautologies and contradictions the null predicates
  themselves generate, which are exactly the shapes the pushed tests create.

Importing this package runs the `@rule` decorators and the registry factories,
registering every rule into `kyber.registry.DEFAULT_REGISTRY`.
"""

from __future__ import annotations

from batcher.kyber.rules.nulls import strictness as _strictness  # noqa: F401  (registers rules)
from batcher.kyber.rules.nulls import three_valued as _three_valued  # noqa: F401  (registers)

__all__: list[str] = []
