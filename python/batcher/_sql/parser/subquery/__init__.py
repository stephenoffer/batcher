"""Subquery handling and decorrelation for the SQL translator.

Three decorrelations, one module each, all turning a subquery predicate into a join the
optimizer and executor already understand:

- `core` — the shared parsers plus `IN`/`EXISTS` over an *equality* correlation, which
  become semi/anti joins on the correlation keys.
- `neq` — a correlated ``<>`` residual, which is not an equi-join: it decorrelates to a
  per-key `min`/`max` bound test (the TPC-H q21 shape).
- `range` — an *inequality* correlation, which decorrelates to a range semi/anti join.
- `quantified` — a pre-pass turning `= ANY` / `<> ALL` into the `IN` / `NOT IN` they are
  defined as, so those inherit `core`'s decorrelation rather than needing their own.

The public import path `batcher._sql.parser.subquery` is unchanged and so is what it means;
the split is the sanctioned response to the parser directory reaching its file-count ceiling,
grouped by the responsibility the three already shared.
"""

from __future__ import annotations

# Every module-level name `core` defines, so the package is a drop-in for the module it
# replaced — the translator reaches several of these as attributes of `subquery` itself.
from batcher._sql.parser.subquery.core import (
    _apply_exists,
    _apply_in_subquery,
    _apply_single_predicate,
    _apply_subquery_predicates,
    _decorrelate_scalar_subqueries,
    _in_subquery_select,
    _is_in_subquery,
    _not_in_antijoin,
)
from batcher._sql.parser.subquery.correlation import (
    _correlation_pair,
    _is_plain_column,
    _local_columns,
    _local_tables,
    _outer_key_reducer,
    _reject_correlated,
)
from batcher._sql.parser.subquery.quantified import normalize_quantified

__all__ = [
    "_apply_exists",
    "_apply_in_subquery",
    "_apply_single_predicate",
    "_apply_subquery_predicates",
    "_correlation_pair",
    "_decorrelate_scalar_subqueries",
    "_in_subquery_select",
    "_is_in_subquery",
    "_is_plain_column",
    "_local_columns",
    "_local_tables",
    "_not_in_antijoin",
    "_outer_key_reducer",
    "_reject_correlated",
    "normalize_quantified",
]
