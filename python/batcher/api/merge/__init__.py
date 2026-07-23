"""``MERGE INTO`` — the full SQL merge, composed from relational algebra.

The pieces, and where the interesting decisions live:

- `clauses` — the three populations (matched / not matched / not matched by source), the
  actions legal for each, and `source_col`/`target_col` for naming the two sides.
- `compose` — the algebra: ``MERGE`` is joins + chained ``CASE`` + ``union``, no new IR,
  which is why it runs on a cluster without a second implementation.
- `plan` — the *decision*: which data files could hold one of the source's keys, and so
  which must be rewritten. Pure, side-effect free, and where the speedup is won.
- `execute` — carrying it out: rewrite the surviving files, swap them in beside the ones
  that were skipped, delete the ones they replaced.
- `format` — what format the table at a path is, inferred from its files when the path
  itself (a directory) does not say.
- `native` — the transactional sinks (Delta/Iceberg) that have a MERGE of their own.
- `builder` — the fluent surface (`ds.write.merge_into`), plus the keyword shorthand.
- `cdc` — applying a *change feed* (deletes, redeliveries, out-of-order rows) rather than
  a clean snapshot.
"""

from __future__ import annotations

from batcher.api.merge.builder import MergeBuilder, MergeWhen, simple_clauses
from batcher.api.merge.cdc import (
    SEQUENCE_COMPARE_COL,
    cdc_stored_columns,
    compose_cdc_apply,
)
from batcher.api.merge.clauses import (
    MergeClause,
    source_col,
    target_col,
)
from batcher.api.merge.compose import compose_merge
from batcher.api.merge.execute import execute_merge, run_merge
from batcher.api.merge.format import target_format
from batcher.api.merge.native import merge_predicate_for
from batcher.api.merge.plan import MergePlan, plan_merge

__all__ = [
    "SEQUENCE_COMPARE_COL",
    "MergeBuilder",
    "MergeClause",
    "MergePlan",
    "MergeWhen",
    "cdc_stored_columns",
    "compose_cdc_apply",
    "compose_merge",
    "execute_merge",
    "merge_predicate_for",
    "plan_merge",
    "run_merge",
    "simple_clauses",
    "source_col",
    "target_col",
    "target_format",
]
