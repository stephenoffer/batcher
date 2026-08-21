"""Predicate translation for source-side pushdown.

Kyber records the `Filter` sitting directly above a `Scan` as that source's
*pushed predicate* (`PhysicalPlan.source_predicates`). A pushdown-capable source
translates the **pushable subset** of that predicate IR into its backend filter
(a pyarrow `Expression`, a SQL `WHERE`, …) to skip I/O at the reader. The engine
keeps the `Filter` operator regardless, so a partial or absent translation is
always safe — it just reads more rows. This module owns the IR→backend mapping.

Pushable subset: comparisons (`= != < <= > >=`) between a column and a literal,
`IS NULL` / `IS NOT NULL`, `IN` lists, `NOT`, the `starts_with`/`ends_with`/`contains`
string predicates, a constant boolean, and `AND`/`OR` of pushable terms. Anything else
makes the term unpushable for that backend.

**An `AND` keeps whichever side translated; an `OR` is all-or-nothing.** Dropping a
conjunct only ever *widens* the rows read, and the engine's `Filter` re-checks every one
of them, so a partial `AND` costs pruning and never a row. Dropping a disjunct narrows the
filter and would lose rows, so an `OR` with an untranslatable side declines entirely.

That asymmetry is why the read-path translators here return a partial filter rather than
`None`: a six-predicate warehouse query with one unpushable term used to extract the whole
table over the network because a single conjunct could not be spelled. The one exception is
a predicate used to *choose rows to replace* rather than to skip I/O (Iceberg's
``replace_where``), where widening would delete rows the caller did not name — that keeps
the strict form, which is why `to_iceberg_expression` asks before it prunes.

**`NOT` is the second exception, and it is the reason every translator here threads an
`exact` flag.** Widening is only safe under an even number of negations: `NOT` of a
*widened* filter is a *narrowed* one, and a narrowed pushdown drops rows the query asked
for. Translating ``~(a == 1 & unpushable(b))`` by dropping the unpushable conjunct yields
``NOT (a = 1)``, which discards every row where ``a = 1`` regardless of ``b`` — a silently
wrong answer from a filter that only meant to prune I/O. So a `NOT` translates its operand
in exact mode, where a partial `AND` and every merely-widening term decline instead. The
same flag is what `to_iceberg_expression`'s `allow_partial=False` has always meant, so the
two are now one mechanism rather than two.
"""

from __future__ import annotations

from batcher.io.predicate._shapes import _combine as combine_conjunction
from batcher.io.predicate._shapes import _conjuncts as conjuncts
from batcher.io.predicate._shapes import _pinned_columns as pinned_columns
from batcher.io.predicate.arrow import to_pyarrow_expression
from batcher.io.predicate.iceberg import to_iceberg_expression
from batcher.io.predicate.mongo import to_mongo_filter
from batcher.io.predicate.native import to_native_predicate
from batcher.io.predicate.sql import to_sql_where

__all__ = [
    "combine_conjunction",
    "conjuncts",
    "pinned_columns",
    "to_iceberg_expression",
    "to_mongo_filter",
    "to_native_predicate",
    "to_pyarrow_expression",
    "to_sql_where",
]
