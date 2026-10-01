"""Memoize the optimizer — the same query, planned once.

Optimization is a pure function of `(logical plan, bound sources, config, learned stats)`.
It is also, on a join-heavy query, the single most expensive thing Batcher does: TPC-H Q8
spends 63 ms in Kyber against 40 ms in the engine and 22 ms for DuckDB's entire query. A
BI dashboard, a scheduled report, and a benchmark harness all re-issue the identical
statement; re-deriving the identical plan each time is pure waste. Every serious engine
caches plans (Spark, Presto, Snowflake); Batcher did not.

**The key is exact, not structural.** `kyber.signature.plan_signature` deliberately
*normalizes literals* so learned statistics generalize across `x > 5` and `x > 6` — which
makes it lethal as a cache key. This module keys on the plan's lowered IR verbatim, so two
queries share an entry only when they would lower to the same bytes.

Three more things go into the key, each because it can change the plan Kyber chooses:

* **the bound sources**, by data-stable identity. A file source identifies by path; an
  in-memory source's `identity()` is only shape-based (schema + row count), so two
  different relations collide on it. That collision is not merely suboptimal — Kyber's
  zone-map pruning folds a filter to `FALSE` from a source's `min`/`max`, so a plan built
  for one relation could return the *wrong answer* for another. In-memory sources are
  therefore keyed by object identity, and the entry pins them alive so a freed `id()`
  cannot be reused underneath it.
* **the optimizer config**, which decides selectivity constants, cost weights, and which
  rules run at all.
* **the learned statistics**, by the `kyber.learning.generation` counter rather than by
  content. Fingerprinting the content does not work: the feedback loop rewrites the stats
  after *every* execution — the exponential average keeps drifting and the q-error history
  keeps growing — so a content hash never repeats and the cache never hits (measured: 0
  hits in 8 identical runs). The generation instead advances only when the loop learns
  something a plan could turn on: a column measured for the first time, or a cardinality
  that corrected its prior by more than 10%. That is the same judgement the adaptive
  executor makes — re-optimize when reality disagreed with the estimate, not because a
  smoothed average moved in its fourth decimal. The `MetadataHub` itself is keyed by object
  identity, so resetting it invalidates every entry.

A hit returns exactly what a miss computes. Correctness does not depend on the key being
*complete* in the "captures every input" sense — an over-broad key would return a plan
optimized for slightly different statistics, which is still a **semantically correct** plan
(that is what the optimizer's differential tests guarantee), merely a possibly worse one. It
depends on the key capturing everything that can change a plan's *meaning*: the plan itself
and the data its pruning decisions read. Both are keyed exactly.
"""

from __future__ import annotations

from batcher.kyber.plan_cache.keys import cache_key
from batcher.kyber.plan_cache.memo import clear, lookup, misses, record_outcome, served, store
from batcher.kyber.plan_cache.writes import record_write

__all__ = [
    "cache_key",
    "clear",
    "lookup",
    "misses",
    "record_outcome",
    "record_write",
    "served",
    "store",
]
