# Batcher vs Databricks — Execution-Engine Parity Scorecard

**Date:** 2026-07-18. **Branch:** `perf/aggregate-and-dist-merge`.

This is a code-checked comparison of Batcher's **execution engine and optimizer** against the
Databricks stack as it exists in July 2026. It is a companion to
`competitive_architecture.md`, which scores Batcher against DuckDB/Polars/Spark/Ray Data; this
document scores it against the one competitor that is simultaneously an optimizer, a vectorized
engine, and an enterprise governance platform.

**Scope discipline.** Batcher is not a storage solution. Delta/Iceberg/Hudi table *formats*,
Liquid Clustering, deletion vectors, Predictive Optimization's file compaction, and Unity
Catalog as a *metastore* are storage-layer concerns and are explicitly **out of scope** for
parity. They appear below only where they feed an execution-time decision (e.g. Delta per-file
statistics feeding Dynamic File Pruning — that is a *pruning* capability, and it is in scope).

**Evidence rules.** Batcher claims carry `file:line`. Databricks claims carry a primary source
or are marked ⚠️ unverified. Several widely-repeated Databricks numbers are **not** primary-
sourced; they are listed in the final section so nobody in this repo cites them.

---

## 0. What Databricks actually is, in July 2026

Three engines, not one, and the version landscape is further along than most write-ups assume:

| Engine | Role | Status |
|---|---|---|
| **Spark/Catalyst** (JVM) | planner + fallback executor | Spark 4.2.0 GA (2026-07-14); DBR 18.2 GA = Spark 4.1.0; DBR 19 Beta = 4.2.0 |
| **Photon** (C++) | vectorized execution layer under Catalyst | GA, default-on for all serverless surfaces |
| **Reyden** | new engine powering **Lakehouse//RT** | **Beta**, announced DAIS 2026-06-16 |

Catalyst plans; Photon executes what it can and pivots columnar→row into the JVM for what it
can't. Reyden is a separate real-time serving class, read-only.

**Reyden is real** ([Lakehouse//RT blog](https://www.databricks.com/blog/introducing-lakehousert-real-time-performance-unified-lakehouse),
[docs](https://docs.databricks.com/aws/en/compute/sql-warehouse/real-time)) — named "Reynold's
Dream Engine" after Reynold Xin. Databricks confirms only: fully asynchronous execution model,
no JVM, open formats, Unity Catalog required. Claimed sub-100ms at 12,000 QPS, up to 16x.

But its **Beta limits make it a narrow comparator**: read-only ANSI `SELECT` (no
INSERT/UPDATE/DELETE/MERGE, no DDL), no GEOGRAPHY/GEOMETRY, **no row-level security or ABAC**.
And the benchmarks are contested — [ClickHouse could not reproduce](https://clickhouse.com/blog/databricks-reyden-benchmark-transparency-clickhouse)
the competitor "crash" shown in the keynote (it was ClickHouse's designed 1,000-concurrency
admission limit), and the datasets were TPC-H SF1 and ~22K rows of NYC taxi. StarTree notes
Lakehouse//RT publishes **average, not P99**, latency.

**Treat Reyden as a directional signal, not a parity target.** A read-only, no-RLS Beta is not
something to chase feature-by-feature. What it signals is real: Databricks believes the next
axis of competition is **high-concurrency low-latency serving on open formats**, which is a
workload class Batcher does not currently address at all (see §6).

---

## 1. The adaptivity claim — where Batcher actually stands

This is the moat claim in `CLAUDE.md`, so it deserves the sharpest possible statement. The
research **confirms** the honest framing already in that file, and lets us sharpen it further.

**Spark AQE adapts only at stage boundaries.** Confirmed at source level, not from docs: each
`Exchange` becomes a `QueryStageExec`; on materialization `reOptimize()` re-runs a small rule
set over the remaining plan, adopted only if a `CostEvaluator` says cost ≤ current. There is
**no operator-internal or mid-stage adaptation anywhere in Spark.** Batcher's breaker tuple —
`(Aggregate, Sort, Distinct, Window, Limit, Join, Union)`, the symbol `ln` in
`api/adaptive/plan_surgery.py` — is the same granularity. `CLAUDE.md`'s "same granularity as AQE, not finer" is correct and now
source-backed.

> **Note on references.** This document was written while a concurrent refactor was moving
> files (`api/adaptive.py` → `api/adaptive/`, `kyber/rules/join_order.py` →
> `kyber/rules/joins/order.py`, and more). Line numbers are therefore indicative; symbol
> names are the durable anchor. Re-verify before quoting a `file:line` externally.

**But Photon *does* adapt below that level** — and this is the nuance the repo currently
misses. Per the [SIGMOD 2022 paper](https://people.eecs.berkeley.edu/~matei/papers/2022/sigmod_photon.pdf),
every Photon kernel adapts per batch to at least **has-NULLs** and **has-inactive-rows**, plus
case-by-case **ASCII-vs-UTF-8** string kernel selection, **sparse-batch compaction** before
hash probes, and adaptive shuffle re-encoding (detecting UUIDs-as-36-char-strings, ints-as-
strings). Databricks chose vectorized interpretation over codegen partly *because* "adaptivity
is natural when dynamic dispatch is already fundamental."

This is **not** plan re-optimization — Photon never re-plans. But it means the sentence "there
is no operator-internal adaptation anywhere" is true of *Spark*, not of *Databricks*. Batcher
should state the distinction precisely:

> Batcher re-optimizes plans at pipeline breakers on measured cardinalities (AQE granularity,
> available single-node) and learns across queries (which neither Spark nor Photon does).
> Photon adapts *kernels* per batch on data characteristics, which Batcher does not do.

Batcher has a partial analogue — `try_dict_compare` (19.6x on a low-cardinality filter) and the
JIT's four null-handling paths incl. a per-batch Kleene fallback (`bc-codegen/src/lib.rs:186`)
— but no null-free/sparsity/ASCII kernel specialization as a systematic layer.

**Where Batcher is genuinely ahead:** the cross-query learned loop. Spark's CBO is off by
default (§2); Photon does not learn. Batcher's calibrated cost model (`kyber/.../calibration.py:125`,
8 coefficients fit from measured `op_stats` with Bayesian shrinkage, refit every 64 rows) and
UCB1 join-algorithm bandit (`learned_tuning/`) have no Photon/Catalyst equivalent.

**The honest caveat:** Databricks has *two* production analogues that must be the benchmark,
not OSS Spark.
- **Automatic statistics** — with Photon, min/max/null/row-count gathered *during writes*, with
  **workload-aware column selection**, and background `ANALYZE` triggered by Predictive
  Optimization as stats degrade.
- **Enzyme** — [SIGMOD 2026 paper](https://arxiv.org/abs/2603.27775), honorable mention. Its
  incremental-vs-full-recompute cost model is **calibrated against historical execution profiles
  of structurally similar prior plans** via normalized physical-plan matching. On TPC-DI it
  picked the cheaper strategy **7/8 times (87.5%)**, roughly constant SF 1,000–10,000.

Enzyme is the closest published prior art to "plans sharpen the more a query runs," executed at
production scale. Any learned-stats claim in this repo should be positioned against Enzyme and
automatic statistics — **not** against `spark.sql.cbo.enabled=false`.

---

## 2. Optimizer parity

Spark's CBO gap is real and bigger than usually stated: `cbo.enabled=false` unchanged since
2.2.0 (2017), and because `LogicalPlanStats` switches to `SizeInBytesOnlyStatsPlanVisitor`,
that default means **no row counts anywhere in the planner** and **`ANALYZE TABLE` output is
silently discarded**. But Databricks [flips it on](https://docs.databricks.com/aws/en/optimizations/cbo),
so this gap is Spark's, not Databricks'. Compare against Databricks-on.

| Capability | Databricks | Batcher | Verdict |
|---|---|---|---|
| Rule-based rewrites | Catalyst, hundreds of rules | **302 rules / 7 phases**, live-verified registry; hybrid fixpoint with depth-scaled bound (`optimizer/driver.py:31,49`) | **Parity** |
| Join reordering | Selinger DP, 3–12 joins, **biased left-deep**; `JoinReorderDPFilters` states bushy suppression "not implemented" | **Bushy** DP: exhaustive ≤12 leaves, DPccp-style ≤20 / 200k pairs, greedy fallback (`kyber/rules/joins/order.py`, `_rebuild_dp` / `_rebuild_dphyp` / `_rebuild_greedy`) | **Batcher ahead** (bushy) |
| Cardinality estimation | histograms off by default even on Spark; Databricks auto-stats during writes | HLL/KLL sketches → `Provenance.SKETCH`; Selinger containment w/ PK-FK detection, MCV skew floors, range pruning (`estimator.py:726`) | **Parity**, different mechanism |
| Multi-column / joint histograms | Spark: off. Databricks: ⚠️ unverified whether they flip `histogram.enabled` | **Absent** | **Gap** (small — both weak) |
| Cost model calibration | Enzyme: history-calibrated on similar plans | 8 coefficients fit from measured stats, Bayesian shrinkage (`calibration.py:125`) | **Parity** |
| Cross-query learning | Enzyme + workload-aware auto-stats | learned stats + UCB1 bandit over {hash, broadcast, sort_merge} | **Parity**, Batcher broader in-query |
| Runtime bloom filters (shuffle) | SPARK-32268, **default-on since 3.4**; cascading (SPARK-45606) | Intra-operator bloom in `JoinTable::build` (`bc-runtime/src/join/mod.rs:61`) + **distributed Bloomjoin** pre-shuffle (`dist/executors/join.py:303`) | **Parity** |
| **Dynamic filter pushdown into scans** | **DPP** (3.0) + **Databricks DFP** — prunes on **non-partition** columns via Delta per-file min/max | **Absent.** No runtime-produced filter ever reaches a Parquet reader; `sink_runtime_filter_to_source` referenced (`evidence.py:26`) but **never implemented** | **Gap — the largest optimizer gap** |
| Join elimination via constraints | `RELY` UNIQUE constraints, DAIS 2026, Photon-only | join-elimination rules exist in REWRITE phase | **Parity** |
| Post-shuffle partition coalescing | AQE `coalescePartitions`, default-on | **Absent** — reducer counts sized a priori | **Gap** |
| Skew join handling | AQE: `size > 5× median AND > 256MB`, replicate non-skewed side, default-on | 3-tier detection + salt-probe/replicate-build — but `skew_join_salt=0` **default-off**, and **absent from the Flight transport** | **Gap — see §5** |

### The dynamic-filter gap, stated precisely

Batcher has bloom filters in two places that matter (intra-join, and pre-shuffle in the
distributed path). What it does **not** have is the thing DFP does: take a filter discovered at
runtime from the build side and use it to **skip files or row-groups on the probe side**.
Batcher's `kyber/rules/extra/runtime_filters/` is misleadingly named — all 14 rules are *static*
PUSHDOWN rewrites driven by *persisted* statistics; `sip.py:199` admits the IR has no bloom-probe
node.

This is an I/O-volume gap, not a CPU gap, and it compounds with the Parquet gaps in §4. On a
star-schema query where a small dimension filter selects 1% of a large fact table, Databricks
reads ~1% of the fact files; Batcher reads all of them and filters afterward.

---

## 3. Vectorized execution parity — the most serious gap

Photon's core bet: **columnar vectorized interpretation, not codegen**, chosen deliberately for
debuggability, per-operator observability, and natural adaptivity. Batcher bets on Cranelift
JIT with an interpreter oracle. Both are defensible. The problem is coverage.

| | Photon | Batcher |
|---|---|---|
| Model | C++ vectorized interpreter, columnar | Arrow-columnar interpreter (`bc-expr`) + Cranelift JIT (`bc-codegen`) |
| Expression coverage | broad; falls back per-node to JVM | **JIT: 8 of 46 `Expr` variants** (`analyze.rs:32-306`); 13/20 `BinaryOp`, 17/26 `MathFunc` |
| Types accelerated | Struct, Array, Map, **Variant, Geometry, Geography, collated strings** | **JIT: 5 types** — I64, F64, Bool (internal-only), Date32, Timestamp(µs) comparison-only. **No string, no decimal, no Int32/Float32** |
| String representation | dictionary + ASCII-vs-UTF-8 adaptive kernels | **Zero `StringView`/`Utf8View` in the repo** (verified); dictionaries decoded at the leaf |
| SIMD | pervasive, core to the design | **Real** Cranelift vector IR (`simd.rs`, F64X2/4/8, bitselect, extractlane) — but subset **narrower than the JIT** (no Div/Mod/Case/Math), AVX-512's 8 lanes never auto-selected, `simd_unroll=1` |
| Fallback | transition node, columnar→row pivot | silent fallback to interpreter |

Two implementation details worth fixing regardless of strategy:

1. **No compile threshold** — but this is *smaller than it looks*, and an earlier draft of this
   document overstated it. `try_compile` (`ops/mod.rs`) is indeed unconditional on row count,
   and `rows_in` is in scope at the call sites yet unused. However: (a) the compile is
   **memoized process-wide**, so the cost is cold-cache-only — the one recorded datum is 16.6 ms
   on a 64-row query (`bc-codegen/src/cache.rs` header), which *motivated* the memo; and (b) the
   **default streaming executor never JITs at all** (`EngineConfig.streaming = true`;
   `stream/mod.rs` calls the non-JIT `filter_batch`/`project_batch`), so the overhead is confined
   to the non-default materializing `par.rs` path. A threshold would also have to be
   lookup-first/compile-above-threshold to avoid *losing* on cache hits, and would flip the
   `backend` tag that `tests/integration/test_jit_cost_parity.py` asserts on small fixtures.
   **Net: low priority, non-trivial risk** — not the easy win it first appeared to be.
2. **Cache key is a `String` including every schema field**, cleared wholesale at 1024 entries
   with no LRU (`cache.rs`). The clear is a cliff, not eviction: a long-lived driver crossing
   1024 shapes re-pays Cranelift for every live operator at once. Note an LRU would turn today's
   contention-free **read**-lock hit path into a write-lock path unless recency is an atomic
   stamp — that constraint, not the eviction policy, is the real design question.

The interpreter itself is in good shape — genuinely columnar over arrow-rs kernels, 57 string /
26 math / 21 date / 18 list functions. **The gap is that the accelerated path is narrow**, and
the widest-impact missing piece is strings: no `StringView`, and no JIT string support at all.
Photon's ASCII-vs-UTF-8 adaptive kernels are a direct answer to a workload Batcher currently
runs entirely through the interpreter.

---

## 4. Operator and I/O parity

Batcher's operator surface is broad — 15 `RelOp`s, 29 aggregates, 18 window functions, 6 join
algorithms, and **everything material spills** (grouped aggregate with grace + recursive
repartition depth 4, distinct, sort, hash join, ASOF, window, and quantile/mode/histogram via
external sort). Spill coverage is genuinely competitive.

**Operator gaps against Photon's supported set:**

| Gap | Detail |
|---|---|
| **Nested-loop join** | No NLJ *operator*, but the capability is now covered by rewrite. **INNER** theta (`ON a.x < b.y`) lowers to `cross_join + filter`; **LEFT/RIGHT** theta lower to that plus the preserved side's unmatched rows recovered via a row-index anti-join and null-extended by a left join against an empty relation (23 differential tests). **Still open:** FULL theta; and because the cross product materializes there is no streaming/blocked NLJ and no band-join optimization — Photon has both. |
| Spatial join | Photon GA; Spark 4.2 adds GEOMETRY/GEOGRAPHY + ST_*. Batcher: absent. |
| Right-semi / right-anti join | **Closed 2026-07-18.** No engine `JoinType` variant was added: `A RIGHT SEMI JOIN B` *is* `B SEMI JOIN A`, so it runs as the left-driven join over swapped operands (the ON equalities are mirrored with them, since key sides are read positionally). Adding `RightSemi`/`RightAnti` would have forked the shared wire-contract enum across four crates to express what operand order already says. |
| Cross join | no IR node — rewritten to equi-join on a synthesized constant key (`frame.py:1477`) |
| GROUPING SETS / ROLLUP / CUBE | no IR node — SQL-frontend expansion to UNION ALL. **Correction: this works and matches DuckDB, and `GROUPING()`/`GROUPING_ID()` are implemented** (per-level bit-vector constants). An earlier draft of this document wrongly listed them as missing. The gap is the *plan shape* (n levels = n scans), not the feature. |
| Ordered aggregates (`string_agg(x ORDER BY y)`) | **Closed 2026-07-18.** Any `ORDER BY` inside an aggregate used to raise `unsupported SQL expression: Order`. The list aggregate appends in input order, so the input is sorted once up front — the same shape as the DISTINCT pre-dedup (9 differential tests, ASC/DESC/global/`array_agg`/nulls). Two aggregates wanting *different* orderings reject, since one pass cannot serve both. |
| DISTINCT aggregates | **Closed for the single-expression case 2026-07-18**: `SUM/AVG/...(DISTINCT x)` now dedups on `(group keys, x)` once and aggregates normally (14 differential tests). Still rejected — cleanly — when a non-DISTINCT aggregate appears alongside, or two different DISTINCT expressions are used; both need multi-phase aggregation joined on the group keys, as DuckDB and Spark do it. |
| `RANGE` window with numeric offset | not supported. **Fixed 2026-07-18:** the interpreter used to *silently downgrade* it to peer-RANGE (a wrong answer); it now returns `InterpError::ValueBasedRangeFrame`. Severity was lower than first assessed — the Python layer already rejected this shape (`plan/logical/window.py` raises `PlanError`; the SQL parser raises `NotImplementedError`), so only directly-constructed IR was exposed. Implementing real value-based frames still needs typed order-key arithmetic |
| Window `IGNORE NULLS` | **Partly closed 2026-07-18.** `last_value(x IGNORE NULLS)` over the default frame and `first_value(x IGNORE NULLS)` over `CURRENT ROW AND UNBOUNDED FOLLOWING` are exactly the runtime's forward/backward fills, so they map onto them — no new operator (8 differential tests). `lag`/`lead`/`nth_value` with IGNORE NULLS, and value functions over other frames, still reject rather than return the null-*respecting* answer, which would be a wrong result. |
| Window `EXCLUDE` | absent (rejected explicitly) |
| `QUALIFY` with the window function only in QUALIFY | **Closed 2026-07-18.** The idiomatic top-N-per-group spelling (`... QUALIFY row_number() OVER (...) = 1`) used to raise — QUALIFY worked only when the SELECT computed the window function and QUALIFY referenced its alias. The window expression is now materialized under a hidden `__qualify<n>` column, filtered, and dropped by the projection (11 differential tests). Still rejected: QUALIFY over a GROUP BY. |
| SQL `PIVOT` / `UNPIVOT` | **Closed 2026-07-18.** Both used to raise "use the Dataset.pivot(...) method"; the SQL modifier now maps onto the relational `Dataset.pivot`/`unpivot` the engine already had (8 differential tests, incl. a value omitted from the `IN` list being dropped rather than merged). |
| Recursive CTEs (`WITH RECURSIVE`) | **Closed 2026-07-18.** Evaluated as a control-plane fixpoint — anchor, then repeatedly run the recursive term against only the previous iteration's rows — so no engine change was needed (10 differential tests incl. graph reachability). `UNION` (distinct) feeds forward only *new* rows via an engine anti-join, which is what makes the degenerate `SELECT 1 FROM c` case terminate; a runaway recursion raises at 1,024 iterations rather than hanging. Spark only added these in 4.1. |
| `LATERAL` (no FROM) | **Closed 2026-07-18.** `FROM t, LATERAL (SELECT <exprs>)` with no FROM of its own is one row in / one row out computed from the outer row — exactly `with_columns` (9 differential tests). A lateral that *reads a table* is a correlated join and rejects. Worth noting: the reject check initially read only sqlglot's `from` key when this version uses `from_`, so a correlated lateral **without a WHERE** slipped through and was mistranslated into computed columns; there is a test for precisely that shape. |
| Correlated `LATERAL`/APPLY, `MATCH_RECOGNIZE` | absent (Databricks added `MATCH_RECOGNIZE` Beta at DAIS 2026) |

**I/O — format breadth is a Batcher win, Parquet depth is a gap.** ~60 registered
sources/sinks, and all three lakehouse formats read (Delta via delta-rs with time travel/CDF,
Iceberg via pyiceberg incl. REST/Unity/Polaris catalogs, Hudi via hudi-rs). Predicate pushdown
works at three layers (plan-time row-group, plan-time file pruning from add-actions, read-time
row-group in Rust).

But: **no Parquet page index** (no `ColumnIndex`/`OffsetIndex` anywhere — the `ParquetSource:50`
docstring claiming "page pruning" is **unsupported by code and should be corrected**); **Parquet
bloom filters neither read nor written**; **no late materialization** (the Rust reader never uses
`RowFilter`, so rows are filtered by the engine's `Filter` after a full row-group read); writer
sets only `compression`; **no ORC statistics/zone maps**; **no column rename** in schema
evolution (name-matching, no field-id resolution).

---

## 5. Distributed parity

Batcher's distributed layer is its strongest area and beats Spark structurally in places:
peer-to-peer Arrow Flight `DoExchange` with **no shuffle service**, credit-based flow control
whose bound is *proven* by an in-flight gauge (`store.rs:16-62`), a zero-copy mmap `/dev/shm`
same-node fast path, TLS/mTLS with constant-time bearer tokens, and **four layers of fault
tolerance** ending in Spark-style lineage recompute with epoch-tagged tickets.

Three gaps, one of which is serious:

1. **No AQE-style post-shuffle coalescing.** Spark's `coalescePartitions` is default-on with a
   1MB floor. Batcher sizes reducer counts a priori.
2. **Skew mitigation and bloom pruning were implemented only on the *disk* transport** —
   `resolve_transport` picks Flight whenever `nodes > 1`, so a genuine multi-node cluster got
   neither. Not a missing feature: a feature that existed and was silently bypassed on the path
   that matters. An attempt to wire skew salting through
   Flight was made and reverted on 2026-07-18 — see the P0 entry in §7 for why, including a
   test-ordering dependence that must be fixed before any such change can be trusted.
3. **Defaults ship the capability off.** `skew_join_salt=0` (correctly — see the P0 hazard
   below; it now has a `salting_is_safe` guard), `locality_aware_scheduling=False`,
   TLS off, adaptive off below 20M rows
   (`_ADAPTIVE_MIN_INPUT_ROWS`, `api/adaptive/gating.py` — a module constant, **not** a config field).

That last pattern — real code behind a default-off flag — is the single largest discrepancy
between what this codebase contains and what a user actually gets. Databricks' equivalents
(AQE, coalescing, skew join, runtime bloom filters, Photon) are all **default-on**.

**Three of these are now closed** (see `ray_pitfall_parity.md` for the full audit):

- **`stream_inference` defaults to True.** The costliest of the set — with it off, every
  distributed batch-inference pipeline ran its CPU decode inside the GPU actor, which is the
  #1 GPU-utilization bug in the Ray guides.
- **`speculation_max_backups` defaults to 1.** One straggler backup per barrier, gated on
  both a learned straggler factor and a minimum finished fraction. Ray Data has no straggler
  mitigation at all.
- **`shuffle_replication` is wired** for the flat aggregate reduce
  (`dist/shuffle_replication.py`), so the `spot` profile's factor of 2 now actually buys
  re-fetch recovery instead of silently getting recompute. The combiner-tree path is still
  unwired.

The question to ask of each remaining entry is: *is the default-off protecting against a real
risk, or only against the absence of a benchmark?* For `skew_join_salt` the answer turned out
to be **a real risk** — it can silently split a group under a finalizing reducer (P0 §1), so
it correctly stays off, and the hazard now has a `salting_is_safe` guard rather than only a
warning in this document.

---

## 6. Enterprise parity

The governance **rewrite** is genuinely well-built and is a real parity win in mechanism: grants,
column masks, tag masks, and row filters resolve to a **pre-optimizer plan rewrite anchored at
the leaf** (`enforce.py:124-161`) — masking precedes any user filter/join, denied columns are
*removed* not flagged (fail-closed), filters sit below projections so a policy can reference
unselectable columns, most-restrictive-wins, with a regression test that Kyber cannot undo it.
Column-level lineage is real and correctly over-approximates on opaque `map_batches`.
EXPLAIN/EXPLAIN ANALYZE, JSON profiles, OpenTelemetry traces and a per-query event log exist.

Notably, Unity Catalog implements row filters/column masks as **SQL UDFs evaluated at query
time**, which is why *time travel and CLONE do not work* on filtered/masked tables. Batcher's
plan-rewrite approach is architecturally cleaner.

**But four gaps block enterprise deployment outright, and they are not incremental:**

| # | Gap | Detail |
|---|---|---|
| 1 | **No authentication** | `Principal` is caller-asserted and carries no credentials *by explicit design*. `bt.Principal("root", roles=["admin"])` bypasses every policy. Governance is **opt-in per `security()` block** — outside one, plans pass through ungoverned. In-memory and stream sources are ungovernable entirely (`_binding.py:32`). Tables key on **exact path strings**, so aliasing evades policy. No revoke; no privileges beyond SELECT — **writes are entirely ungoverned**. |
| 2 | **No multi-tenancy** | Zero hits for `tenant`. No quotas, workload groups, priority scheduling, or charge-back — compounded by process-global shared result cache, plan cache, and UDF pool. Databricks' comparator is concrete: queue depth 1,000; warehouse sizes doubling 2X-Small (1 worker) → **5X-Large (512, Public Preview, pro + serverless only)**; a documented autoscaling ladder (2–6 min of query load → +1 cluster, 6–12 → +2, 12–22 → +3, >22 → +3 plus 1 per further 15 min; also triggered by any query queued 5 min; scale-down after 15 consecutive low-load minutes); and serverless-only Intelligent Workload Management. ⚠️ Serverless may use different instance types than the pro/classic size table, so those worker counts do not transfer to serverless capacity planning. |
| 3 | **No UDF sandboxing** | The process pool exists for GIL escape, not isolation (`processes.py:1-8`). No seccomp, namespaces, rlimits, timeouts, or import allowlist. `forkserver` children inherit the parent environment, so **a UDF can read the `env:` key material** that the otherwise-careful `keyref.rs` design protects. |
| 4 | **No encryption at rest** | Spill files and result cache are **plaintext**, not even `chmod 0600`. No KMS/Vault integration; key refs are `env:`/`file:` only. |

Gaps 1 and 3 compose into a concrete attack: an unauthenticated caller asserts an admin
`Principal`, runs a UDF, and reads credentials from the inherited environment. That is not a
feature gap; it is an exploitable path, and it should be triaged as such.

**Also absent:** Prometheus/OTel *metrics* (traces only), queryable query history, live progress
and **query cancellation**, lineage export (OpenLineage/Atlas/DataHub), materialized views and
incremental view maintenance (vs Enzyme), engine-native transactions, tokenization/FPE/
differential privacy. **Present:** MERGE/upsert with the full clause set, CDC apply, data-quality
expectations with quarantine, Unity Catalog credential vending.

**Databricks-side context worth knowing:** ABAC went GA 2026-05-13 with `CREATE POLICY ... ON
{CATALOG|SCHEMA|TABLE}` and `has_tag()` predicates (10,000 policies/metastore, 50/table). Result
caching has four distinct layers including a **remote result cache** shared across all warehouses
in a workspace. ⚠️ Whether row filters/column masks affect result-cache eligibility is
**genuinely undocumented on the Databricks side** — a real open question, and one Batcher must
answer deliberately for its own cache if governance and caching are ever combined.

---

## 7. Prioritized parity roadmap

Ranked by severity × tractability. P0 items are not "parity" so much as "correctness and
safety."

**P0 — blocking, and mostly not features**
1. **Wire skew mitigation + bloom pruning into the Flight transport** (§5.2). Still open.
   `resolve_transport` picks Flight whenever `nodes > 1`, so a real multi-node cluster gets
   neither. **An attempt on 2026-07-18 was written and then reverted** — recorded here because
   the reasons are the useful part:
   - *It cannot be validated in a single-node dev environment.* Ray cannot schedule the Flight
     fleet there; the pre-existing `test_flight_join_*` tests hang identically. Any Flight
     change must be validated on a real multi-node cluster before it is trusted.
   - *The distributed join tests are order-dependent, which makes A/B testing unreliable.*
     Learned hot keys **persist in-process** (`dist/skew.py` → `MetadataHub`), so an early test
     teaches the system a key is hot and a later test then salts where it otherwise would not.
     Two A/B runs of the same change gave opposite verdicts (6→1 failures run in isolation;
     0→5 failures run in suite order). **Fix this ordering dependence first** — it is a
     prerequisite for trusting any skew work, and arguably a robustness bug in its own right.
   - *A genuine design conflict remains for bloom.* It needs build-then-probe ordering, but the
     Flight map barrier publishes both sides in one call — deliberately, to stop a left-side
     failure masquerading as an unreachable worker. Settle that before coding.
   - *A real hazard was identified and is worth fixing independently of transport:* salting
     spreads a hot key across buckets, but a fused join+aggregate is finalized per bucket
     precisely because group keys ⊇ the join key put each group in exactly one bucket. Salting
     therefore makes each reducer finalize a *partial* group — a silent wrong answer. The disk
     path can reach this today whenever a hot key is detected. Guard: no salting when the
     reducer is a finalizing aggregate.
     **The guard now exists** — `dist/executors/join.py::salting_is_safe` refuses to salt when
     `reducer_ir` is set, with unit coverage in `tests/unit/test_skew_salt_eligibility.py`.
     Note this hazard was reachable on the **default** path, not only under an opt-in: a
     measured hot key engages salting even at `skew_join_salt=0`. The guard is not pinned by
     an end-to-end cluster test for the reasons in this same entry (single-node dev cannot
     validate Flight join behaviour) — that test is still owed.

2. ~~Fix the `RANGE`-window silent downgrade~~ — **done 2026-07-18.** Now
   `InterpError::ValueBasedRangeFrame` with four regression tests
   (`ops::window_frame_tests`); the stale "falls back" doc comments in `bc-ir` and `bc-interp`
   were corrected to match.
3. **Enterprise security floor:** authentication, mandatory-governance mode, UDF sandboxing
   (rlimits + env scrubbing at minimum), spill/cache encryption. Until these land, "enterprise
   parity" cannot be claimed at all.
4. ~~Correct the `ParquetSource` page-pruning docstring~~ — **done 2026-07-18.**

**P1 — largest real performance gaps.** All of these are now benchmark-gate-able (§7c); the
measured TPC-H sf1 losses (q20/q3/q21/q17/q2, all join-heavy) are the shapes to aim at.

5. **Dynamic filter pushdown into scans** (§2) — implement `sink_runtime_filter_to_source`, add a
   runtime-filter IR node, and let a build-side bloom prune probe-side row-groups/files. This is
   Databricks' DFP, the single biggest optimizer gap, and it compounds with #6.
6. **Parquet depth:** page index (`ColumnIndex`/`OffsetIndex`), bloom filter read/write, and
   late materialization via `RowFilter`. Together with #5 this is the I/O-volume story.

   **Measured 2026-07-18, and it changes the shape of this item.** On TPC-H sf1 `lineitem`
   (49 row groups x 122,880 rows), `l_orderkey < 100` matches **105 rows** but row-group
   pruning still decodes a whole row group — **122,880 rows, a 1,170x decode
   amplification**. That is the size of the prize.

   But the page index must be *written* before it can be read, and **neither writer here
   emits one**: verified with parquet-rs that both DuckDB-written files and
   **Batcher's own `write.parquet` output** report `column_index present: false,
   offset_index present: false`. Batcher's writer sets only `compression`.

   So this is a **two-sided** item, and the write half must come first or the read half
   is dead code on any self-produced file. Spark/Databricks writers do emit a page index,
   so read support still pays on ingested data — but the internal win needs both.

   **The write half landed 2026-07-18.** `ParquetSink` now passes `write_page_index=True`
   on both paths (`_write_file` and `_open_stream_writer` — they construct the writer
   separately, so both needed it). Verified with parquet-rs that Batcher-written files now
   report `column_index present: true, offset_index present: true`; **size overhead measured
   at 0.0 KB / 0.00%** on a 300k-row file. Regression tests in
   `tests/io/test_parquet_page_index.py` (they fail on the old writer).

   **Probing the result then found a much larger writer defect.** With the index present it
   became visible that Batcher emitted **1 page per row group**, so page-index *reading*
   could never prune more than row-group pruning already does. The cause: row groups were
   far too small — and the two write paths were wrong in **opposite** directions.

   | | row groups | rows/group | size |
   |---|---|---|---|
   | streaming write, before | **1,459** | 4,113 | 218.9 MB |
   | whole-table write, before | **1** | all rows | — |
   | after (both paths) | **46** | 130,461 | **182.6 MB** |
   | DuckDB, same data | 49 | 122,473 | 207.1 MB |

   `ParquetWriter.write_batch` starts a row group per call, so a streamed write emitted one
   per morsel; `pq.write_table` with no `row_group_size` emitted one for the whole table,
   defeating row-group pruning and read parallelism outright. Both now target 128Ki rows
   (`_ROW_GROUP_ROWS`), streaming through a buffering wrapper that holds at most one group.

   Files are now **17% smaller than before and 12% smaller than DuckDB's**, with all
   6,001,215 rows verified identical via DuckDB `EXCEPT ALL` in both directions. This was a
   bigger win than the page-index reader would have been, and it had to come first.

   **Read-side page pruning is now quantified and deprioritized — with numbers.** After the
   row groups were fixed to 131,072 rows, pages per row group in TPC-H `lineitem` are:

   | column | pages / row group |
   |---|---|
   | `l_orderkey`, `l_partkey`, `l_suppkey` (int64) | **1** |
   | `l_comment` (string) | 4 |

   At Parquet's conventional 1 MB page size, 131,072 int64 values *are* one page — so
   page-level pruning can skip **nothing** on a numeric column and at best 4x on a wide
   string one. The headline "1,170x decode amplification" is real but is addressed by
   row-group sizing, **not** by the page index. Implementing the reader means restructuring
   `predicate.rs` — a module with an explicit superset-safety contract where a bug silently
   drops rows — for~zero measurable gain on this layout. **Do not build it** unless the
   writer also moves to much smaller pages, which is its own trade-off.
7. **Post-shuffle partition coalescing** (§5.1) — AQE-equivalent, and Batcher already has
   measured cardinalities at breakers, so the input is in hand.
8. **String acceleration:** `StringView`/`Utf8View`, and a null-free/ASCII kernel-selection layer
   analogous to Photon's per-batch adaptivity. Widest-impact JIT gap.

   **Scoped 2026-07-18 — it is two-sided, like the page index was.** Measured: **zero**
   `Utf8View`/`StringView` references exist anywhere in `crates/`, while `bc-expr` dispatches
   on `DataType::Utf8` in **24** places (and `LargeUtf8` in 8) — each needs an arm. And
   there is **no producer**: pyarrow's reader has no string-view option, so adding kernel
   support alone would be dead code on every file Batcher reads, exactly as read-side page
   pruning would have been. A real implementation therefore spans the reader, the 24
   dispatch sites, the FFI boundary, and the IPC/spill/shuffle encodings — and it touches
   the one-`Expr`-across-tiers contract. This is a multi-layer feature, not a tweak; size it
   accordingly before starting.

**P2 — coverage**
9. **Nested-loop join operator.** INNER, LEFT and RIGHT theta are closed by rewrite (§4).
   What still needs a real operator: FULL theta, a streaming/blocked NLJ so the cross product
   does not materialize, and band-join optimization.
10. GROUPING SETS/ROLLUP/CUBE as a real **IR node** (they work today via UNION-ALL expansion,
    so this is a performance item — n levels currently means n scans — not a coverage one).
    Multi-phase aggregation for the general DISTINCT case. The DISTINCT
    case has a known shape — dedup on `(group keys, distinct expr)` before aggregating — but the
    general form (mixing DISTINCT and non-DISTINCT aggregates, or several *different* DISTINCT
    expressions) needs multiple aggregation phases, as DuckDB and Spark do it. Doing only the
    single-shared-expression subset **was** implemented (parser-side, so `AggExpr` stays
    unchanged and no public API or IR contract moved); the general form still needs the
    multi-phase design.
11. JIT cache eviction cliff (§3). The compile *threshold* is deliberately **not** recommended —
    see §3.1 for why the memo and the streaming default already absorb most of that cost.
12. Right-semi/right-anti joins; window `EXCLUDE` / `IGNORE NULLS`; recursive CTEs; LATERAL.

**P3 — strategic**
13. Revisit the default-off pattern (§5.3). Each flag needs either a default-on plan or an
    explicit written reason it ships off.
14. Materialized views + incremental view maintenance, benchmarked against Enzyme's published
    87.5% strategy-selection accuracy.
15. Evaluate whether high-concurrency low-latency serving (the Reyden workload class) is a
    direction this project wants at all. It is currently unaddressed, and it is a deliberate
    scope decision, not an oversight.

---

## 7b. Silent wrong answers found while doing this work

Three, all in shapes no existing test covered. Recorded because the *pattern* matters more
than the individual bugs: each was a feature that appeared to work, returned plausible
numbers, and was wrong.

1. **SQL `SEMI`/`ANTI` joins executed as INNER joins.** sqlglot carries SEMI/ANTI in a join's
   `kind`; the translator read only `side`, so the kind was dropped. `ANTI JOIN` returned the
   rows that *matched*, with the right side's columns attached — the exact inverse of its
   meaning, with no error. `RIGHT SEMI/ANTI` likewise became a plain RIGHT join. Fixed: semi
   and anti match DuckDB, and RIGHT SEMI/ANTI now run correctly via the operand swap.
2. **`RANGE <n> PRECEDING` windows silently degraded** to a peer-`RANGE` running aggregate —
   a different frame. Now `InterpError::ValueBasedRangeFrame`. (Reachable only via
   directly-constructed IR; the Python layer already rejected it.)
3. **Skew salting breaks a fused join+aggregate.** Salting spreads a hot key across buckets,
   but the fused aggregate is finalized per bucket precisely because group keys ⊇ the join key
   put each group in one bucket. Each reducer then finalizes a partial group. Not yet fixed —
   it surfaced during the reverted Flight work (§7 P0) — but the **disk path can reach it
   today** whenever a hot key is detected. Guard: no salting when the reducer is a finalizing
   aggregate.

The common thread is the one `CLAUDE.md` already warns about: each needed an operator combined
with a non-default flag or a non-default path. `tests/differential/test_diff_operator_matrix.py`
is the right home for that cross-product, and none of these were in it.

## 7c. Benchmarking is NOT blocked — measured TPC-H baseline

An earlier revision of this document claimed perf work could not be validated because
`benchmarks/run.py` needs `s3://ray-benchmark-data/`. **That was wrong**, and it wrongly
gated several parity items. The corpus can be generated locally in about a minute:

```bash
python - <<'EOF'
import duckdb, os
sf, base = 1, "/tmp/tpch"
con = duckdb.connect(); con.execute("INSTALL tpch"); con.execute("LOAD tpch")
con.execute(f"CALL dbgen(sf={sf})")
for (t,) in con.execute("SHOW TABLES").fetchall():
    os.makedirs(f"{base}/sf{sf}/{t}", exist_ok=True)
    con.execute(f"COPY {t} TO '{base}/sf{sf}/{t}/part0.parquet' (FORMAT PARQUET)")
EOF
export BENCH_TPCH_BASE=/tmp/tpch
python benchmarks/run.py --benchmark tpch --family tpch --scale 1 --engines batcher,duckdb
```

`BENCH_TPCH_BASE` (and the `BENCH_CLICKBENCH_BASE` / `BENCH_TPCDS_BASE` siblings) redirect
the loader at any local path. **Every perf-relevant item below is therefore gate-able**, and
no future change should be deferred on "benchmarks can't run here".

### Measured, 2026-07-18 — TPC-H sf1, batcher vs native DuckDB, all 22 queries

Correctness-gated: the harness refuses to time a query whose result does not match, and
**all 22 passed**.

- **Geomean ratio 1.26x in DuckDB's favour.** DuckDB faster on **16 of 22**.
- Batcher wins 6: q1 (0.77x), q6 (0.71x), q9 (0.86x), q11 (0.74x), q12 (0.73x), q15 (0.49x).
- Worst losses: **q20 (2.84x), q3 (2.67x), q21 (2.24x), q17 (2.23x), q2 (1.80x)** — all
  join/subquery-heavy, which is where the optimizer and join execution earn their keep.

### Measured parallel scaling — the likely source of the join-heavy losses

q3, plan built once and executed 5x per setting (so parse/plan cost is excluded):

| `parallelism` | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| q3 exec (ms) | 171.3 | 92.4 | 86.6 | 43.3 | 42.8 |

**4.0x speedup from 16 cores — ~25% parallel efficiency — and it plateaus after 8.** On a
box where DuckDB uses all 16, that alone can account for the join-heavy gap. This is the
most actionable perf lead in this document, and it is measurable in under a minute with the
recipe above.

**Narrowed by operator** (16 cores vs 1, sf1, plan built once):

| shape | speedup |
|---|---|
| `GROUP BY` alone (lineitem) | **19.2x** |
| join alone (orders ⋈ lineitem) | **5.9x** |
| join + `GROUP BY` | 12.4x |

So the **join is the ceiling**, not the aggregate. Three plausible causes were tested and
**each is ruled out**, which is the useful part — it stops the next person re-testing them:

- *Serial hash-table build?* No. A **big** build (1.5M rows) scales **better** (5.7x) than a
  **tiny** one (150k, 3.2x). A serial build would show the opposite.
- *Fixed per-query overhead?* No. The floor is **0.12 ms**.
- *Output gather being memory-bandwidth bound?* No. Gathering **more** columns scales
  **better** (4 cols 3.4x, 1 col 3.0x, `count(*)` 2.0x).

What remains is an unexplained roughly-constant serial component: p=1 time barely moves
(162/154/160 ms) as output width grows, while p=16 time falls (84/52/47 ms). That is the
signature of a fixed serial section inside the join, and identifying it needs a **real
profiler (perf / flamegraph) on `bc-interp`'s join path** — black-box timing has been taken
as far as it goes. That is the recommended next step, ahead of any of the P1 features.

⚠️ Two profiling caveats found while measuring, both worth fixing because they mislead:
`EXPLAIN ANALYZE` reported `cpu utilization: 6% of cores` for a query that in fact achieves
a 4x speedup, and its per-operator percentages do not sum to 100% (one hash join was
attributed *294%* of a 718 ms wall time). **Do not size a perf decision from those two
fields** until they are corrected.

### Measured improvement: morselizing the join probe (2026-07-18)

The scaling investigation above pointed at the join, and the cause turned out to be
scheduling, not algorithms: the streaming broadcast probe fans its input across cores with
`par_iter`, so its parallelism was bounded by **however many batches the child emitted**,
not by core count. A bare scan hands over the source's own chunking — TPC-H sf1 `orders`
arrives as 13 batches — leaving 16 cores with 13 units of work and a long imbalanced tail.
Every other parallel arm (filter, project, aggregate) already re-morselizes; the join probe
was the one that did not. One line in `par.rs`:

```rust
let probe = ops::morselize_par(&left_batches, opts.morsel_target());
```

**Result — TPC-H sf1, 3 runs per arm, per-query minimum, one release build per arm:**

- **geomean 0.926 — 7.4% faster overall**
- **zero regressions** (nothing worse than 1.05x)
- 15 of 22 queries measurably better, and the wins land exactly on the join-heavy shapes the
  gap analysis predicted: **q21 0.86, q4 0.87, q18 0.87, q22 0.88, q5 0.89, q19 0.90**
- all 22 correctness checks pass; 428 join differential tests pass; `bc-interp` Rust tests
  clean; the 3 spill failures reproduce identically with the change reverted (pre-existing)

**This is the same change an earlier attempt in this session "measured" as a regression and
reverted.** That attempt was invalid: it rebuilt the extension between arms and compared
single runs of a bimodal benchmark. The methodology above (build once per arm, repeat, take
the per-query minimum, verify the artifact size) is what turned a false negative into a
clean 7.4% win — which is the practical argument for the stability rules in this section.

### ⚠️ Benchmark stability — read before trusting any single run

Two runs of the **same code and same build** gave q17 = **2.25x** then **8.70x**. Isolated in
fresh processes it is stable at **155-157 ms vs DuckDB ~19 ms ≈ 8x**, so the low readings are
the outlier and **the 1.26x geomean above is optimistic**. Consequences, all learned the hard
way here:

- **Never A/B a perf change on single runs.** Repeat each arm several times and compare
  distributions; some queries are effectively bimodal in-suite.
- **Check the build mode before trusting any timing — this is the big one.** `maturin
  develop` installs a **debug** build by default, and debug vs release is a **10.6x**
  difference on TPC-H q1 (410.8 ms vs 38.9 ms). The installed artifact is easy to identify:
  `python/batcher/_native.abi3.so` is ~305 MB debug, ~46 MB release. During this session a
  concurrent process replaced the release build with a debug one mid-measurement; that is
  the actual root cause of a "regression" that was chased and reverted here, and of an A/B
  whose two arms disagreed. **Verify the artifact size at the start and end of any
  benchmark run**, and do not run perf work while anything else may rebuild.
- **Rebuilding between arms invalidates the comparison** even at the same optimisation
  level. Build once, then A/B.
- **Re-planning is expensive and is excluded by the harness.** q17 re-planned each iteration
  takes ~570 ms against ~157 ms with the plan reused: **optimizer time exceeds execution time
  by ~2.5x** for this query. That overhead is invisible in the benchmark but real for a
  short interactive query, and is worth profiling in its own right.

This independently corroborates `competitive_architecture.md`'s figure (it reports 16 of 21
and geomean 1.36x); re-measured here at sf1 the gap is 1.26x. Either way the honest reading
is the same: **Batcher currently loses to native single-node DuckDB on TPC-H**, wins on the
scan/aggregate-bound shapes, and loses most on the join-heavy ones. Any "faster than DuckDB"
claim must name the shape.

## 8. Claims not to make

Both directions. Cite none of these without new primary evidence.

**About Databricks:**
- ⚠️ Reyden's relationship to Photon — sources **directly contradict** each other ("integrates
  with Photon, little difference" vs "ground-up rewrite, not an update to Photon"). Both secondary.
- ⚠️ Reyden as a "factory of engines" with ML-selected execution strategy — **secondary only**,
  no Databricks primary source. If true it is a direct competitor to the learned-stats loop.
- ⚠️ Reyden written in C++/Rust / "a Spark rewrite" — blogger inference, never stated.
- ⚠️ Reyden's benchmark numbers — disputed; SF1/22K-row datasets; average not P99 published.
- ⚠️ Photon's TPC-DS 100TB record **still standing** — confirmed only for Nov 2021/Feb 2022.
- ⚠️ A 2025/26 Photon vectorized-shuffle redesign — **no primary announcement exists**.
  "3–7x scan / 2–4x join" figures circulating in secondary blogs are marketing repetition.
- ⚠️ Photon stateful streaming — **roadmap** (DAIS 2026 session), docs still say unsupported.
- ⚠️ "Spark 4.0 AQE is 30% faster" — secondary blogs only. Spark 4.0 added **no** new AQE capability.
- ⚠️ Liquid Clustering "uses Hilbert curves" — primary sources say tree-based/ZCube.
- ⚠️ Z-ORDER is **not** formally deprecated — Databricks "recommends liquid clustering for new tables."
- ⚠️ IWM "fast lane / slow lane" — appears in no Databricks source; treat as fabricated.
- ⚠️ "Instant compute" is not a Databricks product name.
- The Spark join-reorder cost function from the 2017 Databricks blog (linear `w·card + (1−w)·size`)
  is **stale** — replaced by SPARK-33935/34922 with a weighted geometric mean of cost ratios.

**About Batcher:**
- Do **not** claim Batcher beats native single-node DuckDB on TPC-H. Measured 2026-07-18 at
  sf1: geomean **≥1.26x in DuckDB's favour** (optimistic — see the stability note), DuckDB
  faster on 16 of 22 (§7c).
- Do **not** quote a TPC-H ratio from a single benchmark run. q17 alone ranged 2.25x-8.70x
  across identical runs.
- Do **not** repeat that benchmarks cannot be run without the S3 corpus — they can, from
  locally generated data (§7c). That claim wrongly gated several parity items.
- Do **not** aim the `cbo.enabled=false` critique at Databricks — they flip it on. It is a
  critique of OSS Spark only.
- Do **not** claim "no operator-internal adaptation anywhere" about Databricks. True of Spark;
  **false of Photon**, which adapts kernels per batch.
- Do **not** benchmark the learned-stats claim against OSS Spark. The real bar is Databricks'
  workload-aware automatic statistics and Enzyme's history-calibrated cost model.
- Do **not** describe `kyber/rules/extra/runtime_filters/` as runtime filtering. All 14 rules are
  static rewrites over persisted statistics.
- `competitive_architecture.md` said "259 rules"; the live registry has **302** (corrected
  2026-07-18). This number drifts — verify it against the registry rather than quoting a doc.

---

## Bottom line

**Optimizer: at or above parity**, with one large hole (dynamic filter pushdown into scans).
Bushy DP join reordering, 302 rules, a calibrated cost model and a cross-query bandit are
genuinely competitive, and bushy reordering beats Catalyst's left-deep bias.

**Vectorized execution: below parity**, and the gap is coverage, not architecture. 8 of 46 `Expr`
variants JIT; no string acceleration at all; no `StringView`.

**Distributed: at or above parity structurally** — the credit-proven serviceless Flight shuffle
and four-layer fault tolerance are real advantages — **but undermined by the multi-node path
silently skipping skew and bloom handling**, and by defaults that ship capability off.

**Enterprise: below parity, blocking.** The governance rewrite mechanism is excellent and
arguably cleaner than Unity Catalog's query-time UDFs. But with no authentication, no
multi-tenancy, no UDF sandboxing and no encryption at rest, it is a well-designed policy engine
sitting on no security foundation. Fix the foundation before claiming the mechanism.
