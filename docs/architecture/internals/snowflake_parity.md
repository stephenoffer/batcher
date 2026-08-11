# Snowflake parity: the comparison that cannot be measured

**Status:** audit, 2026-07-26. Internal working document, excluded from the published site.
The hub is `platform_parity_scorecard.md`; this is the Snowflake column in depth.

## Read this before using anything below

**Snowflake is the hardest platform in this set to write about honestly.** It is closed source,
its engine has no public benchmark you can reproduce, and almost everything findable about its
internals is either the 2016 SIGMOD paper *The Snowflake Elastic Data Warehouse* — now a decade
old and describing a system that has since changed — or marketing.

So this file has a stricter rule than its siblings: **every claim about Snowflake carries a
primary citation or a ⚠️.** A ⚠️ means "believed true, no source attached" and must be read as a
lead to verify, not as a fact to build on. There are a lot of them, deliberately. Filling them in
is the work; papering over them with confident prose is the failure mode this whole ledger set
exists to prevent.

**No performance comparison exists or can exist here.** There is no Snowflake account, no run,
and no reproducible published number to compare against. Any Batcher-vs-Snowflake ratio is
fabricated. That is binding.

## What is being compared

Snowflake's execution layer: the vectorized, push-based engine that runs inside a virtual
warehouse over its own columnar format in object storage.

Two things follow immediately, and they frame everything else:

1. **Snowflake is a storage-and-compute product; Batcher is an engine.** `CLAUDE.md` scopes
   Batcher explicitly as an engine, not a storage solution. A comparison that credits Snowflake
   for micro-partition pruning over its proprietary format is comparing different things.
2. **The parts of Snowflake an enterprise buys are mostly not the engine.** Managed multi-tenant
   isolation, elastic warehouses, resource monitors, time travel, replication, and a support
   contract are product properties. Batcher has no answer to any of them and is not trying to.

## Architecture: what can be said

| Aspect | Snowflake | Batcher | Confidence |
|---|---|---|---|
| Execution model | Vectorized, push-based | Vectorized, morsel-driven pull (streaming) with a materializing fallback (`crates/bc-interp/`) | Snowflake: SIGMOD 2016 |
| Columnar format | Proprietary micro-partitions with per-column metadata | **Arrow only**, by invariant | Batcher: `CLAUDE.md` invariant 3 |
| Storage/compute separation | The defining property | Batcher reads open formats; storage is not its concern | SIGMOD 2016 |
| Optimizer | Cost-based, statistics from micro-partition metadata | Kyber: rules + cost model + **sketch-backed learned stats across runs** (`kyber/learning.py`) | Snowflake ⚠️ |
| Adaptivity | ⚠️ believed to re-plan | Stage-boundary re-optimization on measured cardinalities, **off below 5M input rows per pipeline breaker** | Batcher: `api/adaptive/` |
| Elasticity | Multi-cluster warehouses, resize under load | **None in-query** | Snowflake ⚠️ |
| Result cache | ⚠️ believed 24 h, query-text keyed | In-memory, keyed by plan signature + tenant + viewer (`api/executors.py`) | Batcher: code |

## Where Batcher has a genuine architectural argument

Restricted to what is evidenced in this repository:

- **Open format, no lock-in.** Arrow is the only columnar contract, end to end and across the
  FFI boundary, zero-copy. Snowflake's format is proprietary and its performance advantages are
  partly *from* that. This is a real trade, not a straightforward win: `competitive_architecture.md`
  records Batcher losing to DuckDB's compressed native store on sort/distinct shapes for exactly
  the same reason, and calls that gap **structural** because invariant 3 forbids the fix.
- **One semantics for one core and a cluster.** `partial → combine → finalize`. There is no
  separate distributed operator with its own behavior, which is why single-node and distributed
  results are required to be identical.
- **A cross-query learned loop.** Cost coefficients calibrated from measured `op_stats`, a UCB1
  bandit over join strategies, learned partition counts and hot keys. ⚠️ Snowflake's optimizer
  learning, if any, is undocumented.
- **It is embeddable.** Batcher is a library you import. Snowflake is a service you send SQL to.
  For an ML pipeline that wants the engine *inside* the training process, that is the whole
  argument.

## Where Snowflake wins, and it is not close

Every one of these is a product property Batcher structurally lacks:

| Capability | Note |
|---|---|
| Multi-tenant isolation | Snowflake puts a VM and a network between tenants. Batcher's boundary is the process — see the hub's blocking gap 1. |
| Elasticity | ⚠️ Multi-cluster warehouses scale under load; Batcher holds the workers a query started with. |
| Encryption at rest with customer-managed keys | ⚠️ Batcher makes artifacts owner-only and stops there (`_internal/paths.py`). |
| Time travel, cloning, replication | Storage features. Out of scope by charter. |
| Workload management, resource monitors, chargeback | ⚠️ Batcher has per-process admission (`carbonite/policies/concurrency.py`) and no cross-node coordinator. |
| Auditability | ⚠️ Snowflake has a durable query history; Batcher's `GovernanceConfig.audit_path` is declared and the sink is not yet written. |
| Governed writes | Batcher's grants cover SELECT. **Writes are ungoverned.** |

## The one honest headline

> Batcher is an **embeddable open-format engine with a learned optimizer**. Snowflake is a
> **managed multi-tenant warehouse**. They overlap on "runs analytical SQL over columnar data"
> and diverge on essentially everything an enterprise evaluation scores. Where they can be
> compared — the execution layer — there is no measurement, and there will not be one from this
> repository.

## Claims not to make

1. Any Batcher-vs-Snowflake **timing, ratio, or TPC-H/TPC-DS comparison**. There is no run.
2. Any statement about Snowflake's internals sourced from this file's ⚠️ rows without first
   attaching a primary citation.
3. That Snowflake's engine is "just" vectorized push and therefore beatable. The 2016 paper is a
   decade old; the engine has changed and the paper does not describe today's system.
4. That open format is unambiguously better. `competitive_architecture.md` records the same
   property costing Batcher measurably against DuckDB's native store.

## What would move this file

1. Attach a primary citation to every ⚠️, from Snowflake's own documentation.
2. Find and read any post-2016 primary material on the engine.
3. If a Snowflake account ever becomes available: TPC-H at a fixed scale factor, warehouse size
   recorded, against Batcher on comparable hardware — knowing the result would measure a managed
   service including provisioning, not two engines.

## See also

- `platform_parity_scorecard.md` — the hub, including the layer-mapping table.
- `competitive_architecture.md` — where the open-format trade is measured against DuckDB.
