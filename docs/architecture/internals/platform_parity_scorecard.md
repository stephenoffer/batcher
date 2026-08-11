# Platform parity scorecard: Batcher as an engine vs the commercial platforms

**Status:** audit, 2026-07-26. Internal working document, excluded from the published site.

This is the **enterprise-requirement** scorecard against the commercial platforms: Databricks,
Snowflake, SageMaker, Anyscale, and SkyPilot. It answers "would this pass an enterprise
evaluation", dimension by dimension, with an evidence citation for every cell.

It does **not** restate performance ratios. `competitive_architecture.md` is the
**execution-model** scorecard against the open-source engines (DuckDB, Polars, Spark, Flink,
Ray Data, Daft), it is the authority on where Batcher is fast and where it is not, and nothing
here contradicts it. Where this document needs a performance fact it cites that one.

## How to read the Evidence column

Every cell carries evidence or it carries `UNMEASURED`. There is no third option, and
`UNMEASURED` renders as a dash in any summary — **never as a verdict**. A blank is a claim
nobody checked, and a claim nobody checked is the thing this file exists to prevent.

| Evidence form | Means |
|---|---|
| `path:line` | Read in this repository's code at that location. |
| `tests/...` | Asserted by a test that runs in CI. |
| `benchmarks/...` | Produced by a committed benchmark, with its conditions recorded. |
| primary source | The vendor's own documentation, cited. |
| ⚠️ | Believed true, **no primary source attached**. Treat as a lead, not a fact. |
| `UNMEASURED` | Not established. Not "probably fine". |

## What is being compared

The comparison unit is the **execution layer**, not the product. Stated once, because most
disagreement about these comparisons is really disagreement about this table:

| Platform | The engine being compared | Note |
|---|---|---|
| Databricks | Photon (vectorized C++) + Catalyst/AQE | The closest true comparator. |
| Snowflake | The vectorized push-based engine behind a virtual warehouse | Closed. Nearly everything public is the 2016 SIGMOD paper or marketing. |
| Anyscale | Ray Data + RayTurbo, on the Anyscale control plane | Awkward: **Batcher runs *on* Ray.** It competes with Ray Data while consuming Ray Core. |
| SageMaker | **No engine of its own.** | It orchestrates containers. The engine is whatever you run inside — often Spark or pandas. |
| SkyPilot | **No engine of its own.** | A multi-cloud provisioner. Same. |

For the last two, the honest comparison is *the engine you would otherwise run there*, which
the existing benchmarks already cover. "Batcher is faster than SageMaker" is a category error
and appears in [Claims not to make](#claims-not-to-make).

## The blocking gaps, hoisted

These are above the matrix because a reader who stops here should still leave with the right
picture. Each would end an enterprise evaluation on its own, regardless of throughput.

1. **The trust boundary is the process, and no in-engine work changes that.** Batcher is a
   library imported into the caller's process. Phase 3 added credential verification
   (`governance/authn/`) and tenant namespacing (`config.TenantConfig`), and both are real and
   enforced — but any code inside the process can construct a `Principal`, read another
   tenant's memory, and call `numpy.zeros(1 << 40)`. Databricks and Snowflake put a network and
   a VM between tenants. Batcher's answer is one process per trust domain, stated plainly in
   `docs/user-guide/hardening.md`. **This is a structural difference, not a to-do.**
2. **No cross-node admission control.** Phase 4A bounds concurrency *per process*
   (`carbonite/policies/concurrency.py`). Databricks publishes a queue depth of 1,000 and an
   autoscaling ladder for a *warehouse*; Batcher has no coordinator, no fencing tokens, and no
   leader election, so N drivers each bound only themselves.
3. **No elasticity within a query.** A running Batcher query holds the workers it started
   with. Databricks and Snowflake resize a warehouse under load. Deliberately out of scope in
   the current plan, but it is a real column in an evaluation matrix.
4. **No encryption at rest.** Artifacts are owner-only as of Phase 2B
   (`_internal/paths.py`, `crates/bc-runtime/src/agg/spill.rs`), which is the whole of the
   at-rest protection. Both commercial platforms encrypt by default with customer-managed keys.
5. **Fault tolerance is verified in mechanism, not in timing.** The recovery machinery is real
   and now observable (Phase 1A), but no cell in the resilience matrix has a recovery *number*,
   because Ray task execution does not work in this sandbox: `ray.init` succeeds and a bare
   `@ray.remote def add(a, b)` then times out at 60 s.

## The matrix

Legend: **W** Batcher wins · **=** parity · **L** Batcher loses · **✗** cannot do it ·
**—** the platform has no engine to compare.

### 1. Fault tolerance

| Platform | Verdict | Evidence |
|---|---|---|
| Databricks | **=** in mechanism, `UNMEASURED` in recovery cost | Lineage recompute + shuffle replication exist (`dist/executors/ray_runtime/reduce.py`, `dist/shuffle_replication.py`). Recovery *is* now observable — `RECOVERY` events for `worker_lost`/`recompute`/`straggler_backup`/`backup_won`/`preempt_migrate`/`replica_retired`/`give_up` (`_internal/events.py`, `tests/unit/test_recovery_events.py`). Before Phase 1A `grep events.publish python/batcher/{dist,carbonite}` returned **nothing**: the machinery ran, and no operator could see it. |
| Databricks (spot) | **W** ⚠️ | Proactive preemption migration at stage boundaries (`carbonite/resilience/preemption.py`). Spark has no out-of-the-box equivalent per `competitive_architecture.md`. ⚠️ no primary Databricks source attached. |
| Snowflake | `UNMEASURED` | Snowflake's failure handling is not publicly documented at this level. |
| Anyscale | **=** | Both sit on Ray; recovery is Ray task retry plus the engine's own lineage. |
| SageMaker / SkyPilot | **—** | They restart a *container*. Not the same unit. |

**Scope the claim exactly.** The fault injectors kill an actor (`ray.kill`). That is **actor
death** — not spot preemption, not node loss, not a network partition, not a slow disk. Any
recovery claim inherits that scope.

**Replication is wired for every Flight shuffle.** It was once the flat aggregate reduce alone —
`replicate_shuffle_output` had one caller and hardcoded `stage=0, epoch=0` — so the combiner tree,
the path a *large* cluster takes and therefore where node loss is likeliest, fell back to full
lineage recompute along with join, sort, and window. All four now call it, and the tree's interior
levels are copied per level by `replicate_interior_outputs` rather than living on one node.

### 2. Reliability (does it return the right answer under stress?)

This is the dimension where the work in this pass moved the needle most, so it is itemized.

| Finding | Was | Now | Evidence |
|---|---|---|---|
| Shuffle hash portability | `ahash` selects its AES-NI backend at **compile time**, so a mixed-instance autoscaled cluster could split one `GROUP BY` group across reducers, **silently**. Demonstrated: same keys routed `[7,1,1,2,6,7,7,7]` under `+aes` and `[2,0,3,6,5,6,1,7]` without. | xxHash64, spec-defined and endian-defined | `crates/bc-arrow/src/hash.rs`, `crates/bc-runtime/tests/shuffle_hash_golden.rs`, `crates/bc-expr/tests/xxhash_crosscheck.rs` |
| Sketch merge portability | Same defect under a comment claiming the opposite — a wrong `approx_count_distinct` with **no runtime oracle** | Pinned value across builds | `crates/bc-sketches/src/lib.rs` |
| Deep plan IR | Uncatchable `SIGABRT` (serde recursion guard disabled); 2 MiB stack died at depth 700 | Catchable `PlanTooDeepError` | `crates/bc-ir/src/depth.rs`, `crates/bc-ir/tests/plan_depth.rs` |
| Shuffle bucket lifetime | `clear_plan` was wired end to end through Rust with **zero production call sites** — every bucket of every query leaked, in RAM and in `/dev/shm` | Evicted at query-scope exit and at `cleanup()` | `dist/fleet/eviction.py`, `tests/integration/test_shuffle_bucket_eviction.py` |
| Tenant result cache | Two tenants running the same query over the same path produced an **identical cache key**; the second was served the first's rows | Keyed by tenant + viewer | `api/executors.py`, `tests/unit/test_tenant_isolation.py` |
| Cancelling a query | Impossible. `allow_threads` holds the GIL open, so Ctrl-C was never delivered | Cooperative cancel at morsel/operator/merge-pass boundaries | `crates/bc-resource/src/cancel.rs`, `tests/integration/test_query_cancellation.py` |

Verdict vs every platform on this dimension: **`UNMEASURED` as a comparison.** Six real
silent-wrong-answer or resource-leak defects were found and fixed in one pass over one engine;
that is a statement about this codebase's maturity, not about the competitors', and nobody has
run the equivalent audit on Photon. Do not turn it into a win.

### 3. Performance

Cited, not restated. See `competitive_architecture.md` and `docs/benchmarks/`.

| Platform | Verdict | Evidence |
|---|---|---|
| Databricks / Spark | **W** distributed batch, per the OSS scorecard | `competitive_architecture.md` |
| Photon specifically | `UNMEASURED` | No Photon license, no run. **Never quote a ratio.** |
| Snowflake | `UNMEASURED` | No account, no run. |
| Anyscale / Ray Data | **W** (50–450x) | `competitive_architecture.md`, and the mechanism is structural: the data plane bypasses the Ray object store (`crates/bc-transport/src/store.rs`) |
| RayTurbo specifically | `UNMEASURED` | Closed, no license. |
| Concurrency (1 → 16 clients) | **L, and it inverts** | `BENCHMARK_RESULTS.md`: 124 QPS at 1 client, **88** at 16; p50 7.6 ms → 178 ms |

On that last row: Phase 4A adds bounded admission and fair-share width, which is the fix that
fits the diagnosis (16 queries each asking for a full-width pool on one box). **It is not
demonstrated to work.** The Python control plane is GIL-bound, so concurrent `collect()` calls
on small in-memory data serialize before they contend for cores — every grant in a live
8-thread run reported `concurrent == 1`. Needs `benchmarks/concurrency/` on a quiet box.

### 4. Scalability and elasticity

| Platform | Verdict | Evidence |
|---|---|---|
| Databricks | **L** on elasticity | Databricks publishes an autoscaling ladder and warehouse sizes to 5X-Large (`databricks_parity.md`, with primary sources). Batcher has no in-query resize. |
| Snowflake | **L** on elasticity ⚠️ | Multi-cluster warehouses. ⚠️ no primary source attached here. |
| Anyscale | **L** on cluster elasticity | Anyscale's control plane autoscales; Batcher composes Ray tasks but does not drive scaling decisions. |
| SkyPilot | **—** on engine, **L** on provisioning | SkyPilot's whole product is cross-cloud provisioning and spot management. Batcher does not provision. |
| Bounded per-node memory at scale | **W** architecturally | Mergeable algebra `partial → combine → finalize` (`crates/bc-runtime/`) plus spill; one implementation for 1 core, N cores, N machines |

### 5. Security

| Capability | Batcher | Evidence |
|---|---|---|
| Authentication | **Now real, bounded by the process** | `governance/authn/` — `ProcessIdentityVerifier`, `HmacTokenVerifier` (stdlib only), `JwtVerifier` (OIDC/JWKS, asymmetric-only by default to block algorithm confusion). `governance.require_verified_principal` makes `bt.security()` refuse an asserted principal. `tests/unit/test_authentication.py` |
| Authorization | **W** in mechanism | Row filters + column masks as a **pre-optimizer plan rewrite anchored at the leaf**, fail-closed. Unity Catalog implements these as SQL UDFs evaluated at query time, which is why time travel and CLONE do not work on filtered tables there. |
| Mandatory governance | **New** | `GovernanceConfig.mode` ∈ `off`/`advisory`/`strict`; strict refuses an ungoverned read *and* an ungovernable source rather than silently exempting it. `tests/unit/test_governance_mode.py` |
| UDF isolation | **L**, defense in depth only | `core/udf/isolation.py` closes credential inheritance (drops every `BATCHER_*`, including `BATCHER_SECRET_COMMAND`) and adds rlimits. **Covers the process path only** — a thread-path UDF *is* the engine process. Not a sandbox. |
| Artifacts at rest | **=** on permissions, **L** on encryption | 0700 dirs / 0600 files by construction (`_internal/paths.py`, `crates/bc-runtime/src/agg/spill.rs`); `tests/integration/test_artifact_permissions.py`. No encryption. |
| Statistics leakage | **Fixed** | A governed column's persisted bloom filter is a **membership oracle** — real values tested PRESENT, fakes absent. Value-bearing stats are now dropped for masked/invisible columns. `api/source_stats.py`, `tests/unit/test_governed_stats_redaction.py` |
| Tenant isolation | **L** structurally | Namespaced where the engine owns the boundary (result cache, learned stats). Cooperating workloads, not adversaries. |

### 6. Multi-tenancy and workload management

| Platform | Verdict | Evidence |
|---|---|---|
| Databricks | **L** | Batcher now has tenant scoping (`config.TenantConfig`, `bt.tenant(id)`) and per-process admission. It has no warehouse manager: no workload groups, no priority scheduling, no chargeback, no queue visible across nodes. |
| Snowflake | **L** ⚠️ | Resource monitors and multi-cluster warehouses. ⚠️ no primary source attached. |
| Anyscale | **=** ⚠️ | Both delegate isolation to the cluster scheduler. |

### 7. Observability and operability

| Capability | Batcher | Evidence |
|---|---|---|
| Query plans | **=** | `EXPLAIN` / `EXPLAIN ANALYZE`, JSON profiles |
| Traces | **=** | OpenTelemetry |
| Metrics | **=** | `observe/metrics.py::prometheus_text()`, now including `batcher_recovery_total{event=...}`. *This settles a ledger contradiction: `databricks_parity.md` said "traces only". It was wrong; a Prometheus exposition exists.* |
| Recovery visibility | **Fixed** | Phase 1A, above |
| Cancelling a runaway query | **Fixed** | `bt.cancel_query` / `bt.running_queries`; Ctrl-C works |
| Query history as a table | **Missing** | Planned (4C). An event log exists; `ActivityStore` is a 100-entry debugging window by design, not an archive. |

### 8. Governance and audit

| Capability | Batcher | Evidence |
|---|---|---|
| Column lineage | **=** | Real, and correctly over-approximates on opaque `map_batches` |
| Row filters / column masks | **W** in mechanism | Plan rewrite, above |
| Write privileges | **Missing** | Grants cover SELECT. **Writes are ungoverned.** |
| Revoke | **Missing** | — |
| Path aliasing | **L** | Policies key on exact path strings, so `s3a://` vs `s3://` evades one |
| Durable audit sink | **Missing** | `GovernanceConfig.audit_path` is defined; the append-only JSONL sink is not written |

## SQL surface: 76 of 99 TPC-DS queries plan

Not one of the eight dimensions, but it belongs here because "can it express the workload"
precedes every other question in an evaluation, and because it was an opinion until this pass.

`benchmarks/internals/tpcds_coverage.py` runs all 99 official texts through parse-and-plan
against the 24 official schemas — both sourced from DuckDB's `tpcds` extension, not written
from memory. **76 plan.** The 23 that do not are grouped by cause in
`BENCHMARK_RESULTS.md`; the two that matter:

- **Decimal support is the single biggest blocker** (6 queries), and it masquerades as window
  and set-operation gaps. One fix closes what looks like four roadmap items.
- **Four queries fail on a synthesized join key referring to an out-of-scope column**
  (`__jk_l0` → `w_warehouse_sk`). That reads as a translator defect rather than a missing
  feature, and it is not yet filed as one.

This also corrects `benchmarks/suites/standard/tpcds.py`, which says expanding past its 7
queries is "mechanical once a query's tables are added". All 24 tables are registered in the
coverage run and 23 queries still fail — every one on the SQL surface, none on tables.

Evidence: `benchmarks/internals/tpcds_coverage.py`, `BENCHMARK_RESULTS.md`. Comparison to
Databricks/Snowflake TPC-DS coverage: `UNMEASURED` — both run all 99, but the meaningful
comparison is at execution and correctness, which this does not measure.

## Per-platform verdict

**Databricks.** The closest comparator and the fairest fight. Batcher wins on distributed batch
throughput and on the mergeable-algebra/Flight architecture; it loses on elasticity, on the
warehouse-manager surface, and on being a managed product with a trust boundary. Photon itself
is `UNMEASURED` and must stay that way until someone runs it.

**Snowflake.** Cannot be scored honestly on performance at all. On enterprise surface it wins on
everything a managed multi-tenant service gets for free. The interesting comparison is
architectural, and it needs primary citations this document does not yet have.

**Anyscale.** Batcher beats Ray Data by a wide, structurally-explained margin, and then runs on
Ray Core underneath, so it inherits Ray's scheduling behavior including the pitfalls. RayTurbo
is closed and `UNMEASURED`. Say "beats Ray Data", never "beats Anyscale".

**SageMaker and SkyPilot.** Not engines. The only shared axis is elasticity and spot handling,
where Batcher has something real to say (proactive preemption migration at stage boundaries)
and something real to concede (its task granularity is roughly a node).

## Claims not to make

Binding. Each of these is either unsupported or a category error.

1. Any Batcher-vs-Photon, -Snowflake, -RayTurbo **timing or ratio**. None can be run here. The
   measured-vs-published form, labelled weak, is for direction only and never a ratio.
2. "Faster than SageMaker / SkyPilot." Category error — they ship no engine.
3. A QPS number missing any of: client count, loop mode (open vs closed), shape mode (repeated
   vs rotating), engine profile, machine fingerprint, load average.
4. "Fault tolerant" without the scope: **worker actor death**, at N workers, on a real cluster.
5. That Phase 4A **fixed** the concurrency inversion. It is a fix that fits the diagnosis and
   is not demonstrated.
6. That the cancellation poll is free. Measured at +2.3% / +1.1% / **-0.4%** across three
   shapes — a negative arm means it is under the noise floor on a box at load 16.9. "No
   regression demonstrated" is the claim; see `BENCHMARK_RESULTS.md`.
7. Anything `competitive_architecture.md` retires: the AQE-superiority claim, "beats DuckDB on
   every TPC-H query" (true only vs `duckdb_arrow`; against DuckDB's native store DuckDB is
   faster on 16 of 21), the 57x in-memory-kernels table, and the `bc-adapt` crate.
8. That Batcher is a trust boundary. It is a library in your process.

## What would move these cells

In rough order of evaluation impact:

1. Run the resilience matrix on a real multi-node cluster. Every fault-tolerance cell is
   currently mechanism-only.
2. Run `benchmarks/concurrency/` on a quiet box, before and after Phase 4A.
3. Wire replication beyond the flat aggregate reduce (Phase 1E), so node loss on a large
   cluster refetches instead of recomputing.
4. Write privileges, revoke, and the durable audit sink — the three governance cells that read
   "Missing" and that an auditor asks about first.
5. Primary citations for every ⚠️ in this file.
6. Decimal support, which unblocks 6 TPC-DS queries and is the cheapest coverage win on the
   list; then the `__jk_` translator defect, which is 4 more.

## See also

- `competitive_architecture.md` — the execution-model scorecard vs the OSS engines. The
  authority on performance; never contradicted here.
- `databricks_parity.md` — the deep Databricks ledger, with the per-phase records of this pass.
- `benchmarks/BENCHMARK_RESULTS.md` — every number cited above, with its conditions.
