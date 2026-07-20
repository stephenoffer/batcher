# Ray pitfall parity: what Batcher must avoid out of the box

**Status:** audit, 2026-07-19. Source corpus is the Anyscale field-engineering
optimization guides (`../optimization-guides`, ~200 docs distilled from customer
engagements). Every Batcher claim below was checked against code, not documentation.

This document exists because that corpus is, read a certain way, **a specification written
in the negative**. It is several hundred pages of things a user must know, measure, and
hand-tune to stop Ray from performing badly. Each one is a question Batcher should be able
to answer with "you don't have to — it does that."

The bar is deliberately harsh: a pitfall counts as **avoided** only if a user who configures
*nothing* is safe. A capability behind a default-off flag does not count — that is the
failure mode `databricks_parity.md` names as "the single largest discrepancy between what
this codebase contains and what a user actually gets."

## Verdict

**Batcher structurally avoids the largest and most damaging class of Ray pitfalls — the
object-store family — because it does not have an object store on the data path.** That is
not a tuning win, it is an architectural one: roughly 40% of the guides' failure catalog
(spill cascades, plasma deadlocks, the 200 GB cap, `/dev/shm` sizing, fragmentation,
serialization cliffs) is unreachable from Batcher's design.

The AI-workload story was **materially weaker than the architecture suggested until this
pass**, for one reason: the CPU→GPU overlap that keeps a GPU fed was implemented, tested,
and **shipped disabled**. It is now on by default. See [What changed](#what-changed-in-this-pass).

Where Batcher is genuinely behind is **straggler handling and shuffle fault tolerance**, and
the honest reason is task granularity: Batcher's unit is ≈ a node, so it has fewer of Ray's
scheduler-saturation problems and correspondingly less of Spark's dynamic load balancing.

## The scorecard

Legend: **A** avoided structurally (cannot occur) · **D** handled by a default (occurs, but
Batcher fixes it with no user action) · **P** partial · **G** genuine gap.

| # | Ray pitfall | Batcher | Why |
|---|---|---|---|
| A1–A14 | Object-store sizing, spill cascade, plasma deadlock, 200 GB cap, `/dev/shm`, fragmentation, `ray.put()` fate-sharing | **A** | Bulk Arrow never enters the Ray object store. Only `(addr, ticket)` strings transit Ray; data moves over `bc-transport` Arrow Flight with **credit-based flow control** whose bound is a proven in-flight gauge (`crates/bc-transport/src/store.rs:16-62`). There is no plasma store to size, saturate, fragment, or deadlock. |
| B1–B9 | Serialization: N-copy args, non-zero-copy torch tensors, the 100 KB inline cliff, closure capture | **A** | Arrow `RecordBatch` is the only columnar contract, zero-copy across FFI via the Arrow C Data Interface. There is no by-value task-argument path for bulk data, so there is no threshold to straddle. |
| C1–C3 | Task-granularity tax; Raylet saturation at ~1K tasks/node/s; the "100 ms rule" the user must compute themselves | **A/P** | Batcher's scheduling unit is a partition-per-node, not a task-per-item, so the submission rates that saturate Ray's Raylet are never reached. **The same coarseness is the cause of gap G1 below** — it is a trade, not a free win. |
| C6 | GCS actor cliff at ~7,000 actors (400 s+ scheduling stalls) | **A** | Actor count is bounded by workers × stages, not by data volume. |
| C7 | Actor creation 1–5 s; 30–120 s with GPU model load; pools respawned per execution | **D** | `warm_inference_pools: bool = True` (`config/config.py:1098`) keeps GPU/load-once pools warm **across `collect()` calls in a session**, keyed by pipeline identity. The model loads once per session. Ray Data respawns the pool per execution — the guides measure this as "~20x slower on the first batch." |
| D5 | Fractional GPUs give **no VRAM isolation**; co-resident actors silently CUDA-OOM each other | **P** | `ml/gpu.py::max_actors_per_gpu` implements the guides' packing formula (`vram / (model×1.25 + cuda_context)`) as engine logic rather than a table the user reads. It sizes correctly; it still cannot *enforce* — no engine can without MIG. |
| E1–E3 | OOM monitor kill loops; the 250 ms allocate-gradually race | **A** | Bounded memory is structural, not reactive: the streaming executor (`memory.streaming: bool = True`, `config/config.py:158`) materializes only at breakers, so peak memory is breaker state plus one morsel per worker — **constant in input size** (measured 3.4 MB at 1M rows, 3.3 MB at 4M, where the materializing path grew 198 → 794 MB). Nothing polls for an OOM that is not coming. |
| G1, G2, G9 | Autoscaler blind to dependent tasks and hash-shuffle ops; scales unboundedly | **P** | Batcher plans the whole DAG up front, so downstream demand is known rather than inferred from ready tasks. Per-stage inference autoscaling does not exist yet (see G3 below). |
| H1, H2 | Heterogeneous clusters: global object-store budget overestimates small nodes; "do NOT use `auto_select_worker_config` with Ray Data" | **A** | Memory is governed per-worker by Carbonite envelopes, not by a cluster-global fraction. There is no global budget to mis-apportion. |
| J1–J3 | Spilling invisible in dashboards; Ray Data metrics attributed entirely to the head-node IP, making per-worker analysis **impossible** | **P** | `observe/` consumes a per-subsystem event bus with real per-worker attribution. Not audited against the guides' specific gap list in this pass — flagged as follow-up. |
| K2 | Fixed batch sizes create stragglers; user must hand-write `estimate_processing_time(item)` | **D** | `ml/autobatch.py::ThroughputController` hill-climbs batch size against measured throughput under a VRAM cap, and **persists the settled plateau across runs** (`learned_batch_size`/`record_batch_size`), so a recurring job starts near last run's optimum. Ray has no batch-size auto-tuning at all; the guides name `batch_size=None` as the #1 OOM cause. |
| §2 (LLM) | `concurrency` defaults to **1** — one vLLM engine regardless of cluster size, "the most common misconfiguration on multi-GPU clusters" | **D** | `ml/gpu.py::resolve_num_workers` derives replicas as `total_GPUs / per-actor num_gpus` — the module comment is explicit: "never one engine idling a multi-GPU." The knob the guides teach you to compute is computed. |
| §1 (BI) | **"CPU preprocessing starving GPU is #1 cause"** of GPU underutilization | **D** | `stream_inference` now defaults **True** — see below. This was a **G** before this pass. |
| K1 | `batch_size=None` is the **default** and is "the single most common cause of heap OOM" — one block becomes one batch, and blocks can be hundreds of MB | **D** | Batch size is a control loop, not a default constant (`ml/autobatch.py`), and the morsel (16,384 rows, `bc-arrow`) bounds the unit independently of file or block size. |
| A1, A2, M3 | **Block count — not actor count — governs parallelism.** `ActorPoolStrategy(size=8)` on a one-block dataset runs sequentially, and "no warning is emitted." Too few blocks also means no autoscaling demand signal | **A** | Parallelism comes from morselization inside the engine, not from the physical layout of the input. A single large input file still fans out across cores. |
| B2 | Ray Data tracks the object-store budget **globally** but spilling triggers **per-node** — "the fundamental issue behind most unexpected spilling and OOM problems in multi-node clusters" | **A** | No global budget and no plasma store. Carbonite governs per-worker envelopes; spill is a local decision against a local bound. |
| F19 | `preserve_order=True` "can halve throughput — a **silent performance killer**"; it is a direct straggler amplifier | **A** | Order is a property of the plan (an explicit `sort`), not a scheduling mode, so there is no ordering flag that silently serializes the pipeline. |
| L2 | GPU actor-pool autoscaling **reloads the model on every scale-up** (30–120 s for large LLMs), so the guides advise pinning a fixed pool and giving up autoscaling | **D** | `warm_inference_pools` keeps pools warm across `collect()`s, so scaling does not imply reloading. This is why Batcher can have both — see G3 for the autoscaling half, which is still missing. |
| H13 | `max_errored_blocks = 0` by default — "a multi-hour batch job fails at 99% on one corrupt file, discarding all progress" | **D** | The distributed scan carries a broken-record policy in each partition manifest (`distributed.on_read_error`), so tolerance travels with the data rather than through global config. Default is still fail-fast, which is right for a small trusted input and wrong for a data lake — worth revisiting per-source. |

## The capability gaps Batcher closes by construction

The guides contain a section (`foundations/data/mental-model.md`,
`transforms/shuffles-and-joins.md`) listing what Ray Data **architecturally cannot do**, with
the recommended workaround usually being "use Spark, Daft, or DuckDB instead." That list is
the clearest statement of the competitive opening, and Batcher already holds most of it.

| Ray Data's stated boundary | Their workaround | Batcher |
|---|---|---|
| **"No external shuffle service — no Celeborn or Arrow Flight Shuffle equivalent for petabyte-scale shuffles"** | Daft on Ray, or Spark + Celeborn | `bc-transport` **is** a credit-controlled Arrow Flight shuffle. This is the single most direct hit on the list. |
| **"No SQL interface"** — a `map_sql` PR using DuckDB "was rejected as unsustainable" | DuckDB/Spark first, then feed Parquet to Ray Data | `_sql/` parser + `bt.sql`/`ds.sql`, differential-tested against DuckDB. |
| **"Ray Data passes 0 of 99 TPC-DS queries."** No join ordering, no bushy join optimization, no window functions | Daft for analytical queries | Kyber: 302 rules, bushy DP join ordering, window functions; TPC-H/TPC-DS/ClickBench suites in `benchmarks/suites/standard/`. |
| **"No auto-tuning. `concurrency`, `batch_size`, `num_partitions` all require manual tuning. Users report 2 weeks to optimize a single job."** | `ds.stats()` + iterate | The whole Kyber/Carbonite premise: cost-based plan choice, sketch-backed cardinality, learned partition counts, throughput autobatching — plus a cross-query bandit loop that improves the plan the more a query runs. |
| **Two-pass aggregation must be hand-written** — no combiner pushdown (`map_batches(partial)` then `groupby().map_groups(combine)`) | Write both passes yourself | `partial → combine → finalize` in `bc-runtime` **is** the only implementation; the combiner is not an optimization you opt into, it is the operator. |
| **"Key salting does not work in Ray Data — salting distributes identical keys across partitions, producing incorrect join results"** | Broadcast, or pre-filter hot keys and union | `skew_join_salt` replicates the build side to every salt bucket, so the join stays correct. Hot keys are found with Misra-Gries rather than by the user hand-plotting a histogram. Still **opt-in** — see G1. |
| **Skew detection is entirely manual** — four hand-run procedures, and thresholds the guides admit are "recommended heuristics, not Ray-defined" | Hand-plot the key distribution | Sketch-backed statistics (HLL/KLL/Count-Min) feed Kyber directly. |
| **No streaming sources**; "architecture is fundamentally batch-oriented" | Kafka → storage → `read_parquet` | Native Kafka/Kinesis/Pulsar sources with watermarks and checkpointing — though see `competitive_architecture.md` ceiling #4 for how far this is from Flink. |
| **No caching**; `ds.count()` twice runs the pipeline twice | Explicit `.materialize()` and hold the ref | Lazy plan with plan-level caching (`Dataset.sql()` plans and `plan_signature` are memoized). |
| **Predicate/projection pushdown only when syntactically adjacent to the read**; lambda filters never push down | Hand-place `filter()` immediately after `read_parquet` | Kyber pushdown is a plan rewrite over the whole tree, and `Expr` is introspectable by construction — there is no opaque-lambda path in the hot path to defeat it. |
| **No pipeline resumption at all** — "restarting the driver re-executes the plan from the source" for a multi-hour, 1B-item job | Three hand-built idempotent-write patterns | Atomic manifests + resume on the write path (`ds.write`). |

Two things this table should **not** be read as saying. First, it is a comparison of
*capability surface*, not of measured performance on every shape — `competitive_architecture.md`
holds the benchmark verdicts, including the ones Batcher loses. Second, several of these are
Batcher wins that only pay off if the relevant default is on, which is exactly the failure this
audit was written to catch.

## What changed in this pass

**`distributed.stream_inference` now defaults to `True`.**

`dist/streaming/pipeline.py` splits a linear `map_batches` chain at its resource-class
boundary into separate CPU-producer and GPU-consumer actor pools, hands morsels between them
over credit-bounded Arrow Flight, and overlaps them — the GPU runs morsel *k* while the CPU
prepares *k+1*. It was complete, tested, and gated behind `stream_inference: bool = False`.

With it off, **the default path ran the entire CPU→GPU chain inside one actor**: the GPU
actor held a whole GPU while it decoded JPEGs. That is precisely the shape the guides name as
the #1 GPU-utilization bug, and the reason their entire CPU:GPU-node-ratio chapter exists
(recommended ratios run 2:1 for image preprocessing to **4–5:1 for video decode**). Batcher
contained the fix and did not apply it.

Flipping the default is safe for three reasons that were checked, not assumed:

1. **The result is unchanged.** Every stage runs the identical sub-plan through
   `core.execute_with_udfs`; only the scheduling overlaps. Pinned by
   `test_streaming_pipeline_equals_single_node`, the three-stage
   (CPU→GPU→postprocess) variant, and a direct comparison against the non-overlapped map.
2. **Non-qualifying plans are untouched.** The dispatcher (`dist/executor.py`) only takes
   this path when `split_at_first_pool_boundary(plan)` finds a resource-class boundary;
   otherwise it falls back to the embarrassingly-parallel map. A homogeneous CPU-only
   pipeline sees no change.
3. **Memory stays bounded.** Two nested credit windows — the Flight transfer window and a
   production window bounding how far a producer may run ahead — cap a producer's resident
   output at `credits` morsels regardless of partition size.
   `test_production_window_bounds_producer_memory` asserts the bound directly.

The equivalence test that compared the two paths had to be repaired as part of this: it took
its non-overlapped baseline from the *default* config, so flipping the default would have
made it compare the streamed path against itself and assert nothing. It now pins
`stream_inference=False` explicitly for the baseline. `test_stream_inference_is_on_by_default`
pins the new default, because a silent revert would cost every batch-inference pipeline its
GPU utilization with nothing turning red.

## Genuine gaps

Ranked by how much they cost on the AI workloads the guides describe. G1's straggler half
and G2 were closed in this pass; G6 was found while working them. G3–G5 remain open, and
Batcher does **not** win every shape today — `competitive_architecture.md` holds the
benchmark verdicts, including the losses.

### G1. Straggler handling — speculation is now on; task granularity is still ≈ node

Batcher's coarse scheduling unit avoids Ray's Raylet-saturation family (C1–C3) but inherits
the opposite problem: **a straggler's entire partition must be redone**, and skew cannot be
diluted by over-partitioning the way Spark does with 10k–100k tasks per stage. That half is
unchanged and is the real remaining work.

**Closed this pass:** `speculation_max_backups` now defaults to **1**. Ray Data has *no*
straggler mitigation of any kind — no speculative execution, no task re-launch — so its
guides can only tell users to detect skew by hand and re-partition. One backup catches the
single worst straggler at a barrier, gated twice (a task must exceed
`speculation_straggler_factor` × the median finished time *and* `speculation_min_finished_frac`
of tasks must already be done), with the factor additionally *learned* per operator family
from measured task-time variance, so a uniformly-finishing stage raises its own bar and
effectively opts out. Backups are result-identical because shuffle tasks are deterministic.

This matters specifically for AI workloads because the guides identify **variable item cost
as the dominant straggler source** — one 500-page PDF among tweets, one long video among
clips, uneven prompt lengths in LLM batch.

**Deliberately still off: `skew_join_salt`.** Do not flip it as "another default-off
capability." See G6 — there is a live correctness hazard behind it, and turning it on would
ship a silent wrong answer.

### G2. Shuffle replication — **wired for the flat reduce** (was: not wired at all)

`shuffle_replication` had an implemented assignment function
(`carbonite/resilience/replication.py::assign_replica_hosts`), an implemented worker method
(`dist/flight_worker.py::replicate_buckets`), and a `replicas=` parameter threaded through
every reducer — with **zero call sites** for any of it. The `spot` profile sets it to `2`,
so a user selecting the profile *designed for preemption* believed they had re-fetch
recovery and silently got recompute.

The driver-side orchestration now exists (`dist/shuffle_replication.py::replicate_shuffle_output`,
called from the aggregate map barrier): it resolves each source's primary from the address it
actually published on, assigns off-node hosts, calls `replicate_buckets`, and passes the acked
addresses into the reducers. `tests/integration/test_shuffle_replication.py` pins both that
the result is correct under worker loss **and** that the recompute path is never entered —
with a control test proving the same kill *does* recompute when replication is off, so the
assertion cannot go quiet.

Two invariants make the fallback safe, and both are load-bearing:
a replica is advertised only once its all-or-nothing ack is in hand, and a source's replicas
are **retired when it is recomputed** — a stale replica holds the old epoch's ticket, and an
unregistered ticket reads back as an *empty bucket rather than an error*, so falling back to
one would silently drop that mapper's rows.

**Still open:** a wide shuffle (`workers > fan_in`) reduces through the combiner tree, which
does not thread replicas yet and still degrades to recompute.

### G6. Skew salting could silently split a group — guard added, and it must stay

Found while auditing G1. `_distributed_join_aggregate` fuses a post-join aggregate into the
join's reducer and finalizes each bucket locally, which is correct *only* because
co-partitioning by the join key puts every group in exactly one bucket. Salting deliberately
spreads a hot key across buckets — so each salted reducer would finalize a **partial group**
and the union would carry several half-summed rows for the hot key. No error; a wrong answer.

The reason this is not merely theoretical: **salting engages on the default path.**
`skew_join_salt` defaults to 0, but a measured hot key turns it on anyway
(`if hot and salt <= 0: salt = DEFAULT_LEARNED_SALT`). The eligibility predicate is now the
named, unit-tested `dist/executors/join.py::salting_is_safe`, which refuses to salt whenever
the reducer finalizes.

Honest scope: the guard is correct by construction and unit-tested, but it is **not** pinned
by an end-to-end distributed test. Reaching the hazard live requires the disk transport AND a
fusable join-aggregate AND a detected hot key to coincide; `resolve_transport` picks Flight on
any multi-node cluster, and `databricks_parity.md` records that Flight join behaviour "cannot
be validated in a single-node dev environment." An integration test written here passes with
or without the guard, which is worse than none — so the invariant is pinned where it can
actually be checked, and this paragraph is the note that it deserves a cluster test.

### G3. Single split point, no per-stage autoscaling

`stream_inference` splits at exactly *one* resource-class boundary. A three-stage pipeline
works (the third stage rides in the consumer), but there is no N-stage topology and no
per-stage pool autoscaling. The guides' RAG indexing pipeline is genuinely four stages with
three different resource classes (`extract` CPU-heavy → `chunk` CPU-light → `embed` GPU →
`write`), and reports **~86% cost reduction and ~69% faster** end-to-end from getting that
placement right versus an all-GPU cluster.

### G4. No variable-shape tensor type; no GPU residency across operators

Ray Data and Daft both have a variable-shape tensor type; Batcher does not. Mixed-resolution
image batches are the common case in multimodal preprocessing, and the guides log repeated
Ray failures there too (`ArrowVariableShapedTensorArray`, ray#49883/#50229) — so this is a
shared weakness rather than a Batcher-specific one, but Batcher does not currently win it.

Separately, data round-trips to host memory between operators; it never stays resident on the
GPU. Ray has the same limitation (its GPU object store is RFC-only, ray#51173).

### G5. Small-file / metadata cost — **measured, and Batcher loses**

The guides treat this as first-order, not an edge case: **file listing can be ≥50% of total
runtime**, metadata fetch takes **5–25 minutes for 500K–1.5M files**, and thousands of small
Parquet files measured **~5x slower than Spark**. This pass benchmarked it rather than
leaving it asserted.

`read.parquet(glob) → group_by → agg`, 3 reps, min, local NVMe:

| Shape | Batcher | DuckDB | Polars | vs DuckDB | vs Polars |
|---|---|---|---|---|---|
| 2,000 files × 200 rows (2.2 KiB/file) | 821 ms | 125 ms | 261 ms | **6.6x** | **3.2x** |
| 20 files × 20,000 rows (46 KiB/file) | 39 ms | 4 ms | 9 ms | **9.6x** | **4.5x** |

Two separate problems, and the second is the surprising one:

1. **A per-file tax.** ~0.39 ms/file against DuckDB's ~0.06 ms/file.
2. **A fixed per-query overhead** that dominates the small case — the ratio is *worse* on 20
   large files than on 2,000 small ones. At 400K rows Batcher is 9.6x DuckDB, which sits
   badly beside the "wins below ~10M rows" claim; that claim is measured on in-memory
   sources, and it does not survive contact with a file-read + aggregate pipeline.

**Fix shipped: `files_version` no longer thread-pools local stats.** It fanned every file's
stat across a 64-thread pool, justified as latency-hiding. That is right for an object store
and badly wrong locally, where a stat is a ~6 µs syscall and the dispatch dwarfs it. Measured
on 2,000 files: raw `os.stat` 12.1 ms, serial `file_identity` 15.4 ms, **pooled 194.1 ms** — a
12.6x penalty for "parallelism". It now runs serially when the path is local and keeps the
pool for remote schemes, taking `files_version` **194 ms → 18 ms**. The digest is proven
byte-identical to the pooled implementation, and an A/B of both stale-metadata tests is
unchanged with the branch on or off — this only moves where the work runs.

That leaves the native Parquet read as the top cost in the profile, which is where it
belongs. Two of the three per-file stats remain, dispatched through `io/_concurrent.py::read_each_file`
by the footer readers; that helper is shared with genuine footer *parsing*, which may release
the GIL, so the same local/remote split needs its own measurement rather than being assumed.

**Where the rest of the per-file tax lives, and a fix that was tried and reverted.** Profiling the
2,000-file read put `io/stats/file_identity.py::_stat` at the top by self time — ~6,000 calls
per query (**3 stats per file**), each dispatched as its own thread-pool task, costing more
than the Parquet read itself. The cause is structural: `file_identity` documents a fast path
where the directory listing supplies size/mtime for free, `expand`'s *directory* branch
populates that cache — and `expand`'s **glob** branch returns early and throws the same
information away. A glob is the ordinary way to name a many-file relation, so the documented
fast path is the one that never fires.

Recording it in `_glob` takes **821 ms → 513 ms** (DuckDB 6.6x → 3.8x, Polars 3.2x → 1.8x).
It was reverted, because it is also **wrong**: those entries outlive the listing on a
long-lived filesystem object, so a file overwritten afterwards still reports its old
`(size, mtime)` and `file_identity` calls it unchanged — reintroducing precisely the stale-
metadata hit that module exists to prevent. `test_iceberg_count_is_not_answered_from_a_stale_summary`
fails with the loop and passes without it, in isolation.

A correct version scopes the cache to the listing's lifetime — a generation stamp
`file_identity` checks — rather than just moving the loop. That is the open work, and it is
worth doing: it is the largest single measured control-plane cost on the shape multimodal
corpora actually have. The rationale is duplicated at the call site in `io/_backend.py` so the
next person does not re-attempt the naive version.

## How to read this against the competitive scorecard

`competitive_architecture.md` is the authority on where Batcher stands versus DuckDB, Spark,
Polars, Flink, and Ray Data, and its ceilings are unchanged by this document except for
ceiling #5, which this pass partially closed. This file is narrower: it asks only whether the
specific failures Ray users hit in the field are reachable in Batcher's design.

The two documents agree on the important thing, and it is worth stating plainly rather than
letting the table above read as a victory lap: **Batcher's advantage over Ray Data on AI data
workloads is real and architectural, and it was substantially unavailable to users by
default.** The pattern that produced that — real code behind a default-off flag — is the one
to keep hunting. Each remaining default in `databricks_parity.md`'s list deserves the same
question this pass asked of `stream_inference`: *is the default-off protecting against a real
risk, or only against the absence of a benchmark?*
