# Ray pitfall parity: what Batcher must avoid out of the box

**Status:** audit, 2026-07-19; extended and re-verified 2026-07-27 (per-stage ML
profiling, the silent-failure guard family, and the complete 105-pattern scorecard).
Source corpus is the Anyscale field-engineering optimization guides (`../optimization-guides`, ~200 docs distilled from customer
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
not a tuning win, it is an architectural one: 41 of the 105 catalogued patterns (39%)
are structurally unreachable — spill cascades, plasma deadlocks, the 200 GB cap,
`/dev/shm` sizing, fragmentation, serialization cliffs. See the pattern scorecard below for
the per-pattern verdicts.

The AI-workload story was **materially weaker than the architecture suggested**, for two
reasons, both now fixed. The CPU→GPU overlap that keeps a GPU fed was implemented, tested,
and **shipped disabled**; it is now on by default. And the pipeline shape the ML surface
exists for could not be profiled at all — `stats()` refused it — so the first step of every
guide in the corpus had no answer here. Per-stage measurement now exists, and with it the
insights engine and three findings specific to inference pipelines.

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
| J1–J3 | Spilling invisible in dashboards; Ray Data metrics attributed entirely to the head-node IP, making per-worker analysis **impossible** | **P** | `observe/` consumes a per-subsystem event bus with real per-worker attribution. Not audited against the guides' specific gap list — flagged as follow-up. |
| §1 (all guides) | **"Always call `stats()` after a pipeline run"** — every guide's step one is reading per-operator wall time to find the bottleneck stage | **D** | `stats()` / `explain(analyze=True)` now measure a `map_batches`/ML pipeline per stage (`plan/profile/stages.py`), where they used to refuse. The user does not read the table to find the bottleneck either: `observe/insights/` names it, plus a starved GPU, an opaque plan, and a fan-out stage. See [What changed](#what-changed-in-the-observability-pass-2026-07-27). |
| data-020 | `torch.no_grad()` used where `inference_mode()` belongs, and the user must remember either one in every `__call__` | **D** | `ml/gpu.py::inference_mode_call` applies it at the engine's three GPU UDF call sites. No numerical effect, so it needs no probe — unlike `autocast`, which has to prove it pays. |
| troubleshooting §"`limit()` pushdown can produce wrong results" (ray#36295) | A **live upstream wrong-answer bug**: the optimizer pushes a `limit` above a map assuming it preserves row count, so a map that filters or expands rows silently returns the wrong rows | **A** | `kyber/rules/extra/limit_extra.py` lists pushing a limit below `MapBatches` as **unsound** by construction — "opaque row count" — alongside filter/unnest/sample/distinct/aggregate, and does not implement it. Verified against the guides' own check (compare to the full result, sliced) for both a dropping and an expanding map, with order-sensitive assertions because `limit` is a prefix and an order-independent comparison cannot see the bug. |
| troubleshooting §"task retries can produce duplicates" | A retried write re-emits rows already written | **A** | `FileSink.write`'s retry is safe *because* a failed attempt publishes nothing. Pinned at the dangerous case: a write that succeeds in writing bytes and *then* throttles still lands 3 rows, not 6 — the atomic writer discards the partial. |
| symptoms §"Training loss not decreasing" / data-026 | Training on ordered data: the model sees a contiguous run of the sort key per step and learns the ordering. The guides' listed causes are the learning rate and the data — never the ordering | **D** | `to_torch_dataloader(shuffle=False)` matches `torch.utils.data.DataLoader`'s own default and stays, because a user reaching for that method comes from torch. The *plan* is inspected instead: a `sort` the user wrote, feeding a loader with no shuffle window, warns. Pure plan shape — no scan — and silent for a corpus that was never sorted, or where the order is the point (a sequence model). |
| llm-batch-inference | An instruction-tuned model sent **un-templated** prompts still answers, in a format it was never tuned on — the guides' "degraded output with nothing to signal it" | **D** | The two backends had opposite `chat` defaults (`vllm_engine=False`, `http_engine=True`), each right for its ecosystem, so switching backends silently changed whether the template applied. `chat` left unset now checks the tokenizer: a model that ships a `chat_template` and is being sent raw completions warns once. An explicit `chat=False` is treated as a decision and stays silent — constrained-choice classification wants exactly that. |
| rag-pipelines §"Embedding dimension mismatch between indexing and query" | The index is built with one model and the query embedded with another; the two sides differ only by a number nobody looks at | **D** | The engine already refused, but as a `RuntimeError` from inside a Rust kernel ("string function list.CosineSimilarity: list dimensions must be equal") after the whole scan. Both widths are in the schema whenever the vectors came from `ds.ml.embed` (which emits `fixed_size_list`), so `similarity_join` now refuses at build time with a typed `PlanError` naming both widths and the actual cause — two different embedding models. A plain `list` column declares no width and is still left to the engine's per-row check. |
| batch-embeddings | Embedding vectors must be L2-normalized for cosine similarity; whether the *endpoint* already did it varies by provider, and getting it wrong ranks by magnitude as well as direction | **D** | `openai_embedding_encoder` defaults to `normalize=False` because OpenAI returns unit vectors — but the same request shape is spoken by Azure, Together, and vLLM's embedding server, and **vLLM does not normalize**. Rather than guess a default per provider, the first batch's norm is measured and a mismatch warns. Found by auditing why two sibling encoders had opposite `normalize` defaults. |
| memory §"Single giant row" | Marked **Unconditional**: "a single row larger than available task memory causes an unrecoverable OOM regardless of any other tuning... a hard architectural constraint, not a guideline". Safe ceiling ~10 MB, and the user is expected to know it | **P** | Byte-adaptive batching shrinks the chunk as rows widen, which handles the wide-row case Ray needs `target_max_block_size` for — but it bottoms out at one row, because rows are atomic. Both sizing paths now warn once at that dead end, naming the measured row width and the fact that the remaining fix is in the data (carry a handle, decode later). Batcher cannot remove the constraint; it can stop the user meeting it as an unexplained OOM. |
| multimodal §"`ArrowTypeError` when writing variable-size tensors" / G4 | Mixed-resolution images produce arrays of differing shape, which Arrow has no type for. Ray hits the same wall (`ArrowVariableShapedTensorArray`, ray#49883/#50229) | **A** | Carried, not refused. `io/formats/ml/ragged.py` stores each row as its own buffer plus its own shape and dtype, and the conversion paths reach for it automatically — a `map_batches` returning mixed-resolution arrays, or a `from_pydict` given them, both produce a variable-shape tensor column, and `to_numpy` decodes it back to per-row arrays. It is a plain struct, so the engine needed no change to carry it through a filter, a shuffle, or a Parquet write. `data` is a binary buffer rather than a list of elements because the FFI widens narrow numerics, which would have cost a `uint8` image eight bytes a pixel. |
| transforms §"Arrow Pickled Object Type" | A UDF returning PIL Images / torch tensors / custom objects gets a **pickle-backed** column: "10-100x slower than Arrow for inter-operator transfer", found by eyeballing `ds.schema()` for `object` | **A/D** | Batcher refuses rather than pickling — the column never enters the engine, so the slowdown is unreachable. It used to refuse with a raw `pyarrow.lib.ArrowInvalid` naming neither the column nor the fix; it now raises a typed `PlanError` naming the column, its element type, and the one-line remedy (`.cpu().numpy()` for a torch tensor, `np.asarray` for a PIL Image). |
| data-008, data-009 | `map()` where `map_batches()` belongs — row-at-a-time Python, 10-100x slower, "the most common performance mistake" | **P** | The vectorized spelling is the documented default and expressions are the primary surface, but `ds.map` exists and a docstring preference is not a signal. A run that spends real time in a per-row stage now says so (`per-row-map`), which required teaching the profile to tell `MapRows` from `MapBatches` — the engine sees one operator for both. |
| data-001, data-002 | Column pruning at read time: the guides' highest-impact IO fix, applied by hand with `columns=` | **D/P** | Kyber prunes automatically as a plan rewrite over the whole tree. The one exception is an opaque `map_batches`, which forces a full read; that case now warns at the call site naming `input_columns`, because it is the only fix and it cannot be inferred. |
| §8 (LLM) | `max_model_len` defaults to the model's full window; the guides prescribe sampling the corpus to find P95/P99 and setting it by hand, for 2-10x throughput | **P** | `vllm_engine(max_model_len="auto")` measures the prompts and sizes the KV cache from them, erring generous in every direction so it can never truncate. Opt-in rather than default: it defers the engine build to the first batch, which is a real change in when the model loads. |
| §5 (shuffle) | `local_shuffle_buffer_size` is a **row** count with no memory bound: the guides record a 2–5x slowdown and OOMs on image/embedding rows, fixed by hand-picking 512–2048 | **D** | The block is bounded by bytes as well as rows, so the request degrades to the widest window that fits instead of an OOM. `ml/loader/lazy.py`. |
| data-041 | `write_parquet(partition_cols=[...])` repartitions the **whole dataset** at write time, breaking streaming — a customer measured a **76% object-store spike** and write tasks hanging 30s+ | **A** | `FileSink.write_partitioned` shards a worker's *own* table into Hive directories; there is no global repartition step to spike. `sort_by=` is the explicit opt-in for a caller who genuinely wants clustered output, and it says so. |
| data-034, symptom "S3 SlowDown" | A fast pipeline drives a burst of concurrent PUTs at one key prefix; the store answers `SlowDown`/503 and the job dies. The fix is hand-throttling writers with a higher `num_cpus` per write task | **D** | `FileSink.write` retries a transient storage failure with jittered backoff, the same classifier and shape the read path already used — it had covered reads only. Safe because a failed attempt publishes nothing: the atomic writer discards the partial and the table is still in memory. `write_stream` deliberately has none: its batches are an iterator a retry cannot rewind. |
| K2 | Fixed batch sizes create stragglers; user must hand-write `estimate_processing_time(item)` | **D** | `ml/autobatch.py::ThroughputController` hill-climbs batch size against measured throughput under a VRAM cap, and **persists the settled plateau across runs** (`learned_batch_size`/`record_batch_size`), so a recurring job starts near last run's optimum. Ray has no batch-size auto-tuning at all; the guides name `batch_size=None` as the #1 OOM cause. |
| §2 (LLM) | `concurrency` defaults to **1** — one vLLM engine regardless of cluster size, "the most common misconfiguration on multi-GPU clusters" | **D** | `ml/gpu.py::resolve_num_workers` derives replicas as `total_GPUs / per-actor num_gpus` — the module comment is explicit: "never one engine idling a multi-GPU." The knob the guides teach you to compute is computed. |
| §1 (BI) | **"CPU preprocessing starving GPU is #1 cause"** of GPU underutilization | **D** | `stream_inference` now defaults **True** — see below. This was a **G** before this pass. |
| K1 | `batch_size=None` is the **default** and is "the single most common cause of heap OOM" — one block becomes one batch, and blocks can be hundreds of MB | **D** | Batch size is a control loop, not a default constant (`ml/autobatch.py`), and the morsel (16,384 rows, `bc-arrow`) bounds the unit independently of file or block size. |
| A1, A2, M3 | **Block count — not actor count — governs parallelism.** `ActorPoolStrategy(size=8)` on a one-block dataset runs sequentially, and "no warning is emitted." Too few blocks also means no autoscaling demand signal | **A** | Parallelism comes from morselization inside the engine, not from the physical layout of the input. A single large input file still fans out across cores. |
| B2 | Ray Data tracks the object-store budget **globally** but spilling triggers **per-node** — "the fundamental issue behind most unexpected spilling and OOM problems in multi-node clusters" | **A** | No global budget and no plasma store. Carbonite governs per-worker envelopes; spill is a local decision against a local bound. |
| F19 | `preserve_order=True` "can halve throughput — a **silent performance killer**"; it is a direct straggler amplifier | **A** | Order is a property of the plan (an explicit `sort`), not a scheduling mode, so there is no ordering flag that silently serializes the pipeline. |
| L2 | GPU actor-pool autoscaling **reloads the model on every scale-up** (30–120 s for large LLMs), so the guides advise pinning a fixed pool and giving up autoscaling | **D** | `warm_inference_pools` keeps pools warm across `collect()`s, so scaling does not imply reloading. This is why Batcher can have both — see G3 for the autoscaling half, which is still missing. |
| H13 | `max_errored_blocks = 0` by default — "a multi-hour batch job fails at 99% on one corrupt file, discarding all progress" | **D** | The distributed scan carries a broken-record policy in each partition manifest (`distributed.on_read_error`), so tolerance travels with the data rather than through global config. Default is still fail-fast, which is right for a small trusted input and wrong for a data lake — worth revisiting per-source. |

## The complete pattern scorecard

The corpus's two pattern catalogs — 46 `core-*` (Ray Core) and 59 `data-*` (Ray Data) — are
its most concrete artifact: one file per named mistake or technique, each with a stated
impact. This is every one of them checked against code, so a later pass can see coverage at a
glance instead of re-deriving it. **A row was only written once its cited module resolved**;
that check caught one wrong path (`data-050` cited a `kyber/skew.py` that does not exist —
the real detection lives in `kyber/stats/skew.py`).

Legend as above: **A** avoided structurally · **D** handled by a default · **P** partial.

Rather than restate 105 rows, the shape of the result:

* **Structurally unreachable (A) — 41 patterns.** Almost the whole `core-*` catalog. The
  object-store family (`core-007`…`core-013`, `core-036`, `core-037`, `core-043`,
  `core-044`) cannot occur because bulk Arrow never enters an object store; the
  task-granularity family (`core-001`, `core-002`, `core-014`, `core-015`) cannot occur
  because the scheduling unit is a morsel and the plan is lazy; the serialization family
  (`core-025`, `core-027`, `core-009`) cannot occur because Arrow is the only wire format.
  On the data side: column pruning and filter pushdown are plan rewrites (`data-001`,
  `data-002`, `data-015`), `group_by` is mergeable rather than a materializing shuffle
  (`data-024`), streaming is the default (`data-022`, `data-023`), and window functions are
  real operators rather than `map_batches` state (`data-036`, `data-055`).
* **Handled by a default (D) — 48 patterns.** Everything the guides tell a user to compute
  and set: batch size (`data-010`, `data-011`, `core-003`), GPU packing (`core-020`,
  `data-019`), model load-once (`data-016`, `data-017`), per-task memory and CPU
  (`core-018`, `core-019`, `data-021`), read concurrency (`data-003`), split sizing
  (`data-005`, `data-006`, `data-007`), retry (`core-028`, `core-029`, `data-034`), skew
  (`data-050`, `data-051`), schema evolution (`data-053`), and the CPU→GPU split
  (`data-030`).
* **Partial (P) — 4 patterns.** `core-035` (runtime env is a deployment concern, not an
  engine one), `core-046` (lease-based sessions survive an ungraceful kill, but no engine
  can run Python cleanup after SIGKILL), `data-057` (`max_model_len="auto"` exists but is
  opt-in), and `data-032`/`data-040`-class autoscaling items covered under G3.
* **Not applicable — 12 patterns.** Ray-mechanics entries with no Batcher analogue:
  `core-033` (`PYTHONUSERBASE`), `core-034` (`importlib.reload`), `core-040` (per-handle
  FIFO), `core-041` (worker setup hook), `core-042` (GCS port 6379), and the Ray-version
  changelog entries among `data-039`…`data-046`.

The count that matters is not how many rows say A or D — it is that **no row says "the user
must tune this"**, which is the property the whole corpus is a catalog of.

## The capability gaps Batcher closes by construction

The guides contain a section (`foundations/data/mental-model.md`,
`transforms/shuffles-and-joins.md`) listing what Ray Data **architecturally cannot do**, with
the recommended workaround usually being "use Spark, Daft, or DuckDB instead." That list is
the clearest statement of the competitive opening, and Batcher already holds most of it.

| Ray Data's stated boundary | Their workaround | Batcher |
|---|---|---|
| **"No external shuffle service — no Celeborn or Arrow Flight Shuffle equivalent for petabyte-scale shuffles"** | Daft on Ray, or Spark + Celeborn | `bc-transport` **is** a credit-controlled Arrow Flight shuffle. This is the single most direct hit on the list. |
| **"No SQL interface"** — a `map_sql` PR using DuckDB "was rejected as unsustainable" | DuckDB/Spark first, then feed Parquet to Ray Data | `_sql/` parser + `bt.sql`/`ds.sql`, differential-tested against DuckDB. |
| **"Ray Data passes 0 of 99 TPC-DS queries."** No join ordering, no bushy join optimization, no window functions | Daft for analytical queries | Kyber: 722 rules, bushy DP join ordering, window functions; TPC-H/TPC-DS/ClickBench suites in `benchmarks/suites/standard/`. |
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

## What changed in the scheduling pass (2026-08-06)

Five changes, all in the *placement and diagnosis* half of the Ray integration. Each one
either closed a case where a query hangs or runs badly with nothing anywhere saying why, or a
case where the standard Ray node labels carried an answer Batcher was not reading.

**A pending request now says why it is pending.** `ray.wait` cannot tell a task queued behind a
busy cluster from one asking for more CPUs than any node has, and the second never finishes.
Both used to be reported the same way — "it has waited a while, go run `ray status`" — so the
worst failure mode in the distributed path was also the least legible.
`capacity.describe_pending_demand` compares the ask against the live topology and returns one
of three answers: unsatisfiable (naming the *binding resource* and the widest node's figure,
because a reader told only "no node can host this" compares the cores they can see and
concludes the engine is wrong), short of capacity (the three numbers, since a gang needs
everything at once and a barrier does not), or nothing. It is attached to the map and inference
barrier's stall warning and to the placement-group timeout, which used to fall back to default
scheduling in silence — and the tasks it falls back to ask for the same resources the bundles
did, so the query would then hang at a barrier that has no deadline.

**Honest scope:** the *shuffle* barrier (`gather_with_backups`) still prints the unresolved
warning. Its loop is Carbonite's, the topology reader is `dist`'s, and Carbonite must not import
`dist` — so closing it means either threading a diagnoser through ten call sites in contested
files or moving the scheduling-envelope `ContextVar` down into `plan`. The second is the right
shape and is open work; a one-entry registry to bridge the layers would not be.

**A shuffle fleet is reserved inside one availability zone when one can hold it.** A shuffle
moves nearly all of its bytes worker to worker, and every cloud prices and delays those bytes
by the zone boundary they cross. The corpus is unusually emphatic about the size of this: cross-AZ
transfer "can exceed 40% of total AWS spend" and is listed at "up to 48% of AWS bill", field-
validated at Torc Robotics, with 20-40% added latency on synchronous workloads
(`foundations/infra/platform/cluster-and-infra-optimization.md`,
`foundations/infra/platform/anyscale-infrastructure.md`). Its own mitigation is
`STRICT_ZONAL_PACK`, which is an **Anyscale compute-config placement strategy, not a Ray one** —
`VALID_PLACEMENT_GROUP_STRATEGIES` in Ray 2.56 is `{PACK, SPREAD, STRICT_PACK, STRICT_SPREAD}`.
So it is a provisioning-time fix, available only to whoever configures the cluster, and the
corpus itself names the case where it is unavailable: `foundations/infra/compute/autoscaling-and-spot.md`
says to *always* enable cross-zone autoscaling for H100/H200 because single-AZ deployments
"frequently fail to provision". A fleet spread evenly
over three zones sends about two thirds of its shuffle across that boundary for nothing, since
the bundles are interchangeable. `capacity.preferred_fleet_zone` picks the zone with the most
*free* capacity that can host the whole fleet and pins the bundles there with
`bundle_label_selector`. No-op on a single-zone cluster, on unlabelled nodes, and whenever no
one zone has room; and because the pin is on the bundles rather than the tasks, a group that
cannot form is abandoned at the existing timeout rather than leaving tasks pending forever.
`distributed.zone_aware_placement` (default on, placement-only, result-identical) turns it off
for a fleet whose zone diversity is bought deliberately.

**A shuffle replica avoids the primary's *failure domain*, not just its node.** The copy has
always gone to another node; a spot reclamation takes an instance group, so on the fleet
`resilience="spot"` turns replication on for, the second copy went away in the same wave as the
first. `assign_replica_hosts` now ranks a non-preemptible candidate above a preemptible one,
below the off-node rule and above the load tie-break. Spot is read from `ray.io/market-type`
*and* the Karpenter, EKS, and GKE capacity labels, because a KubeRay fleet is labelled by
whichever provisioner brought the node up. Preference, never exclusion: an all-spot or
unlabelled fleet places exactly as before.

**The distributed scheduling decisions reach the dashboard.** `FanoutTrace` documented its own
gap: the bus attributes events by a `query_id` minted in `api`, `observe.store` silently drops
any event naming no live query, and `dist` must not import `api` to learn the id — so the
single most useful distributed diagnostic ("why did the job use a tenth of the cluster")
existed only as a log line. `events.query_scope`, set by the conductor around execution, puts
the id in layer 0 where both can reach it. The fan-out chain and the fleet placement (strategy,
bundle count, zone) now appear in `explain(analyze=True)` beside Kyber's and Carbonite's.

**The placement phase stopped re-reading the cluster.** The topology snapshot collapsed the
`ray.nodes()` reads and left the per-node *free CPU* read live, and `node_classes` goes through
it on every call — from five places in the placement phase. Measured: five `node_classes()`
calls inside a scope, **10.6 ms → 0.1 ms**.

Also fixed while in the area: `_apply_autoscale_floor` used `if resources: ... elif gpus:`, so
naming a custom accelerator *replaced* the GPU bundles instead of joining them. A job with a
GPU stage and a TPU stage asked the autoscaler for one of the two and waited out the window for
the other. And `dist/executor.py`'s two fan-out sizers swallowed a topology read with no
`note_suppressed`, so the failure that halves a large cluster's fan-out left nothing to
attribute it to — beside a `FanoutTrace` whose entire purpose is answering that question.

### Open: `ml/gpu.py` rounds device fractions off its own ladder

`_internal/device_share.py` exists because "five different files had each written their own
answer", and its `PACK_QUANTA = (0.25, 0.5, 1.0)` ladder is there because Ray packs by *summing*
requests, so arbitrary fractions strand slivers no task fits into. Kyber, Carbonite and `dist`
all go through `pack_fraction`. **`ml/gpu.py` does not**: `recommend_gpu_fraction` and
`recommend_num_gpus` return `round(1.0 / n, 2)` for arbitrary `n` (four call sites), which yields
`0.33`, `0.17`, `0.14`.

Two independent sources say that is the wrong shape.
`foundations/core/scheduling-and-resources.md` — "use values that divide evenly into 1.0 (0.25,
0.5)" — and the canonical `resources/canonical-params/model-to-gpu-matrix.md`, which prices
every model in the corpus at `0.1`, `0.25`, `0.5` or `1` and never at a non-dividing value. More
to the point, it is the exact failure `device_share`'s own docstring describes: "a stage packed
at `0.25` by the optimizer and admitted at `0.2` by the resource manager is not a rounding
difference; it is five tenants on a device sized for four."

Not fixed here, because changing device packing is a device-tier behaviour change and
`.claude/rules/device-tier.md` requires a recorded `gpu_shadow_verify` run on real hardware,
which this pass had none of. The fix is to route those four sites through
`device_share.quantize_fraction`. (Batcher's floor of `0.25` against the matrix's `0.1` is a
*deliberate* difference, argued in `PACK_QUANTA`'s comment on CUDA-context and copy-engine
contention — do not "fix" that one to match.)

### Open, with the discovery already paid for: CPU-fleet isolation without cluster setup

`heterogeneous_node_isolation` keeps a relational fleet off accelerator nodes, and it is
default-off **and** requires the operator to advertise a `cpu_node` custom resource on every
CPU node. By this document's own bar that is a capability nobody gets. Ray's label selectors
can express it against labels Ray sets by itself, and the syntax was verified on a live 2.56
cluster rather than read off the documentation:

| Selector | Result |
|---|---|
| `{"ray.io/accelerator-type": "!GPU"}` on an **unlabelled** node | matches — an absent label counts as not-equal |
| `{"ray.io/market-type": "on-demand"}` (control) | matches |
| `{"ray.io/accelerator-type": "!"}` | **`ValueError`** — a bare `!` is not a valid value |

So the form is `!in(...)` over the accelerator models the live fleet actually advertises,
which `scaling.node_classes` already carries per node — not `"!"`, and not `"!GPU"`, since no
node is ever labelled with the literal string `GPU`. Applied to `bundle_label_selector` it
inherits the same benign failure mode as the zone pin. It is not built here because it cannot
be verified end to end on a CPU-only cluster, and shipping an unexercised scheduling
constraint is how a fleet becomes unplaceable.

## What changed in the observability pass (2026-07-27)

Every guide in the corpus opens the same way: *call `ds.stats()`, find the slowest stage,
tune that*. Bottleneck attribution is step one of all of them. Batcher answered that question
for relational queries and **refused to answer it at all for AI pipelines** — the exact
workload class the ML surface exists for:

> `explain(analyze=True)/stats() is not available for map_batches/ML pipelines (the opaque
> UDF path emits no per-operator metrics); profile the relational portion instead.`

That was true of the engine and false of the orchestrator. The engine emits `ExecMetrics`
per operator and never sees a Python UDF — but `core.udf` runs those stages itself and knew
exactly how long each one took, how many rows it consumed and emitted, and how many bytes it
produced. Nothing collected it.

**Fixed.** `plan/profile/stages.py::StageRecorder` collects per-stage measurements in the
same shape as the engine's `ExecMetrics`, so the profile builder joins Python stages and
engine operators with one code path. Both UDF execution routes report: the materializing
tree walk (`core/udf/execute.py::_execute_node`) and the stage-overlapped streaming chain
(`core/udf/stream.py`, via `plan/profile/stages.py::metered`). Stages are numbered by a
pre-order walk of the logical plan — the numbering the planned tree already uses — so a
measurement cannot land on another stage's row.

One caveat worth stating, because a profile that is misread is worse than none. On the
stage-overlapped streaming path a stage is metered by wrapping its output generator, so its
number is **residency**, not pure compute: when its input queue is empty the wait for
upstream is included, and a stage fed by a slower one reads high. That is the honest reading
of a pipeline — the stage really was occupied — and it is why the bottleneck call compares
stages rather than trusting any single figure. The materializing path brackets the UDF call
itself and has no such ambiguity.

Three consequences, in increasing order of value:

1. `stats()` and `explain(analyze=True)` work on a `map_batches` pipeline, naming the
   bottleneck stage with its measured rows and time.
2. **The whole existing insights engine came with it.** Fourteen rules
   (`observe/insights/`) were already written against a measured profile and had simply never
   had one to read for this workload class. Spill, memory-headroom, CPU-idle, and
   dominant-operator findings now fire for inference pipelines with no new code.
3. Findings that were previously impossible became expressible, because per-stage timing is
   what they need. Three shipped in `observe/insights/stages.py`:
   **`gpu-starved`** (the CPU stages feeding a GPU stage cost more than it does — the guides'
   named #1 cause of low GPU utilization, and invisible in a plan because the pipeline is
   *correct*, the device is just idle), **`udf-dominates`** (Python stages own most of a plan
   that also has relational work, so pushdown and fusion stop at a wall that need not be
   there), **`row-exploding-stage`** (a one-to-many stage multiplying everything
   downstream), and **`per-row-map`**.

   That last one needed the profile to learn a distinction it did not have. `map` and
   `map_batches` are the *same operator* to the engine — `map` lowers to `map_batches` over a
   row loop — so the profile called both `MapBatches` and could not say which one a run had
   paid for. The gap between them is the corpus's most-repeated number: 10-100x for anything
   expressible over columns (`data-008`/`data-009`). The row adapters now mark themselves and
   the profile reports `MapRows`, so a large per-row stage owning real time gets named,
   with the vectorized spelling to replace it. It stays quiet on a small stage and on the
   cases where per-row is genuinely right — row-shaped work, or an async per-row API call
   whose cost is the network.

**The findings reach the terminal, not only the dashboard.** `derive_insights` was consumed
in exactly one place — the web UI's run page — so the engine could know a run had spilled or
starved its GPU and tell nobody who had not run `bt.start_ui()`. `RunStats.findings` now
carries them, and `str(stats)` prints the warnings and criticals with their actions beneath
the operator table. `stats()` is where every guide in the corpus sends a user, so it is where
the conclusions belong. Two deliberate restrictions: `info` findings are carried on the
object but not printed (an advice block under every healthy run is how a reader learns to
skip the one that mattered), and a rule that raises costs the commentary and nothing else —
findings are commentary on a measurement that already succeeded.

**The rules stopped going silent on distributed runs.** They read `profile["ops"]`, which on
a distributed run carries no measurements: the work happened on the workers and arrives as
`worker_ops` in its own op-id space. So every rule returned nothing for cluster runs — where
a spill or a starved GPU is both likelier and more expensive than on one node. `derive` now
falls back to `worker_ops` when the driver tree is unmeasured.

**`max_model_len="auto"` sizes the vLLM context window from the data** (`ml/llm/sizing.py`).
The KV cache is reserved from that number and the default is the model's full window, so a
128K default over 2K-token prompts spends nearly all of the cache on lengths the data never
reaches — capacity that would otherwise be concurrent sequences. The guides put the gap at
2-10x throughput and prescribe sampling the corpus by hand. Batcher sees the prompts, so it
measures instead. The sizing may only ever err generous: characters convert to tokens at a
rate no tokenizer beats, the full generation budget is added, headroom covers prompts the
sample never saw, and the result rounds up to a bucket — because an oversized cache costs
throughput while an undersized window truncates a prompt and degrades output silently. If
the model refuses the proposal, the run falls back to the model's own window with a warning
rather than failing. It stays opt-in: it defers the engine build to the first batch, which
is a real change in when the model loads, and it should be chosen rather than inherited.

**The one place automatic projection pushdown stops working now says so.** Pushdown is the
highest-impact IO optimization in the corpus (`data-001`/`data-002`: 2-10x on a wide table,
10-50x past 50 columns) and it is one Batcher does for free — right up to a `map_batches`,
whose `fn` the optimizer cannot see into and must therefore assume reads every column.
`input_columns` is the declaration that restores it, and nothing can infer it. So a
`map_batches` over a wide table with no `input_columns` raises a `PerformanceWarning` at the
call site that caused it, rather than leaving the unpruned scan to be found in a profile.
Narrow tables stay silent: advice under every call is how a reader learns to filter it out.

**Transient-failure retry reached the write path.** `is_transient`/`with_retry` — the
classifier that separates a 503 slow-down from a 404, and the jittered backoff that keeps a
wide scan's retries from stampeding — existed and was wired into `FileSource` only. A
throttled PUT therefore killed the job. It now covers `FileSink.write`, which is the shared
path under chunked and Hive-partitioned directory writes.

Four smaller fixes in the same pass, the first two cases of a documented hazard becoming an
enforced bound:

**`torch.inference_mode()` is now applied by the engine** (`ml/gpu.py::inference_mode_call`,
wired at all three GPU UDF call sites). PyTorch builds a backward graph on every forward
unless told not to; an inference stage never runs that backward pass, so the graph is pure
waste — host overhead per op, plus the activations it pins, which is what caps the batch
size. The pattern catalog has an entry for it (`data-020`) precisely because remembering it
in every `__call__` is the user's job in Ray Data. It is a pure resource win with no
numerical effect, which is what makes it safe to apply unconditionally, unlike `autocast`
(which changes precision and therefore has to prove it pays first). A UDF whose *output* is a
gradient declines with `batcher_inference_mode = False`.

**`local_shuffle_buffer_size` is now bounded by bytes as well as rows.** It is a row count,
and a row count says nothing about memory: 50,000 narrow tabular rows is a few MB, and 50,000
decoded 224×224 images is ~30 GB. The guides record a 2–5x slowdown and OOMs on large rows,
with "use 512–2048 for wide rows" as the manual fix. The knob now means *decorrelate as much
as fits*: a request too large for the row width degrades to a narrower window instead of an
OOM. Cutting a block early only narrows the shuffle window — it never drops or repeats a row.

### The guard family, and why it stays quiet

Several of the fixes above are *warnings* rather than behaviour changes, which is a weaker
kind of answer and worth justifying. Each one is a case where the engine can **detect** a
hazard but must not silently resolve it: normalizing an already-unit vector is pure cost,
applying a chat template to a base model is wrong, shuffling a sequence dataset destroys it,
and a row cannot be split at all. Guessing would trade one silent failure for another.

They were also all found the same way, and it did not involve the guides: scanning for the
same parameter carrying **conflicting defaults across sibling APIs**, then asking why the two
disagreed. In each case both defaults were individually right and the *divergence* was the
bug — so for `chat` and `shuffle` the fix keeps both defaults and detects the hazardous
combination instead of unifying them.

**Validate a rule by running the symptom, not by constructing its profile.** Two of the four
new rules shipped with logic that their own unit tests could not see, because the tests were
written against the same mental model as the bug. `gpu-starved` put its size floor on the
*GPU stage's* duration — but a starved GPU is by definition one that spends little time
working, so the floor suppressed the finding precisely when starvation was worst (silent at
11x). Running the guides' actual pipeline shape end-to-end is what exposed it; a synthetic
profile never would, because writing one means choosing the numbers. Every rule here is now
checked against a real pipeline as well as a fixture.

Note the near-miss beside it: `udf-dominates` gates on the *UDF's* time and that is correct,
because a large UDF share **is** the finding. Same code shape, opposite verdict — which is
why the fix was not applied symmetrically.

The bar for advice is that it stays silent on correct code, because a guard that cries wolf
teaches users to skip the section that mattered. Measured: across the whole executed docs
corpus these six warnings fire **zero** times, and across the ~9,280-test unit suite exactly
once — in the test that deliberately triggers it.

## What changed in the earlier pass

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

### G1. Straggler handling — speculation is on, and the map task unit is no longer ≈ node

Batcher's coarse scheduling unit avoided Ray's Raylet-saturation family (C1–C3) by inheriting
the opposite problem: a straggler's *entire* partition had to be redone.

**Closed this pass for the Flight shuffles.** The map stage cuts its input into `workers x
map_partition_multiplier` partitions (4x by default) and `map_barrier` deals them out as actors
go idle, with exactly `workers` in flight. A slow worker takes fewer partitions rather than
holding the barrier open on a full one, and a dead worker's share is re-dealt across survivors.
The count is bounded by the splits the source actually has, so a small input is unaffected, and
`map_partition_multiplier = 1` pins the old unit. Aggregate, join, sort and window all take it;
the disk transport and the non-shuffle map fan-out do not.

The *other* half of the Spark argument — that fine partitions dilute skew — does not follow and
should not be written down as though it does. Map partitions divide the input; a shuffle's
imbalance lives in its hash buckets, and a single dominant key is indivisible however fine the
hash. That one is salting's job (G6), not this.

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

**`skew_join_salt = 0` is not "off"** and must not be described as an unflipped default-off
capability — that reading is what the config comment used to invite, and it is wrong twice
over. Salting already engages on *measured* skew at 0 (G6), so there is nothing to flip; and
what does happen at 0 is guarded by `salting_is_safe`, which is the hazard G6 describes. The
knob is a fan-out: a positive value forces the pre-pass and pins the fan-out, and a negative
value is the off switch for pinning the plain co-partition shuffle.

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

**Also closed:** a wide shuffle reduces through the combiner tree, whose *interior* levels were
single-copy at any replication factor — one lost combiner discarded every level built so far and
restarted from the leaves, which is the recompute leaf replication exists to avoid, reintroduced
one level up. `dist/shuffle_replication.py::replicate_interior_outputs` now copies each level's
merged partials off-node before the next level is built on them, and `_tree_reduce` threads the
acked addresses into the next level's positional fallbacks. It is the cheapest copy in the
shuffle — a level's output is `fan_in` partials already merged into one — and it needs no epoch
fence, because an interior ticket cannot outlive the attempt that made it (every level's
fallbacks are built inside one `_tree_reduce` call and no reference escapes it).

Honest scope: `test_shuffle_replication.py` pins that the interior copies are *placed and acked*,
which is the defect that existed. It does not kill a worker between combiner levels — the fault
hooks fire before and after the map barrier, not mid-tree — so "an interior replica served a
loss" rests on `_combine_sources`' positional fallback rather than on an end-to-end test.

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

### G3. Single split point, no per-stage autoscaling — **both halves now closed**

**The N-stage split landed** (`dist/streaming/relay.py` plus
`plan_analysis.split_into_resource_stages`): the chain is cut at *every* resource boundary, so
two chained models get a pool each and a CPU postprocess no longer runs on the GPU actor. The
paragraph below describing "exactly one boundary" is the state before that, kept because the
measured argument for the placement is still the argument.

**Per-stage autoscaling landed with this pass, and it started as a behaviour bug rather than a
missing feature.** `concurrency=(min, max)` is documented in four places as "the pool autoscales
to the backlog", and on the `_drive_actor_pool` path it does. The streaming pipeline resolved the
same spec *statically* — `_resolve_pool_size` clamps the partition count into the range — so the
identical public argument meant "autoscale" on one path and "a fixed pool sized from a number you
never wrote" on the other, and a stage that fell behind stayed behind for the whole query.
`consumers.consumer_pool_bounds` now returns `(start, ceiling)` and `schedule._rescale_stages`
grows a stage whose post-dispatch backlog could not be placed, reusing `map._autoscale_action` so
the two paths cannot drift. A plain int or an absent spec yields `start == ceiling`, which the
scheduler reads as "do not scale" — so nothing that was not asking for autoscaling changes.

It grows and deliberately does not shrink, which is the opposite of the actor-pool path. There
`pending` is a partition queue that only drains, so a reap at the tail is safe; here the backlog
rises and falls morsel by morsel, and the same rule would reap an actor the moment a stage caught
up and respawn it on the next morsel — each of those actors holding a *loaded model* whose reload
costs tens of seconds against a fraction of a second of idling. Growth also has no correctness
hazard where a reap does: a relay's published morsels live on its own Flight server, so killing
one that still holds any voids a subtree and forces a replay.

Pinned in `tests/unit/test_stream_schedule_algebra.py` (a backlogged stage grows, the ceiling is
respected, a fixed stage is untouched, and a grown pipeline delivers the identical row multiset)
and `tests/unit/test_stream_pipeline_split.py` (the bounds themselves). Both run against the
deterministic stand-in scheduler, so they need no cluster.

**Narrower than it reads.** Checked against the guides' own four-stage shape
(`extract` CPU → `chunk` CPU → `embed` GPU → `write` CPU): the single split groups both CPU
stages into the producer and the GPU stage plus its postprocess into the consumer, which is
the placement the measured win comes from — the decode never holds a GPU. What is still
missing is a *third* pool for the postprocess, which costs only when that stage is heavy.
Pinned in `tests/unit/test_stream_pipeline_split.py`; the split is pure plan inspection, so
that half is testable single-node even though the N-pool Flight topology is not.

`stream_inference` splits at exactly *one* resource-class boundary. A three-stage pipeline
works (the third stage rides in the consumer), but there is no N-stage topology and no
per-stage pool autoscaling. The guides' RAG indexing pipeline is genuinely four stages with
three different resource classes (`extract` CPU-heavy → `chunk` CPU-light → `embed` GPU →
`write`), and reports **~86% cost reduction and ~69% faster** end-to-end from getting that
placement right versus an all-GPU cluster.

### G4. ~~No variable-shape tensor type~~ — **closed**; no GPU residency across operators

**Closed.** `io/formats/ml/ragged.py` carries mixed-resolution arrays as one column, and the
conversion paths reach for it automatically: a `map_batches` returning arrays of differing
shape, and a `from_pydict` given the same, both produce a variable-shape tensor column instead
of failing. `to_numpy` and `batch_format="numpy"` decode it back to per-row arrays.

The layout is a plain `struct<data: binary, shape: list<int32>, dtype: string>`, and both
halves of that are load-bearing. It is ordinary Arrow, so it crosses the FFI, writes to
Parquet, and passes through every operator **with no engine change, no IR tag, and no
wire-contract change**. And `data` is a *binary buffer* rather than a list of elements because
the boundary widens narrow numerics: a `list<uint8>` image column arrives as `list<int64>`,
eight bytes per pixel, for the one workload the type exists to carry.

What it deliberately does not do is present a ragged column to torch as a tensor. Rows of
differing shape have no stacked form, so the bridges hand back per-row arrays rather than
silently padding.

Still open in this item: data round-trips to host memory between operators; it never stays
resident on the GPU. Ray has the same limitation (its GPU object store is RFC-only,
ray#51173).

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

:::{note}
**Superseded since this section was written.** The glob-listing fix described below as
"tried and reverted" was re-landed in `0f4c27e` with the staleness hole closed: `_glob` now
calls `_record_listing(infos)`, and `atomic_writer`/`remove` call `_forget_listing(path)` as
they write, so this process overwriting its own deterministically-named output drops the
stale entry. An overwrite by *another* process still needs a stat, which is the same exposure
the directory-listing branch has always had. Re-benchmark before quoting the numbers below.
:::

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
