# Competitive architecture: where Batcher wins, where it loses, and what it must build

**Status:** audit, 2026-07-14. Every claim below was checked against code, not documentation.
Where the docs and the code disagreed, the code won and the doc is named as wrong.

The mandate is *"beat DuckDB, Spark, Ray Data and Polars across the whole range — sub-second to
PB, batch and streaming, single-node and distributed"* (`CLAUDE.md`). This document answers, for
each competitor, whether the **architecture** can currently cash that cheque.

## Verdict

**Batcher does not currently always win, and four of the gaps are structural** — no amount of
kernel tuning closes them, because they are consequences of the execution *model*, not of the
code quality inside it. The engineering that exists is genuinely good; the ceilings are real.

The honest one-line summary:

> Batcher today is **the best distributed Arrow engine with a learned optimizer**, and it is
> *not yet* a general-purpose engine that wins at every scale. It loses to DuckDB single-node
> above ~10M rows, it cannot express Flink's streaming guarantees, and its AI moat is switched
> off by default.

Four claims in the repo are not supported by the code and should stop being made until they are.
They are listed in [Claims to retire](#claims-to-retire).

## Where Batcher genuinely leads (verified in code)

These are real, and none of the competitors have all of them:

1. **Mergeable algebra as the single semantics.** `partial → combine → finalize` in `bc-runtime`
   serves one core, many cores, and many machines. Spark shuffles raw rows; Batcher's aggregate
   mappers publish *pre-aggregated partials*, which is what makes the hierarchical combiner tree
   (`dist/flight_aggregate.py:388`) and bounded reducer memory cheap.
2. **The data plane bypasses the Ray object store entirely.** Only `(addr, ticket)` strings transit
   Ray; bulk Arrow moves over Flight with **credit-based flow control** whose bound is *proven* by
   an in-flight gauge (`crates/bc-transport/src/store.rs:16-62`). This is the single biggest reason
   Batcher beats Ray Data 50–450× — Ray Data's object-store spill storms are structural.
3. **A learned cross-query loop nobody else has.** Sketch-backed cardinality (HLL/KLL wired end to
   end), cost coefficients *calibrated from measured `op_stats`*, a UCB1 bandit over join
   strategies, learned partition counts and hot keys (`kyber/learning.py`, `learned_tuning/`,
   `dist/skew.py`). DuckDB and Spark have nothing comparable.
4. **O(1)-memory global shuffle for training.** A 4-round Feistel permutation with cycle-walking
   (`ml/permutation.py`) makes the epoch order a *computed bijection*, not a materialized index
   list — so a rank streams an exabyte corpus in constant memory and seeks anywhere instantly.
   This is MosaicML-StreamingDataset's signature feature, done properly, plus mid-epoch resume
   and elastic (world-size-independent) ordering that Ray Train lacks.
5. **Proactive spot-preemption migration** at stage boundaries (`carbonite/resilience/preemption.py`);
   Spark has no out-of-the-box equivalent.
6. **Adaptive aggregate switching on *measured* reduction ratio** (`bc-interp/src/agg_par.rs`) —
   more principled than a static optimizer estimate.
7. **Session-warm inference actor pools** — the model loads once per *session*, reused across
   `collect()`s. Ray Data respawns per execution.

## The scorecard

Legend: **W** Batcher wins architecturally · **=** parity · **L** Batcher loses · **✗** cannot do it.

| Dimension | vs DuckDB | vs Polars | vs Spark/Databricks | vs Flink | vs Ray Data / Daft | vs Snowflake |
|---|---|---|---|---|---|---|
| Small-query latency | = | = | **W** | — | **W** | **W** |
| Single-node ≤10M rows | **W** | **W** | **W** | — | **W** | — |
| Single-node ≥100M rows | **L** (2–11×, **OOM** on q3/q4/q5) | **L** on 6 shapes | — | — | **W** | — |
| Distributed batch | **W** | **W** | = | — | **W** (50–450×) | L |
| Optimizer breadth | = (302 rules, bushy DP join order) | **W** | **W** | — | **W** | L |
| Learned/adaptive | **W** | **W** | **W** | — | **W** | = |
| String execution | **L** (no StringView, dict decoded at leaf) | **L** | = | — | L | L |
| Streaming guarantees | — | — | L | **✗** | — | — |
| Fault tolerance | — | — | **L** (RAM-only shuffle) | L | = | L |
| Skew handling | — | — | **L** (AQE splits; Batcher salts opt-in) | — | = | L |
| AI / GPU pipelines | — | — | — | — | **L** by default | — |
| Lakehouse formats | — | — | L (all via pyiceberg/delta-rs) | — | = | L |

## AI-preprocessing coverage: the native expression surface (verified in code)

The scorecard row "AI / GPU pipelines" is about *execution* (GPU residency, the disabled
streaming pipeline) and stays **L by default** — see ceiling #5, unchanged. This section is
narrower and separate: the **in-engine, no-per-row-Python preprocessing surface** — image,
audio, vector, and text-mining operators that run in the Rust data plane over Arrow rather
than in a `map_batches` UDF. On *this* axis Batcher is now unusually broad, and two primitives
(native mel-spectrogram and MFCC) have no in-engine equivalent in any competitor surveyed.

What runs natively (data plane, parallel, per-row-null-tolerant), with the kernel:

| Capability | Where | Note |
|---|---|---|
| Image decode → resized `uint8` tensor | `bc-expr/.../media/image.rs::to_tensor` | DCT-scaled JPEG + SIMD resize |
| Image decode → **normalized `f32` tensor** (`/255`, per-channel mean/std, HWC/CHW) | `image.rs::to_tensor_f32` | the torchvision `ToTensor`+`Normalize` step, in-engine |
| Image **center-crop** (torchvision-style zero-pad) | `image.rs::center_crop` | the crop half of resize→crop→tensor |
| Image **grayscale** (Rec.601 luma → 1 channel) | `image.rs::to_grayscale` | color-convert for 1-channel models |
| Perceptual hash (dHash) for image dedup | `image.rs::dhash` | Hamming-thresholded near-dup |
| Audio decode / mono / resample (sinc) | `media/audio.rs` | WAV/FLAC (symphonia) |
| **Mel power spectrogram** (STFT + HTK filterbank) | `media/mel.rs` | matches `torchaudio.transforms.MelSpectrogram` to 1e-6 |
| **MFCC** (mel → AmplitudeToDB → DCT-II) | `media/mel.rs::mfcc` | matches `torchaudio.transforms.MFCC` to 1e-6 |
| Vector distances: cosine, dot, L2, **L1, Hamming** | `eval/list.rs`, `list_ops/coerce.rs` | accept `FixedSizeList` (the tensor type), `f32` fast path |
| Vector reductions: l2_norm, normalize, mean/max pool, argmin/max | `eval/list.rs` | embedding sanity/pooling |
| MinHash / SimHash LSH signatures | `eval/str/minhash.rs`, `list_ops/simhash.rs` | fuzzy-dedup / similarity-join blocking |
| Fuzzy string match: Levenshtein, **Damerau-Levenshtein, Jaro, Jaro-Winkler** | `eval/str/{mod,jaro}.rs` | matches DuckDB; entity resolution |
| Exact top-k vector retrieval verb | `api/dataset/ml.py::nearest_neighbors` | brute-force, composes the distance kernels |
| **Vector search in SQL** (`ORDER BY list_cosine_similarity(emb, [...]) LIMIT k`) | `_sql/parser/expressions/functions.py` | the two-arg vector functions run in SQL |
| RAG document chunking, HTML strip, token estimate | `eval/str/{chunk,html}.rs` | corpus prep |

How the competitors compare **on this native-preprocessing axis** (not on GPU execution or
training, where Batcher does not compete — see ceilings):

- **Daft** is the closest. It has native URL-download, image decode/resize/crop/`to_mode`,
  a variable-shape tensor type, and embedding/cosine expressions — a strong multimodal surface.
  Batcher's image surface now **covers Daft's completely**: center-crop, offset `crop`,
  `encode` (png/jpeg/bmp/gif), and `convert` over the full `L`/`LA`/`RGB`/`RGBA` palette,
  with the four header facts (`width`/`height`/`channels`/`mode`) from **one** read where
  Daft spends a call per fact. It leads on **audio**: Daft has **no native
  mel-spectrogram / MFCC** (per-row UDFs), which Batcher does in-engine (both torchaudio-matched).
  Daft still leads on the **variable-shape tensor type** (ceiling below). The
  color-conversion gap this bullet used to record is closed; see
  `daft_parity_ledger.md`.
- **Ray Data** has good CPU preprocessors (scalers/encoders/imputers) and a tensor type, but
  multimodal decode is a torch/PIL **UDF per batch**, not an engine expression; **no native
  mel-spectrogram**; fuzzy-match/minhash are user code.
- **BigQuery ML** has the richest *SQL-level* ML (learned `ML.STANDARD_SCALER` etc.,
  `VECTOR_SEARCH`, `AI.GENERATE`). Batcher now has **vector search in SQL** — the two-arg
  vector functions (`list_cosine_similarity`/`list_distance`/`list_dot_product`, DuckDB-matched)
  run in `SELECT … ORDER BY … LIMIT k`, so brute-force retrieval is a plain query. BigQuery
  still leads on **model inference in SQL** (`ML.PREDICT`/`AI.GENERATE`) and **learned
  preprocessing state** (`TRANSFORM`), which Batcher does not have — the standing ML-in-SQL
  gap. And BigQuery decodes **no media in its engine**: image/PDF ops are per-row Cloud Run
  containers that HTTP-fetch each object, so in-engine preprocessing throughput is not its game.
- **Spark (MLlib)** has the mature training/feature ecosystem but **no native multimodal
  expressions** and no mel-spectrogram; **spark-rapids** accelerates SQL, not media decode.
- **Polars / DuckDB** have no image/audio surface at all; DuckDB has the fuzzy-string and some
  list/vector functions Batcher is differential-tested against, but no multimodal.

**Honest bottom line for this axis:** among general-purpose engines, Batcher has the broadest
*native* (non-UDF) AI-preprocessing expression surface, and native mel-spectrogram + MFCC are a genuine
first. This does **not** change the GPU-execution verdict (ceiling #5) or close the two AI gaps
called out next: **no ML-in-SQL** (BigQuery leads) and **no variable-shape tensor type** (Daft
and Ray Data lead). Do not let a preprocessing-coverage win be read as a GPU-pipeline win.

## The structural ceilings

Ranked by how much they block the mandate. These are the work.

### 1. The executor materialized every operator's output — **fixed (streaming is now default)**

`bc_interp::par::exec()` returned `Vec<RecordBatch>` for **every** node, breaker or not, so a
`scan → filter → project → join` chain held the scan's output, the filter's output, the project's
output, and both join inputs in RAM at once. At **sf100, q3/q4/q5 peaked at 133 GB and were
OOM-killed** — queries DuckDB completes in ~1.2 s. That single fact falsified "any data scale,"
and it was the highest-value work in the repository.

**It is now implemented and default** (`crates/bc-interp/src/stream/`, dispatched from
`bc_py::execute_plan`; `EngineConfig.streaming` defaults true). The streaming executor pulls
morsels through the linear runs and materializes only at breakers — the pipeline/breaker model
the docs always described — so its peak memory is the breakers' state plus one morsel per worker,
a **constant, independent of the input size**. Measured on the q3/q4/q5 shape: 3.4 MB at 1M rows,
3.3 MB at 4M, where the materializing path grew 198 → 794 MB. The sf100 OOM disappears
*structurally*, not by tuning — and it is also **1.24× faster** than the materializing parallel
path on that shape, which is why it could become the default rather than an opt-in for the large
case.

The hash-join **probe** streams (`BroadcastProbe`, built once) and the aggregate folds
incrementally (`partial`→`combine`) — the mergeable algebra applied to the two operators that
were still reading their whole input first. A budget check hands a query whose breaker state
exceeds the memory envelope back to the spilling (materializing) executor, so a large in-memory
win never becomes an out-of-core OOM. Breakers stay breakers on purpose — they are the adaptive
re-optimization points (the moat), and streaming *preserves* them, reaching the first correction
sooner. Verified `execute_streaming == execute` over every operator (serial and sharded), metrics
agree with the oracle, and the full DuckDB differential suite passes with streaming as the
default.

**Still open** (RFC Proposals 2–5, incremental from here): selection vectors to kill the
high-selectivity filter gather tax; a spilling streaming breaker to remove the fall-back-to-
materializing condition; growing the JIT to vectorized aggregate/hash kernels.

### 2. No string-optimized representation (partly closed)

Zero `StringView`/`Utf8View` anywhere in the codebase (German strings — DuckDB and Polars both
use them). And **dictionary encoding is destroyed at the leaf**: `decode_dict`
(`bc-expr/src/eval/dispatch.rs`) casts any `Dictionary` column to its value type *before any
kernel runs*, so dictionaries never survive into project/join. String join keys still fall to the
generic byte-row `RowConverter` path.

**Closed this pass, for comparisons.** `try_dict_compare` (`bc-expr/src/eval/binary.rs`) gives
`<dict column> cmp <literal>` the same treatment `InList` already had: compare the *distinct
values* (cardinality-length) and gather one bit per row through the keys, never decoding the
column. Measured on 6M rows × 25 distinct values — TPC-H `l_shipmode`, ClickBench's whole string
shape — `m = 'AIR'` goes **144.9 ms → 7.4 ms (19.6x)** against the same column already decoded;
the old dictionary path was worse still, paying a full decode on top of that 144.9 ms. It is
bit-identical to the decoded path by construction (an elementwise comparison commutes with a
gather: `cmp(take(v, k), lit) == take(cmp(v, lit), k)`), which is what lets it live inside the
correctness oracle, and that equality is pinned by tests over every comparison operator, both
operand orders, null keys, and null dictionary *values*.

Still open on this axis: `StringView`, dictionary survival through project/join, and a
dictionary-aware string join key. Filters and group-by are now dict-native; the rest is not.

**Also closed this pass, for conjunctive filters.** Batcher evaluated *every* conjunct of an
`AND` over *every* row and `and_kleene`'d the masks, so a five-predicate filter that kept 1.65%
of its rows paid five full-width passes. DuckDB does not: `ExpressionExecutor::Select` walks
the conjuncts against a selection vector so conjunct *n+1* sees only what conjunct *n* kept.
`Expr::short_circuit_filter_mask` (`bc-expr/src/select.rs`) is the Arrow-shaped equivalent —
evaluate the cheapest conjunct at full width, then *compact* (gather the survivors of only the
columns the remaining conjuncts name) and evaluate the rest against that. Measured over 64
morsels of a 17-column lineitem shape, whole Filter operator including the gather both paths
pay: **TPC-H q6 1.50x, a `LIKE` behind a cheap guard 2.03x, a `regexp_matches` behind one
5.73x.** The mask alone is 1.90x / 2.92x / 8.75x.

It is bit-identical by construction rather than by measurement, and the guard is the
interesting part: only conjuncts whose failures are *schema*-driven, never *row*-driven, may
be skipped (`Expr::is_infallible_predicate`, exhaustive with no wildcard arm). Checked
arithmetic, `Div`, a strict `CAST` and every string-*producing* function can raise on one row
and not its neighbour, so their predicates take the old path untouched; the six
boolean-returning string predicates cannot, and are admitted — but only over a column that is
already UTF-8, since evaluating one over `Binary` casts it and *that* rejects a row's bytes one
at a time. The equivalence is pinned by
`crates/bc-expr/tests/short_circuit_filter.rs`, which asserts mask-for-mask agreement with
whole-batch evaluation on every predicate shape above before it reports a timing.

What DuckDB still has here that Batcher does not is the *online* half: `AdaptiveFilter` is a
randomized hill-climb over the conjunct order on measured time, which is what corrects a
static cost model that ordered a cheap unselective predicate ahead of an expensive selective
one. See `competitor_technique_review.md` for the mechanism and the four call sites that need
a per-operator state slot.

### 3. Shuffle is RAM-only, and the fix is written but never called

`PartitionStore` is an in-memory `HashMap` (`bc-transport/src/store.rs:72`). No disk tier, no
spill, no external shuffle service. A dead worker loses its buckets → lineage recompute (re-read
the source, re-run the map — the longest phase). Spark writes shuffle to disk and re-fetches it.

The mitigation **already exists and is dead code**:
- `carbonite/resilience/replication.py::assign_replica_hosts` — implemented, doctested, **zero
  call sites**.
- `dist/flight_worker.py:292::replicate_buckets` — "the core of recompute-free recovery",
  **never called**.
- Every reducer takes `replicas=None`; **no caller ever passes it**.
- `DistributedConfig.shuffle_replication` is defined, validated, and **set to 2 by the spot
  profile** — and *nothing reads it*.

So on a spot cluster today you get recompute, not re-fetch, while the config claims otherwise.
Wiring four call sites converts preemption from recompute to re-fetch.

Related: `clear_plan`/`release` are bound in Rust and called **only** by the streaming pipeline —
so a long-lived batch fleet accumulates every bucket of every stage until OOM. The Rust doc
comment warns about exactly this.

### 4. Streaming is micro-batch, and cannot express Flink's guarantees

Not a dataflow runtime: no operator graph, no barriers, no keyed state backend. One Python driver
thread running `while True: pull → execute_plan → write`.

- **The watermark is a single global scalar** on the driver (`core/streaming.py:201`) —
  `max(event_time) - lateness` over whatever the driver saw. There are **no per-partition
  watermarks and no `min()` across inputs**, so one fast Kafka partition drags the watermark
  forward and every slow partition's rows are then dropped as "late." No idle-source detection.
- **State is one in-memory Arrow batch**, checkpointed by rewriting it *in full* every
  micro-batch, to a **local filesystem only** (`sqlite3.connect`, `os.replace` — it cannot write
  `s3://`, though `resilience="spot"` tells you to). No incremental checkpoint, no changelog, no
  spillable state. The repo *has* an out-of-core spill system; it is not wired to any streaming
  operator.
- **Sliding windows silently fall back** to an unbounded running aggregate with no eviction.
  Session windows do not exist as a streaming construct.
- **The driver is a single point of failure** with no HA and a local-disk checkpoint.

Exactly-once *is* real for Delta (idempotent `txn` actions, genuinely good), but only for Delta.

### 5. The AI moat was shipped disabled — **fixed (the CPU→GPU overlap is now default)**

`dist/streaming/pipeline.py` is real: separate CPU producer and GPU map actor pools, Flight
handoff, credit-bounded window, spot recovery. It was gated behind `stream_inference: bool = False`.

**The default path therefore ran the entire CPU→GPU chain inside one actor** — the GPU actor
held a whole GPU while it decoded JPEGs. "The GPU stays saturated because CPU decode is a
separate concurrently-executing stage" — the actual Ray Data feature, and the thing every
GPU-pipeline benchmark and doc leans on — **was not what a user got**. The Ray guides name this
exact shape as their #1 GPU-utilization bug ("CPU preprocessing starving GPU is #1 cause",
`workloads/batch-inference/`), and it is the one their whole CPU:GPU-node-ratio tuning chapter
exists to work around.

**`stream_inference` now defaults to True**, so the overlap is what a user gets without
configuring anything. It is a pure scheduling change — every stage runs the identical sub-plan
through `core.execute_with_udfs` — and the equivalence is pinned three ways in
`tests/integration/test_stream_inference.py`: against single-node, against the three-stage
(CPU→GPU→postprocess) shape, and against the non-overlapped map itself. A chain with no
resource-class boundary to split at falls back to the embarrassingly-parallel map, so a
homogeneous pipeline is unaffected. `test_stream_inference_is_on_by_default` pins the default,
because a silent revert would cost every batch-inference pipeline its GPU utilization with
nothing turning red.

**Still open on this ceiling:** it splits at exactly *one* boundary (2 stages, the third stage
riding in the consumer); there is no N-stage topology and no per-stage autoscaling.

Also missing for AI: **no variable-shape tensor type** (Ray Data and Daft both have one) and
data never stays resident on the GPU across operators (host round-trip per op).

Video decode *was* the third item here — a per-row Python loop that materialized **every frame
of a clip** to sample 8 of them (a 1-minute 1080p clip is ~1,800 x 6.2 MB ≈ 11 GB resident, for
8 frames of output). **Fixed:** `ml/decode.py` now learns the frame count first and keeps only
the frames it was going to keep, so peak memory is the output plus one frame — measured
**275 MB → 15 MB (18x)** on a 10-second clip, and *flat* in clip length rather than linear. The
output is byte-identical to the retaining implementation, which is pinned by a test that keeps
the old version as its oracle. Still a Python loop, and still per row; what changed is that it
is no longer unbounded.

### 6. Task granularity ≈ node, not partition

Shuffle map tasks = one per node; non-shuffle tasks are clamped to cluster cores. Spark decouples
tasks from executors (routinely 10k–100k tasks/stage), which is what buys dynamic load balancing,
work stealing off a slow node, and fine-grained recovery. Batcher's coarse unit means a straggler's
*entire* partition must be redone, and skew cannot be diluted by over-partitioning.

Compounding it: **speculation is off by default** (`max_backups = 0`), **salting is off**
(`skew_join_salt = 0`), **locality is off** (`locality_aware_scheduling = False`). Out of the box,
resilience is Ray task retries + single-stage lineage recompute + whole-query retry.

## Claims to retire

These are asserted in the repo and contradicted by its own code.

1. **`CLAUDE.md:7-9`: adaptive re-optimization "which Spark AQE's stage-boundary adaptation cannot
   match."** The loop is real (`api/adaptive.py:342-410`, genuinely re-optimizing on *measured*
   cardinalities) — but it **is** stage-boundary adaptation. `api/adaptive.py:1` says so verbatim
   in its own first line: *"Adaptive (intra-query) execution: stage-boundary re-optimization."*
   Same mechanism, same granularity as AQE. It is also **off for every query under 20M input
   rows** (`_ADAPTIVE_MIN_INPUT_ROWS`), so most queries never touch it.
   *Defensible replacement:* "stage-boundary re-optimization like Spark AQE, but available
   single-node too, **plus** a sketch-backed cross-query learned-stats and bandit loop that
   neither DuckDB nor Spark has." That is true, and still interesting.
2. **`BENCHMARK_RESULTS.md`: "beats DuckDB's execution engine on every TPC-H query" (21/21).**
   True only against `duckdb_arrow` — DuckDB forced through an Arrow scan, which strips its zone
   maps, compression *and* dictionary encoding, and the scan cost lands inside the timed region.
   Against DuckDB's **native** store (how every official TPC-H result runs it), `docs/benchmarks/
   vs-duckdb.md` reports **DuckDB faster on 16 of 21, geomean 1.36× in DuckDB's favour**. Both
   numbers are in the tree, and they contradict each other. Lead with the native one.
3. **The "in-memory kernels" table in `vs-duckdb.md`** (filter 28 ms vs DuckDB 1,601 ms = 57×).
   DuckDB does not filter 60M rows 57× slower than Rust. That figure is almost certainly timing
   DuckDB's Arrow scan plus result materialization, not its filter. It is the least credible
   number in the repo and it discredits the real results around it.
4. **`core/executor.py:1-11` references "the `bc-adapt` re-optimization loop."** There is no
   `bc-adapt` crate. The Rust side has no adaptivity at all.

## Defects found and fixed in this pass

All were silent — right answers, or no error, while something load-bearing did not happen.

| Defect | Consequence |
|---|---|
| `io/predicate.py` emitted `{"node":"null"}`; `bc_io::Pred::IsNull` deserializes `"is_null"` | Tag drift → `parse()` rejected the **whole** predicate → any filter containing a null test pruned **zero** row-groups. Silent, because pruning only ever affects speed. |
| `WindowedAggregateProcessor` defined no `snapshot_state` | `has_state()` duck-types on it → reported *stateless* → state never checkpointed **while offsets were still committed**. A crash resumed past consumed data with every open window and the watermark **silently lost**. |
| `KafkaSource._poll` committed offsets at **read** time | Crash between poll and publish → group offset already advanced, and `_apply_seek` was a no-op in subscribe mode → **at-most-once**, the exact opposite of the module docstring. |
| `_is_distributable_aggregate` never checked `plan.watermark` | A watermarked windowed aggregate passed the gate into a distributed runner with **no watermark at all** → silently degraded to an unbounded complete-mode aggregate. **Same query, single-node vs distributed, different results** — a direct breach of invariant #7. |
| No `get_worker_info()` anywhere | `DataLoader(loader, num_workers=4)` — the ordinary thing to write — made every worker replay the full rank shard, so **the epoch saw each sample 4×**. Silent training-data corruption. |
| `compile_regex` had no cache | Every regex *and every `LIKE`* recompiled its automaton **once per 16,384-row morsel** (~3,700× over 60M rows). |
| `IcebergTableSplit.identity()` read a field not in `__slots__` | Raised `AttributeError` unconditionally. |
| `DeltaSnapshot` fallback read `self._table` (no such field) | The safety net that "keeps correctness independent of the optimization" raised `AttributeError` itself. |

### Investigated and **not** a bug — do not "fix" this

The `available_now` / `once` **drain** path in `api/io_namespace/writer.py` returns *before* the
distributed eligibility gate, and `start_distributed_stream_drain`'s docstring claims the caller
"guarantees eligibility (stateless plan, …)" — which the caller does not check. That looks
exactly like the watermark hole that *was* real on the micro-batch path, and the obvious
"fix" is to refuse a watermarked plan here too.

**Don't.** It was checked empirically. A drain reads a *bounded* source once, so the first batch
arrives with the watermark still unset and no row can be behind it; the late-drop never fires.
A watermarked hourly aggregate over `[00:00, 10:00, 00:10]` returns `{00:00: 100, 10:00: 5}` on
the single-node drain — byte-identical to the batch oracle, and to what the distributed drain
(`_write`, the ordinary batch path) produces. Streaming and batch semantics *coincide* on a
bounded single-pass drain. Refusing watermarked drains would be a capability regression with no
correctness basis behind it.

The loose docstring is worth tightening; the behaviour is correct.

### A near-miss worth remembering: the JIT parity guard earned its keep

Mid-audit, `bc-expr/src/eval/binary.rs` switched `numeric::{add,sub,mul}_wrapping` → the
**checked** `numeric::{add,sub,mul}`, so the interpreter began **erroring** on i64 overflow while
the Cranelift JIT still **wrapped** — a live breach of invariant #6 (Tier-1 must be bit-for-bit
identical to the Tier-0 oracle). The same expression would return a wrapped negative or an error
depending purely on whether the JIT happened to compile it. The diff also *deleted the comment
that explicitly warned against exactly this change*.

Nothing about that is visible in a result: it needed a test that runs both tiers on the same
input and compares. `differential_integer_overflow_wraps_identically` is that test, it went red
immediately, and the change was corrected (the JIT now matches the checked interpreter; the full
Rust workspace is green).

The lesson is the one CLAUDE.md already states and this episode paid for: **a tier is only an
optimization if something mechanically proves it agrees with the oracle.** Keep the seq == par ==
JIT differential tests, and treat a comment that explains *why* two implementations must match as
load-bearing code, not commentary.

## The roadmap that would make the claim true

In dependency order. (1) and (2) are the ones that change what Batcher *is*.

1. **Implement the streaming executor RFC.** Removes the sf100 OOM, the gather tax, and the
   window/materialization losses. Without it, "any data scale" is false on one node.
2. **Adopt `StringView` + preserve dictionaries through the kernels.** The other half of the
   single-node gap, and it compounds with (1). *Comparisons are done* (`try_dict_compare`,
   19.6x on the low-cardinality string filter); what remains is `StringView`, keeping the
   dictionary alive through project/join rather than decoding at the leaf, and a dict-aware
   string join key.
3. **Wire the shuffle replication that already exists** (four call sites) and call `clear_plan` on
   the batch path. Converts spot preemption from recompute to re-fetch, and stops long-lived
   fleets leaking every bucket.
4. **~~Default `stream_inference=True`~~ (done), generalize to N stages with per-stage
   autoscaling.** The AI moat is now on out of the box. Remaining: the N-stage topology,
   per-stage autoscaling, a variable-shape tensor type, and keeping data resident on-device
   across ops.
5. **Per-partition watermarks + a real state backend** (spillable, incremental, object-store
   checkpoints). Until then, do not claim Flink parity — and refuse, loudly, the shapes that
   cannot be honoured (the distributed watermark gate now does).
6. **Decouple tasks from nodes**; turn on speculation and skew splitting by default.

Until 1–2 land, the defensible positioning is narrower than the current one, and *still strong*:

> The fastest **distributed** Arrow engine, with an optimizer that learns across runs — beating
> Ray Data by 50–450×, Spark by 13–197×, and Polars at most shapes; on a single node it wins below
> ~10M rows and cedes to DuckDB above it.

That is a claim the benchmarks in this repo actually support.
