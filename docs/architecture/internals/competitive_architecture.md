# Competitive architecture: where Batcher wins, where it loses, and what it must build

**Status:** audit, 2026-07-14; partially re-audited 2026-07-29, 2026-08-01, 2026-08-15,
2026-08-16 and 2026-08-25. Every claim below was checked against code, not documentation. Where the docs and the code disagreed, the
code won and the doc is named as wrong.

**What the 2026-08-25 pass changed — TPC-H sf10 is a win, and JOB is the largest move.**
Measured as an **A/B against the tree itself**: `HEAD` built in a separate worktree, the new
build in the main one, two engines on both arms (`batcher,duckdb`) and the same 96-core /
184 GiB node. Against the figures recorded in the passes below rather than a re-run of `HEAD`,
two rows would have been credited with drift they did not earn — the operator mix reads 0.740
in the 2026-08-23 entry and **0.690 on `HEAD` today** — so every number in this pass is a
same-day pair.

| suite | HEAD | after |
|---|---:|---:|
| **TPC-H sf10 (22)** | 1.087 | **0.963 — W** |
| **JOB (113, real IMDb)** | 1.265 | **1.112**, 34 → 44 wins |
| TPC-H sf1 (22) | 0.782 | **0.742** |
| TPC-DS sf1 (99) | 0.945 | 0.947 (total 3,408 → **3,256 ms**) |
| ClickBench (43) | 0.630 | **0.620** |
| operator mix (21) | 0.690 | **0.678** |
| H2O `join` (5) | 0.697 | 0.695 (total 555 → **476 ms**) |
| H2O `groupby` (10) | 1.006 | 1.025 |

**Two rows below change their verdict, and one of them is this document's oldest reservation.**

* **TPC-H sf10 (60M-row `lineitem`) is no longer a loss.** 1.087x → **0.963x**, suite total
  2,938 → 2,323 ms, carried per query rather than by a geomean: q9 456 → 233 ms, q13 325 → 174,
  q5 189 → 122, q3 116 → 87, q4 117 → 96, q10 158 → 139. The `≥100M rows` row keeps its **L**
  on the sf100 evidence — 600M rows still OOMs q3/q4/q5 and that is untested here — but the
  sentence locating the boundary *at sf10* is retired.

* **JOB moves further than any other suite: 1.265x → 1.112x, and Batcher's total drops below
  DuckDB's** (8,131 ms against 8,885). The geomean stays above 1 because it wins the large
  queries and still loses many small ones — q17f 298 → 75 ms, q10c 185 → 46, q8d 188 → 69,
  q30a 172 → 106 — so quote both or the row reads as a loss it no longer straightforwardly is.
  This is the suite the design is aimed at, and it is the one that moved most.

**Why they moved, in one line each** (`benchmarks/BENCHMARK_RESULTS.md` carries the
measurements): a probe-side bloom that allocated one full-size filter per build shard and merged
them serially — 34.3 ms against the 31.3 ms parallel hash insert beside it — is now sharded the
way the heads are and merges nothing; the two fitted constants that existed only to compensate
for that cost are gone, which is what lets a multi-join spine keep every core instead of
collapsing to one; an ordered group key now uses the key-disjoint partition its own layout
already provides instead of gathering the relation into hash buckets; and the group-count
estimator no longer reads a *clustered* key as a thousand-value domain when it holds fifteen
million.

**Read JOB's per-query deltas as noise and its total as signal.** 113 queries share one process,
one learned-stats hub and one memory pool, so a query's time depends on what ran before it: a
full run shows q25a at 83.7 → 264.5 ms, and run on its own the whole q25 family is *faster* on
the new build (q25a 135.5 → 72.7, q25b 95.3 → 63.5, q25c 109.5 → 82.7).

**What the 2026-08-16 pass changed.** Re-measured every suite on the same 96-core / 184 GiB
node, each on a **two-engine** lineup (`batcher,duckdb`) so no third engine's resident memory
perturbs the timing — a four-engine run reads H2O `join` at 1.83x against 0.93x on two, so the
lineup is part of the measurement and is now stated with every figure.

Against DuckDB's **native compressed store**: JSON 0.245x (5/5), ClickBench 0.636x (28/43),
operators 0.656x (11/19), TPC-H sf1 0.789x (16/22), H2O `join` **0.928x (3/5, now a win)**,
TPC-DS 0.963x (38/98), H2O `groupby` **1.191x** (was 1.28x), JOB **1.285x** (was 1.37x). Six
of eight.

Against DuckDB on the **same zero-copy Arrow** — the bar `methodology.md` designates the
like-for-like execution comparison — **every suite that can run it is a win**: JSON 0.039x
(5/5), ClickBench 0.072x (43/43), H2O `groupby` 0.089x (10/10), H2O `join` 0.244x (5/5), TPC-H
0.256x (22/22), operators 0.362x (15/19). TPC-DS still has no figure on that bar — DuckDB
over registered Arrow views is SIGKILLed on q64 (see `engines/lineup.py`).

**JOB now has one, and it is the weakest bar in this paragraph.** All 113 queries run on a
three-engine lineup: **0.404x, winning 106 of 109** against the same Arrow, against 1.221x on
the same run versus the native store. Read it with the reservation this sentence used to state
as a reason for having no figure at all: JOB exists to measure *join ordering*, and DuckDB over
registered Arrow has no storage statistics to order a many-way join with — so part of that 0.404
is a planner deprived of its inputs rather than an executor beaten on even terms. What it does
establish is the direction of the residual: on identical input Batcher is not losing JOB, and
the 1.221x is against DuckDB's storage engine. (The two-engine lineup this scorecard prefers
reads JOB at 1.308x; the lineup is part of the measurement, as the 2026-08-16 note says.)

Two rows below move, and neither claim in them is retired:

* **H2O `groupby` (1.28x -> 1.191x)** — still L against the native store, and still for the
  reason recorded below (dictionary-encoded keys). What changed is the *composite* key: each
  byte column's distinct values are now numbered in first-seen order and the ranked columns
  take the ordinary integer grouper, which took a two-string-key group-by over 10M rows from
  41.7 ms to 32.6 ms — identical to the same query with two `int64` keys, so nothing about the
  strings is left to pay for. The suite's residue is its two single-key 100,000-distinct
  queries (q3, q7), which is the dictionary case exactly.
* **JOB (1.37x -> 1.285x)** — still L, and this pass located *why* the cross-run loop is not
  answering it: `record_cardinality_outcome` files a measured row count under the plan **as
  written**, whose root for any `SELECT ... GROUP BY` is a `Project` — a node type excluded
  from the estimator's `_CORRECTABLE` set. Nine lookups, nine misses over three rounds of
  three h2o group-bys, while the writer wrote the true count each time. Correcting the key
  **regresses** the suites (TPC-DS 0.965 -> 0.988, JOB 1.252 -> 1.293), because every
  threshold downstream was calibrated while that input was ~10x low. Recorded in
  `BENCHMARK_RESULTS.md` and pinned as a strict xfail in
  `tests/unit/test_learned_rows_scope.py`; it must land with that recalibration, not before.

**What the 2026-08-15 pass changed**, measured on a 96-core / 184 GiB node rather than the
16-core one the older rows were taken on, so read it as a re-measurement and not only as
movement. Against DuckDB's **native compressed store** — the harder of the two bars, and the
one this scorecard's "single-node" rows are about — Batcher now leads TPC-H sf1 (0.76x, 17 of
22), **the full 99-query TPC-DS** (0.96x, 44 wins, where it read 1.13x at the start of that
day and 1.51x a week before), ClickBench (0.62x, 30 of 43), JSON (0.25x), the operator mix
(0.64x) and the H2O.ai `join` task (0.95x). Two rows below are now understated by it, and one
is confirmed:

* **Single-node ≤10M rows (vs DuckDB): W** — confirmed, and by more than recorded.
* **Single-node ≥100M rows (vs DuckDB): L** — still L on the sf100 evidence (600M rows, where
  q3/q4/q5 OOM), but **the sf10 half of this row is retired**: TPC-H at sf10 (60M-row
  `lineitem`) is a **win** as of 2026-08-25, 0.963x against 1.087x on the same day's `HEAD`.
  The sentence below recorded it as a loss and is kept for the history of the number.
  Nine of thirteen shapes still scale *sublinearly* from sf1 to sf10; four do not (q5 14.9x,
  q13 12.7x, q18 12.5x, q9 11.2x), and those four carry the highest-cardinality group-bys and
  the largest intermediates in the benchmark.

  **The sf10 figure is 1.07x, not the 1.27x recorded here** (2026-08-23): six runs across two
  passes read 1.227 / 1.225 → 1.118 / 1.115 → 1.066 / 1.078, with 8-10 of 22 won. What moved is
  in `benchmarks/BENCHMARK_RESULTS.md`; q18, q20, q21, q7, q3 and q9 carry most of it. Read the
  row as *still a loss, and a much smaller one*.

  **TPC-DS sf1 moved with it, and further: 0.970 → 0.918**, 43 of 99 won, and the Join Order
  Benchmark 1.265 → **1.185** (40 of 109). Two thirds of what remains at sf10 is three queries
  — q9, q21, q13 — and they profile as one thing each: a high-cardinality grouped aggregate and
  a join-output gather.

  **How much of the residual is storage rather than execution is now answerable**, and the
  answer is: nearly all of it. Against `duckdb_arrow` — DuckDB executing the same zero-copy
  Arrow — the same sf10 run is **0.297x, and the only query Batcher loses is q13** (1.84x).
  JOB reads the same way: 1.185x against the native store and **0.383x against the same Arrow**,
  106 of 109.
  So the sf10 gap against the native store is DuckDB decompressing a dictionary/RLE column
  where Batcher reads full Arrow, which is invariant #3 and not a kernel deficit. State the
  row with both numbers or it reads as an execution loss it is not.
* **The Join Order Benchmark now completes.** This document's ceiling on it was recorded when
  two runs were OOM-killed; all 113 queries now run with none killed (geomean 1.37x, 31
  wins), and `job-q7c` — the query that took the process down — is a **win** at 291 ms against
  DuckDB's 504. That is not a claim to have fixed the OOM: it was a 30 GiB box then and a
  184 GiB one now, and peak RSS here is 15-22 GiB.

Unchanged and still true: H2O.ai `groupby` loses to the native store (1.28x) because its keys
are low-cardinality strings held dictionary-encoded there and read as full Arrow `Utf8` here —
on identical Arrow input the same suite is **0.11x, winning 10 of 10**.

**What the 2026-08-01 pass changed:** ceiling 3 no longer says the shuffle is RAM-only — a byte
cap and a disk spill landed in `24d89d9e`/`f18d6a36`, so the memory half is closed and only the
survivability half (no external shuffle service) remains, which is why the scorecard row is
still **L**. The rule count was stale at 375; the live registry holds **725** as of 2026-08-08 (the `@rule` decorator count undercounts, because a `_register(...)` loop produces several rules from one decorator). It drifts as rules land, so read it from `DEFAULT_REGISTRY.rules()` rather than from this line; the seven phases it spreads over are `NORMALIZE` (545), `REWRITE` (83), `PUSHDOWN` (68), `FUSION` (23), `SELECTION` (3), `ENFORCE` (2) and `JOIN_REORDER` (1).

**What the 2026-07-29 pass changed**, so a reader knows which numbers are current:
ceiling 3 (shuffle replication) is now wired to all four shuffles rather than one; ceiling 8
(first-seen query latency) was re-measured and is **3-5x better than recorded**, so it is no
longer the largest latency item; claim 4 in [Claims to retire](#claims-to-retire) is fixed.
It also retired a claim this document did not make but the research paper did — that
high-cardinality grouping is a systematic kernel gap. Measured, plain high-cardinality
grouping wins by 7-11x; the loss was grouped `COUNT(DISTINCT)`, and it is closed.
Everything else here is the 2026-07-14 reading.

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
| Small-query latency | **= on a repeated shape** (2x faster), **L on a first-seen one** (2.8x slower — 8 ms of optimizer, twice; ceiling 8) | = | **W** | — | **W** | **W** |
| Single-node ≤10M rows | **W** | **W** | **W** | — | **W** | — |
| Single-node ≥100M rows | **L** (2–11×, **OOM** on q3/q4/q5 at sf100) — but the boundary is **above sf10** as of 2026-08-25: 60M-row TPC-H is **0.963x, a win** | **L** on 6 shapes | — | — | **W** | — |
| Distributed batch | **W** | **W** | = | — | **W** (50–450×) | L |
| Optimizer breadth | = (722 rules, bushy DP join order) | **W** | **W** | — | **W** | L |
| Range / inequality joins | **W below 1M** (2.6–3.0x at 10K–100K, 1.5x at 500K), **= at 1M**, **L above** (0.73x at 2M, 0.44x at 5M) — ceiling 7 | — | — | — | — | — |
| Learned/adaptive | **W** | **W** | **W** | — | **W** | = |
| String execution | **L** (no StringView, dict decoded at leaf) | **L** | = | — | L | L |
| Streaming guarantees | — | — | L | **✗** | — | — |
| Fault tolerance | — | — | **L** (no external shuffle service; buckets die with their worker) | L | = | L |
| Skew handling | — | — | **=** (AQE splits on a size rule; Batcher salts on measured hot keys, no opt-in) | — | = | L |
| AI / GPU pipelines | — | — | — | — | **L** by default | — |
| Lakehouse formats | — | — | L (all via pyiceberg/delta-rs) | — | = | L |
| OLTP *interoperability* (drive a database) | — | — | **W** (row-level DML as a mode; Spark JDBC has append/overwrite only) | — | **W** (neither writes a database) | — |
| OLTP *engine* (be the database) | **✗** | **✗** | **✗** | — | **✗** | **✗** |
| OLTP *read speed* | **✗** — 369x a raw driver per query, **263 QPS/process**, GIL-bound | — | — | — | — | — |

## OLTP: what Batcher does and does not have (verified in code, 2026-08-18)

This row is split in two on purpose, because the two halves have opposite answers and the
distinction is the whole of the honest claim.

**Batcher is not an OLTP engine and cannot become one without a storage layer it does not
have.** It owns no storage. It has no transaction manager, no MVCC, no lock manager, no
write-ahead log, no crash recovery, and no index it builds or maintains. Two Batcher
processes writing the same target coordinate through *that target*, not through Batcher.
Nothing in `crates/` or `python/batcher/` implements any of those, and nothing on the
roadmap proposes to: the engine's unit of work is a plan over Arrow batches, not a row in a
page. A claim that Batcher "supports OLTP" is false in the sense that matters — you cannot
point an application's transactions at it.

**What it does have is OLTP interoperability, and there the comparison is favourable.** It
can drive an OLTP system as a source and as a sink:

- *Reads.* Predicate and projection push into the submitted SQL; a row cap and a top-N push
  where the dialect can spell them (`io/formats/sql/dbapi/source.py`,
  `adbc/source.py`). On a partitioned store a predicate that pins the partition key stops
  the fan-out entirely — DynamoDB issues one `Query` rather than N `Scan` segments billed
  per item *examined*, and Cassandra reads one partition rather than 64 token ranges
  (`io/formats/nosql/{dynamodb,cassandra}.py`).
- *Writes.* `append` / `overwrite` / `upsert` / `update` / `delete` / `delete_insert`, one
  transaction per call, executed **by the database** as `ON CONFLICT` / `ON DUPLICATE KEY`
  / `MERGE` (`io/formats/sql/dbapi/`). Spark's JDBC writer has save modes only; an upsert
  there is `foreachPartition` and hand-written SQL. Ray Data and Daft write no database at
  all.
- *Streaming.* A keyed micro-batch write is exactly-once without a transaction log, because
  replaying it writes the same keys to the same values.

The transactional guarantee is therefore **the target's, scoped to one call**. There is no
cross-shard and no cross-call transaction: a distributed write is one transaction per shard,
which is safe for the keyed modes because a shard only ever touches the keys its own rows
name, and which is why `overwrite` is refused past the first shard. Where cluster-wide
atomicity is the requirement, a lakehouse commit is the mechanism that has it.

**Speed settles it, and throughput settles it faster than latency does.** Measured against
raw `sqlite3` over the same 50,000-row table with an `INTEGER PRIMARY KEY`
(`BENCHMARK_RESULTS.md`, 2026-08-18, release engine):

| | Batcher | raw `sqlite3` | ratio |
|---|---|---|---|
| Point lookup, **different key each call** | 3.555 ms | 0.010 ms | **369x** |
| Point lookup, same key each call | 2.756 ms | 0.010 ms | 276x |
| 1,000-row range scan | 4.475 ms | 0.607 ms | 7.4x |
| 10,000-row upsert | 21.8 ms | 7.6 ms | 2.85x (459,189 rows/s) |

| Throughput | QPS |
|---|---|
| Batcher, 1 / 8 / 16 threads | 225 / 320 / 312 |
| Batcher, 8 processes | 2,106 (**263 per process**) |
| Batcher **tuned** (`observability.event_log=False`), 1 thread | 374 |
| raw `sqlite3`, 1 thread | **5,205** |

The default event log costs 62% of the empty-query floor and 16% of a real point lookup, so
the best supported configuration for this shape is 2.93 ms and ~374 QPS per process. That is
still 14x short on throughput and ~290x on latency.

Sixteen threads buy 1.4x and plateau; processes scale perfectly linearly. The ceiling is the
**GIL** — the control plane is Python, and plan construction, optimization, IR serialization
and routing all hold it. One raw driver thread out-serves Batcher across sixteen threads by
16x.

**Three quarters of it is the architecture, and the remaining quarter does not close the
gap.** A `Dataset.to_pydict()` over a **one-row in-memory table with no operators** costs
**1.9 ms** against pyarrow's own 0.0077 ms — the price of entering the engine at all. Of that,
the engine itself (`execute_plan_metered`) is 17%; stubbing out every avoidable control-plane
cost found — the event-bus close-out (0.32 ms) and Carbonite's per-query resource re-sizing
(0.15 ms) — takes the floor to 1.43 ms, i.e. 527 -> 700 QPS per process against a single raw
driver thread's 5,205. Worth collecting; not a path to OLTP. `count()` on the same table is
0.078 ms precisely because it is answered from metadata *without entering the engine*.

The designed mitigation does not reach this shape either: `orchestration/prepared.py`
memoizes the whole derivation, but its entries are created only by `fast_path`, whose gate
admits **resident Arrow sources only** and which is off by default. A database query is never
resident.

So the honest bound is: a fixed ~2 ms cost amortized over a large result, which is what an
analytical engine is for. At one row that cost *is* the query. Closing the gap would mean a
second terminal-op path that never builds a `LogicalPlan`, never crosses the optimizer and
never leaves Python for one row — a different engine wearing the same API. **Do not treat it
as a tuning backlog.**

On files rather than a database, `ds.filter(col("id") == 5)` is a *pruned scan* — zone maps,
then a bloom filter for the equality that lands inside `[min, max]` — not an index seek.

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
| Image decode → resized `uint8` tensor | `bc-expr/.../media/image/mod.rs::to_tensor` | DCT-scaled JPEG + SIMD resize |
| Image decode → **normalized `f32` tensor** (`/255`, per-channel mean/std, HWC/CHW) | `image/mod.rs::to_tensor_f32` | the torchvision `ToTensor`+`Normalize` step, in-engine |
| Image **center-crop** (torchvision-style zero-pad) | `image/mod.rs::center_crop` | the crop half of resize→crop→tensor |
| Image **grayscale** (Rec.601 luma → 1 channel) | `image/mod.rs::to_grayscale` | color-convert for 1-channel models |
| Image **geometry**: rotate (right angles), flip H/V, pad-without-scaling, letterbox | `image/transform.rs` | the augmentation half; `letterbox` is the detection preprocessing |
| Image **photometric**: brightness/contrast/saturation/hue, blur, sharpen, invert, posterize, solarize, equalize, autocontrast | `image/transform.rs` | `PIL.ImageEnhance` + AutoAugment conventions, so a torchvision policy ports unchanged |
| Perceptual hashes for image dedup: **dHash, pHash (DCT), aHash** | `image/mod.rs::dhash`, `image/hash.rs` | Hamming-thresholded near-dup; block on `ahash`, confirm with `phash` |
| Image **curation measures**: brightness, sharpness, entropy, colourfulness, mean colour, is-grayscale | `image/quality.rs` | the screens that find rows which decode perfectly and teach a model nothing |
| Image **header facts**: aspect ratio, alpha, container (magic-byte sniffed) | `image/probe.rs` | one header read, no pixels |
| Audio decode / mono / resample (sinc) | `media/audio.rs` | WAV/FLAC (symphonia) |
| Audio **level and hygiene**: rms, dBFS, peak dBFS, clipping ratio, silence ratio | `media/level.rs` | recording-quality triage as a predicate |
| Audio **shaping**: rms-normalize, pre-emphasis, `pad_or_trim`, slice, **WAV encode** | `media/level.rs` | `pad_or_trim` is what makes a clip corpus batchable (Whisper's fixed 30 s) |
| **Linear spectrogram** and spectral descriptors (centroid, rolloff, bandwidth, flatness) | `media/spectral.rs` | rolloff finds the band-limited-then-upsampled recordings nothing else sees |
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
  Batcher's image surface **covers Daft's and now goes past it**: alongside center-crop,
  offset `crop`, `encode` (png/jpeg/bmp/gif) and `convert` over the full
  `L`/`LA`/`RGB`/`RGBA` palette, the geometry, photometric, hashing and curation families in
  the table above have no Daft equivalent — a Daft user writes each of them as a per-row PIL
  UDF. Header facts come from **one** read where Daft spends a call per fact. Measured on
  that difference: the entropy + pHash + flip pass runs **5.7x** faster than the Pillow loop
  it replaces (`benchmarks/scenarios/image_decode.py --suite curate`, 96 cores, release
  build). On the shared ingest path — 2,000 JPEGs decoded and resized to 224², the operation
  both engines have natively — Batcher leads by **1.9x** after two read-path fixes (a
  header parse that ignored the projection, and local reads fanned across a thread pool
  that made them 2.5x slower than a serial loop); it was 1.3x before them. It leads on **audio** by a wider margin: Daft has **no native audio surface at
  all**, where Batcher has 23 operations including torchaudio-matched mel-spectrogram/MFCC.
  The variable-shape tensor gap this bullet used to record is closed too: mixed-resolution
  arrays are carried as a column (`io/formats/ml/ragged.py`), so the two gaps it named are
  both gone. Daft still leads on decoding *into* that type from more source formats.
- **Ray Data** has good CPU preprocessors (scalers/encoders/imputers) and a tensor type, but
  multimodal decode is a torch/PIL **UDF per batch**, not an engine expression; **no native
  mel-spectrogram**; fuzzy-match/minhash are user code. That shows up in the one comparison
  both engines can run natively: `read_images` at 724-747 img/s against Batcher's
  4,649-4,788 on the same 2,000-frame corpus, **6.4-6.6x**.
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
first. This does **not** change the GPU-execution verdict (ceiling #5) or close the AI gap called
out next: **no ML-in-SQL** (BigQuery leads). Do not let a preprocessing-coverage win be read
as a GPU-pipeline win.

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

**A hole in the claim above, found and closed 2026-07-24.** "Peak memory is the breakers' state
plus one morsel per worker, a constant, independent of the input size" was **not true when a
join fanned out**. The streaming probe was `stream.map(|morsel| gather_join_output_with(...))`:
one input morsel produced exactly **one** output `RecordBatch`, however many rows that was. A
16,384-row probe morsel against a 20,000-row build side with a single distinct key produces 327
million rows, and that was emitted as one batch — measured at **13.1 GB RSS in a fresh process**
with streaming on by default.

It was never a cartesian-only problem. Any high-fan-out join hit it, including an ordinary
equi-join on a skewed key: a build side holding 100,000 duplicates of one key turns each probe
row carrying it into 100,000 output rows in the same single batch. The bound does hold for
row-preserving and row-reducing operators, which is most of them, and that is why this survived
the 46-test oracle suite — every one of those tests is correct, and none of them fans out. A
correctness suite cannot see a memory property.

**Fixed:** the probe now morselizes its *output* rather than assuming one morsel in implies one
morsel out, emitting as many `DEFAULT_MORSEL_ROWS`-bounded batches as the fan-out requires. The
common 1:1 foreign-key case is untouched — when the whole result fits one morsel it is gathered
from the unsliced indices, so `gather_join_output_with`'s identity-permutation fast path (which
replaces a full column copy with an `Arc::clone`) still fires and that path is byte-for-byte what
it was. Metrics keep their contract: probe `rows_in` is attributed to the first chunk only, so
chunking cannot inflate the input count, while `rows_out` accumulates. The `peak_bytes` counter
becomes *true* rather than merely defined — its doc comment always said "a pipeline operator
holds one morsel at a time, so its peak is the largest morsel it ever produced".

Measured on the same cartesian query, in two steps. Morselizing the *output* gave
**13,139 MB → 5,179 MB and 17,104 ms → 10,287 ms**; the residual was the `JoinIndices`
themselves, two `u32` arrays over every output row, which only slicing the probe on the *input*
side can bound. Adding that (sized from the fan-out the previous slice measured) gave
**5,179 MB → 728 MB and 10,287 ms → 5,360 ms**. Together: **13,139 MB → 728 MB (18x) and
17,104 ms → 5,360 ms (3.2x)**, identical results throughout. The time came free with the memory
— the predicate above the join now runs over cache-resident morsels instead of a 10 GB batch. `a_high_fanout_join_emits_bounded_morsels`
pins the property against a skewed equi-join, asserting both the morsel cap and equality with the
materializing oracle.

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
the old dictionary path was worse still, paying a full decode on top of that 144.9 ms.

**Do not quote the 19.6x outside this kernel comparison.** `competitor_technique_review.md`
item 6 re-measured it end to end and retired it: a plain-string filter over 6M rows costs
**3.8 ms**, not the 144.9 ms this ratio divides into, and the boundary change that would have
let a dictionary reach the kernel from Python was built, benchmarked at **1.2x-2.8x on
consuming shapes only**, and reverted for regressing `SELECT <string col>` to 0.63x. So no
dictionary reaches the engine from Python on a live path, and the figure describes a kernel
in isolation rather than anything a query can currently obtain. It is


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

### 3. Shuffle durability — the memory half is closed, the survivability half is not

**This section said "RAM-only, no disk tier, no spill" and that is now wrong** (corrected
2026-08-01, against the code). `PartitionStore` carries a byte `cap` and a `spill_dir`, and a
bucket past the cap is written out rather than held: `Body::Memory` is one of two shapes a
published partition can take (`bc-transport/src/store.rs`, `PartitionStore::with_cap`, landed
in `24d89d9e` / `f18d6a36`). The store also keeps its own `retained` byte total, which is what
made the shuffle visible to memory accounting at all — Carbonite's pool tracks reservations the
engine *asks* for, and a published bucket is never asked for.

What that fixed is **memory**: the shuffle can no longer grow without bound on a worker.

What it does not fix is **survivability**, which is what the Spark comparison is actually
about. The spill directory is worker-local scratch, so a dead worker's buckets die with it
whether they were resident or spilled; only replication (below) makes a bucket outlive its
host. Spark's external shuffle service decouples the two — the buckets survive the executor —
and Batcher has no equivalent. So the row in the scorecard stays **L**, for a narrower reason
than it used to give.

The mitigation is now **wired to every shuffle** (this section previously said "dead code",
then "exactly one caller"; both are out of date):
- `carbonite/resilience/replication.py::assign_replica_hosts` and
  `dist/flight_worker.py::replicate_buckets` are called by
  `dist/shuffle_replication.py::replicate_shuffle_output`.
- Its callers are now **aggregate (flat reduce *and* combiner tree), join, sort, and window**.
  A join publishes two shuffle stages (left on 0, right on 1) so its replica is all-or-nothing
  across both — a half-copy would silently under-join rather than raise.
- A dead worker used to force a recompute of its sources *even when a replica held them*,
  because `run_bucket_reduce` treated "the reducer host died" and "its mapped output is gone"
  as one event. With `workers == n_buckets` that is every kill, so replication did nothing on
  exactly the small clusters it was testable on. The regeneration is now skipped when a copy
  survives on a live peer (at most once per source, so an unreachable replica still falls
  through to a recompute rather than exhausting the attempt budget).
- `DistributedConfig.shuffle_replication` still defaults to **1**, rising to 2 under the
  `spot` resilience profile — an on-demand cluster pays no copy.

So on a spot cluster you now get re-fetch recovery for every shuffle. What is still missing
is the rest of the durability half: there is no external shuffle service, so a bucket cannot
outlive its worker except by replication, and inside the combiner tree only the leaf partials
are copied — an interior combiner's output lives on one node, so a loss there still costs a
recompute round.

**Bucket eviction: FIXED (2026-07-26).** `clear_plan`/`release` were bound end to end through
Rust with **zero production call sites** — only the streaming pipeline released anything — so a
session-scoped fleet (`reuse_session_fleet`, on by default) accumulated every bucket of every
stage of every query until the node OOMed. Now evicted at two points: `dist/fleet/eviction.py`
from `query_shuffle_scope`'s exit (cross-query), and `FlightMaterializedSource.cleanup()` before
its `_actors is None` early return (intra-query, the borrowed-fleet case that was the leak).
`bc-transport::clear_plan_shared` frees the matching `/dev/shm` files, which nothing freed at
all — tmpfs is RAM-backed, so that was a second leak on the same node. Regression coverage:
`tests/integration/test_shuffle_bucket_eviction.py`, keyed on `ShuffleSession.partition_count`,
which is the only thing that can observe a leak before it is fatal.

### 4. Streaming is micro-batch, and cannot express Flink's guarantees

Not a dataflow runtime: no operator graph, no barriers, no keyed state backend. One Python driver
thread running `while True: pull → execute_plan → write`. That model is the ceiling, and it has
not moved. Four specific defects underneath it *have*, so the section separates what is fixed
from what remains structural.

**Fixed — the watermark is per-partition, and takes the minimum.** It was one global scalar,
`max(event_time) - lateness` over whatever the driver saw, so one fast Kafka partition dragged
the frontier forward and every slow partition's rows were then dropped as "late" — silently, the
only trace a window total that came out short. It is now `plan/streaming/tracker.py`: per-partition
event-time maxima and a watermark that is their **minimum**, which is the strongest claim a
multi-partition stream can actually support. Partitions are named by the source
(`BrokerSource.watermark_partition_columns = ("topic", "partition")`) and the windowed-aggregate
driver widens Kyber's projection to keep those columns alive, stripping them again before the
batch reaches the plan. A minimum stalls on a silent partition, so it is paired with Flink's
answer — `streaming.watermark_idle_timeout_seconds`, a documented trade rather than a free fix,
and settable to zero for the fully conservative frontier. The source's *expected* partition set
holds the minimum back at startup so it is not taken over a subset. Per-partition state
checkpoints with the fold, because restoring only the frontier would let the first partition to
speak after a restart set the minimum on its own. The dedup, the interval join and the session
window share the same tracker; they consume a pipeline's output rather than the source, so they
get per-partition attribution only where the pipeline preserves the columns, and degrade to the
old maximum where it does not. There is still **no watermark propagation across operators**,
because there is no operator graph to propagate along.

**Fixed — sliding windows are windowed.** `window(ts, w, slide)` is exploded before grouping, so
the aggregate's group key is a plain column with the geometry two nodes below it, and the driver
— which recognized only a `window_start` key — fell through to the *unwatermarked* running
aggregate. Nothing was dropped as late, nothing was evicted, and nothing was emitted until a
source that never ends ended. `_window_key` now reads the `Unnest`/`WindowBuckets` shape and the
eviction sweep keys on the **hop** rather than the width. The remaining un-windowable shape — a
watermark on an aggregate with no event-time group key — is refused over an unbounded source
instead of silently degrading. Session windows *do* exist as a streaming construct
(`api/terminal/stream/session.py`); that line was stale.

**Fixed — the checkpoint can be written where the durability advice says to write it.** The store
called `os.makedirs` and `sqlite3.connect` on the raw location, so `s3://bucket/ckpt` created a
local directory named `s3:` — while `resilience="spot"` warned that a node-local checkpoint is
lost with the node and told the caller to use `s3://`. The advice and the implementation
disagreed, silently. A remote location now uses one immutable file per batch id through the
`io.filesystem` façade (`checkpoint/fs_logs.py`, Spark's design, atomic as a single PUT); a local
one keeps SQLite plus the fsync-then-rename-then-fsync-the-directory ordering, which `pyarrow.fs`
cannot express and which the commit ordering depends on. `is_local_location` is the one answer
both the warning and the store dispatch on, and
`tests/unit/test_checkpoint_log_conformance.py` drives both log backends through the same
sequence so they cannot drift.

**Still open, and structural:**

- **The windowed aggregate now spills; the other stateful operators still do not.** A
  watermarked windowed aggregate reaching `memory.streaming_state_max_bytes` used to raise —
  on the shape that reaches it most legitimately, an open set `allowed_lateness / hop` windows
  wide with a row per group key, behaving exactly as designed and simply large. It now moves
  its oldest windows to disk (`core/streaming/spill.py`), splitting on the median window start
  so each pass halves resident state and leaves the newest windows — the ones incoming rows
  land in — in memory.

  What makes this cheap is the property this operator has and a general state store does not:
  **the watermark only moves forward**, so windows are evicted in increasing order and a
  spilled window is read back exactly once, never sought into. That turns the hard keyed
  random-access problem into an ordered run of Arrow IPC files. Correctness rests on the same
  invariant as everything else here — the runs hold *partial* state and `combine` is
  associative and commutative (#7) — so a late row landing in an already-spilled window is not
  a special case: it folds into memory and meets its spilled half at eviction.
  `tests/integration/test_streaming_state_spill.py` pins the spilled answer against the
  unspilled one, and pins that the same query *without* the tier still raises, so the equality
  cannot pass vacuously.

  Snapshots became multi-part to match: a spilled fold's state is larger than the cap by
  construction, so the store streams the resident state and each run into one IPC file with a
  single part resident at the peak, and recovery combines them back. The local tier streams;
  the object-store tier still buffers, because a PUT needs the whole object.

  **The windowed aggregate now writes a changelog too**, which is what keeps a *spilled* fold's
  checkpoint affordable — otherwise it rewrote a state deliberately larger than the cap on
  every epoch. It removes state, which is normally disqualifying, and qualifies on a property
  worth stating because it generalizes: **eviction drops a prefix of a totally ordered axis**,
  so the entire tombstone set compresses to one integer. Replay combines the partials and
  re-applies the bound; on a whole snapshot the bound is a no-op, so restore does it
  unconditionally rather than branching. 3.0x fewer checkpoint bytes at 40 micro-batches,
  4.4x at 80. `tests/integration/test_streaming_window_changelog.py` pins it against a
  negative control that strips the bound and asserts the windows *do* come back — without
  which the positive test could pass for the wrong reason.

  **Still open:** the *unwindowed* running aggregate and keyed state still raise at the cap.
  Neither has a monotone eviction order to exploit — the running aggregate finalizes every
  group every epoch, and keyed state expires arbitrary keys at arbitrary times — so both need
  a genuinely keyed store with point lookups rather than an ordered run, and keyed state needs
  a per-key tombstone its changelog cannot carry. That is a different piece of work.

  **The changelog half is done, for the operator it mattered most for.** A running aggregate
  — the one with no watermark, so nothing evicts and the state only grows — now records the
  *partial* each micro-batch folded in rather than rewriting the whole state
  (`core/streaming/folds.py::_AggFold.take_delta`, written by
  `checkpoint/state_store.py::snapshot_delta`, replayed by `restore_chain`). It is sound for
  exactly the reason the mergeable algebra exists: `combine` is associative and commutative
  (invariant #7), so a base plus every partial after it *is* the state. Measured over a run,
  whole-snapshot cost is quadratic in the epoch count and the changelog is linear — 3.4x
  fewer bytes at 50 micro-batches, 8.2x at 400, and rising.

  Two limits are deliberate and should not be read as oversights. A delta is written only
  when it is at least twice as small as the state, so a stream that touches every group every
  batch keeps whole snapshots and can never be worse off. And **only an operator whose state
  never shrinks may use a chain**: the windowed aggregate, the watermark dedup and keyed state
  all evict, a chain cannot express a removal, and replaying one would resurrect what they
  dropped. That is opt-in per processor
  (`streaming_query/processors.py::AggregateProcessor.snapshot_delta`) and pinned by
  `tests/integration/test_streaming_state_changelog.py`, because nothing else would catch it
  — a resurrected window is a wrong number, not an error.
- **The driver is still a single point of failure.** `_run_resilient` restarts from the last
  committed checkpoint on a transient fault, which covers a preempted worker or a dropped
  connection *in process*; it does not cover the process. There is no standby and no leader
  election, so a driver that dies needs an external supervisor to restart the query — which will
  resume correctly from the checkpoint, and only if the checkpoint is somewhere the supervisor's
  next host can read.

Exactly-once *is* real for Delta (idempotent `txn` actions, genuinely good), for the file sink
by batch position (`resume=True` skips a `part-batch<id>` already present), and — by a different
route — for a **keyed** write to a database or operational store: `mode="upsert"`/`"update"`/
`"delete"` writes the same keys to the same values, so a replayed micro-batch is a no-op with no
transaction log involved (`TransactionalStreamSink._KEYED_MODES`). Everything else — Iceberg,
Hudi, and any `mode="append"` — gets an ordinary append, and `TransactionalStreamSink.open()`
warns once that a replayed epoch writes its rows twice.

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

Also missing for AI: data never stays resident on the GPU across operators (host round-trip
per op). The variable-shape tensor type this paragraph used to name is now implemented.

Video decode *was* the third item here — a per-row Python loop that materialized **every frame
of a clip** to sample 8 of them (a 1-minute 1080p clip is ~1,800 x 6.2 MB ≈ 11 GB resident, for
8 frames of output). **Fixed:** `ml/decode.py` now learns the frame count first and keeps only
the frames it was going to keep, so peak memory is the output plus one frame — measured
**275 MB → 15 MB (18x)** on a 10-second clip, and *flat* in clip length rather than linear. The
output is byte-identical to the retaining implementation, which is pinned by a test that keeps
the old version as its oracle. Still a Python loop, and still per row; what changed is that it
is no longer unbounded.

### 6. Task granularity — the shuffle map side is finer than a node; the rest is not

**Closed for the Flight shuffles.** The map stage cuts its input into `workers x
map_partition_multiplier` partitions (default 4x, `dist/executors/ray_runtime/reducers.py::map_partitions`)
rather than one per worker, and `map_barrier` hands them out as actors go idle, keeping exactly
`workers` in flight. So a slow worker takes fewer partitions instead of holding the barrier open
on the one oversized partition it was statically dealt, and a dead worker's outstanding
partitions are re-dealt across every survivor rather than replayed whole onto one. It is a
ceiling, not a target: a source that cannot yield that many splits produces fewer partitions
instead of empty tasks, and an in-memory source stays at one per worker (cutting driver-resident
batches finer buys no recovery). Wired through aggregate, join, sort and window; the join pads
the shorter side's partition list because both sides map through one barrier under one source id.

Do not restate this as "skew can now be diluted by over-partitioning." It cannot, and that half
of the Spark argument was always wrong here: map partitions divide the *input*, while a shuffle's
imbalance lives in its hash buckets, where a single dominant key is indivisible however fine the
hash (measured: max/mean bucket load 3.8 at 8 buckets and 51.8 at 128). Splitting one key is
salting, below.

**Still coarse:** non-shuffle tasks are clamped to cluster cores, the disk transport still maps
one partition per worker (it is the single-node fallback — `resolve_transport` picks Flight on any
multi-node cluster), and the reducer remains one task per bucket.

On salting: `skew_join_salt = 0` is **not** "off" and has not been since hot keys became
measured. `dist/skew.py::resolve_hot_keys` takes them from the set learned for the join shape,
else from the column statistics Kyber holds, else from a Misra-Gries pre-pass it runs on its own
above ~8.4M input rows — and sizes the fan-out from the key's measured share. `0` means "let the
measurement decide"; a negative value is the actual off switch. Locality-aware reducer placement
is **on** (`locality_aware_scheduling = True`), which does not make the task unit finer — it only
decides which node a reducer runs on.

Speculation is **on** (`speculation_max_backups = 1`) — this section previously said it was off
at `max_backups = 0`, which the config contradicts. A backup duplicates one task; it makes the
unit no finer, which is what the paragraph above is for.

Since 2026-07-26 all of this is at least **observable**: recompute rounds, worker loss,
speculative backups, replica retirement, and proactive spot migration publish
`events.RECOVERY` (`_internal/events.py`), which `observe` folds into
`batcher_recovery_total{event=...}` and the live job view. Before that the entire recovery
machinery ran silently, so a query that survived losing two workers and one that was merely
slow were indistinguishable from outside.

### 7. A range join was a materialized cartesian product — **fixed below ~1M rows**

The IR had exactly two join nodes: `HashJoin` (equi) and `AsofJoin`. Everything else — every
inequality, every interval containment, every band join — lowered to a cartesian `HashJoin` on a
synthetic `__cross_key` with the predicate as a `Filter` above it. The intermediate was
`|L| x |R|` rows no matter how few survived: quadratic in time *and* memory, 13.1 GB RSS for
`n = 20,000`, and at `n = 100,000` it did not run.

**`bc_ir::RelOp::RangeJoin` is now a real operator** (`crates/bc-runtime/src/join/range.rs`): a
sorted-suffix scan for one inequality, **IEJoin** (Khayyat et al., the algorithm DuckDB's
`PhysicalIEJoin` implements) for two, with both axis sorts and the sweep parallelized.
`derive_range_join` (`kyber/rules/joins/range_join.py`) moves up to two crossing inequality
conjuncts off the filter and into the join, materializing a computed operand
(`a.ts - 5 < b.ts`) as a hidden column when it has to.

Batcher's own before-and-after is decisive — this removed the wall:

| n | Batcher before | Batcher now |
|---|---|---|
| 10,000 | 3,380 ms | **35 ms** |
| 20,000 | 15,573 ms | **42 ms** |
| 40,000 | 64,249 ms | **49 ms** |
| 100,000 | did not run | **78 ms** |
| 2,000,000 | did not run | **1,093 ms** |

**Against DuckDB it is still a loss, and the earlier version of this section said the opposite.**
DuckDB picks `IE_JOIN` only when its cardinality estimate for both inputs clears
`merge_join_threshold` (default 1,000, `src/execution/physical_plan/plan_comparison_join.cpp`);
a table registered from Arrow does not clear it, so DuckDB falls back to `NESTED_LOOP_JOIN`. The
first measurement here was against that fallback and reported 15-643x in Batcher's favour. With
the same data ingested by an untimed `CREATE TABLE`, DuckDB plans `IE_JOIN`:

| n | Batcher | DuckDB (native, `IE_JOIN`) | DuckDB (arrow, `NESTED_LOOP_JOIN`) |
|---|---|---|---|
| 10,000 | 35 ms | **15 ms** | 492 ms |
| 20,000 | 42 ms | **19 ms** | 1,886 ms |
| 40,000 | 49 ms | **38 ms** | 7,561 ms |
| 100,000 | 78 ms | **78 ms** | 47,503 ms |
| 200,000 | 146 ms | **97 ms** | *(not run)* |
| 500,000 | 251 ms | **138 ms** | *(not run)* |
| 1,000,000 | 476 ms | **274 ms** | *(not run)* |
| 2,000,000 | 1,093 ms | **383 ms** | *(not run)* |

**That was the state when this section was written. It has since been rewritten**, and the
operator is now 2.4x to 7.5x faster than the
table above, which puts it *ahead* of DuckDB below about half a million rows:

| n | Batcher (now) | DuckDB (`IE_JOIN`) | Batcher is |
|---|---|---|---|
| 10,000 | 3.7 ms | 9.5 ms | **2.6x faster** |
| 100,000 | 19 ms | 56 ms | **3.0x faster** |
| 500,000 | 87 ms | 132 ms | **1.5x faster** |
| 1,000,000 | 194 ms | 188 ms | **0.97x, parity** |
| 2,000,000 | 418 ms | 303 ms | 0.73x |
| 5,000,000 | 1,493 ms | 656 ms | 0.44x |

(Best of three runs each, taken once the box was quiet. Earlier runs shared it with another
session's benchmark and the ratios moved by up to 0.2x; the shape held across all of them. The sorts became the bottleneck once
the mark-scan hypothesis was measured and discarded; an order-preserving `u64` key per
primitive type and a dense-rank sweep took them from 758 ms to 158 ms at two million rows.)

**Batcher now wins below a million rows and reaches parity at a million.** Above that DuckDB
still wins, and the cause is specific. This implementation's cost is
`O(n log n)` for the sorts, `O(k)` for the emitted pairs, and a third term for *skipping* the
unset bits of the mark array — each left row scans the axis-1 suffix from its own bound to the
end. `MarkSet`'s summary level makes that `~ L x n / 4096` rather than `L x n / 64`, which is
why it stays invisible to about a million rows and dominates past that. DuckDB does not pay it:
`PhysicalIEJoin` decomposes the sorted union into blocks and prunes block *pairs* whose key
ranges cannot intersect, so its inner loop never walks a suffix holding no answers.

**So the honest state of this ceiling: the quadratic plan is gone, the memory wall with it, and
the operator now wins below ~500,000 rows and loses above ~1,000,000.** The named next step is the block
decomposition above — which is also what would make the operator *distributable*, since both
need the same "which block pairs can intersect" pruning. Today the distributed planner has no
range-join staging and executes the operator whole, which satisfies single-node == distributed
(`test_distributed_equals_single_node`) without scaling it out.

One property worth recording separately, because it is real and it is not an execution win: on
the zero-copy Arrow input both engines actually share, Batcher is **14x to 600x faster**, because
DuckDB will not choose its own range join without statistics it only has for its own storage. A
user handing DuckDB an Arrow table gets the quadratic plan. That belongs in the same place as
the rest of the `duckdb` vs `duckdb_arrow` findings in `benchmarks/TPCH_FINDINGS.md`, and it
must not be quoted as "Batcher beats DuckDB at range joins."

### 8. A first-seen query shape costs ~8 ms of optimizer, in Python — **stale; re-measured 2026-07-29**

**Read this before the section below, which is kept for its diagnosis and its four failed
attempts, not for its numbers.** Re-measured on the current tree (96 cores, `optimize()`
timed directly rather than inferred from end-to-end latency, a fresh literal per iteration so
the plan cache cannot hit):

| shape | cold `optimize()` | warm (plan-cache hit) |
|---|---|---|
| single-table filter + project | **1.52 ms** (median of 15) | **0.002 ms** |
| filter -> join -> group-by | **2.75 ms** (median of 10) | 0.002 ms |

That is **3-5x faster than the 5-8 ms this section reported**, so "the largest single latency
item on the board" is no longer true. End to end it does not read as a gap at all: over 12
never-seen shapes at n = 10,000, Batcher is **0.80x** DuckDB on a cold filter and **0.27x** on
a cold `count(*)` over a filter — DuckDB pays its own first-query cost, and ours is now
smaller. What remains is a **warm** ~2x on the narrowest shape (filter 10,000 rows and return
them: 4.07 ms against 2.04 ms), and planning is 0.002 ms of that, so it is per-query fixed
cost — FFI, Arrow hand-off, orchestration — and *not* the optimizer. The measurement was taken
under load average ~18, which if anything overstates the Batcher numbers.

The diagnosis below (282 rules over 7 phases, 19 traversals, no single dominant term) and the
four measured-and-reverted attempts remain worth reading. Do not re-quote its timings.

#### The warm gap, addressed 2026-08-07

The paragraph above ends by locating the remaining **warm** ~2x in per-query fixed cost rather
than the optimizer. That diagnosis held, and most of that cost is now gone on the opt-in path.
Over five small shapes at 1,000 rows, measured against the engine's own cost and A/B'd between
two sandboxes sharing one `lib_native.so`
(`benchmarks/BENCHMARK_RESULTS.md`, 2026-08-07): per-query control-plane overhead falls from
**5.77 ms to 0.27 ms (21.4x)** and end-to-end from 6.35 ms to 0.85 ms, comparing today's
default path against `execution.fast_path=True` after the change.

Three things were paying it, and the first was the largest: a re-issued query re-derived its
whole plan every call, which `api/orchestration/prepared.py` now memoizes. The other two are on
**every** path — `config_context` re-resolved the config on every terminal op (10.3 us -> 1.4
us), and `MetadataHub.record` JSON-encoded each feedback row to store it in an in-process dict.

Two limits on what this retires. It is **opt-in**: the default path improved only 1.09x,
because its remaining overhead is the learned-stats write side (~218 us a query), and skipping
that is the trade `fast_path` exists to document. And it is a **warm** result only — a
first-seen shape still pays the cold optimize measured above, which this did not touch.

#### The original measurement, superseded

Measured at `n = 10,000`, DuckDB in native storage, median over 12 never-seen query shapes:

| Query shape | Batcher | DuckDB | optimizer runs |
|---|---|---|---|
| `count(*)` over a filter | 25.8 ms | 10.0 ms | **2** |
| `sum()` over a filter | 17.8 ms | 9.2 ms | 1 |
| plain projection, no aggregate | 18.1 ms | 11.2 ms | 1 |
| filter only, one table | **9.2 ms** | **2.0 ms** | 1 |

A *repeated* shape hits `kyber.plan_cache` and costs 0.12 ms of planning; the same range-join
query is then 4.5 ms against DuckDB's 9.4 ms, i.e. **2x faster**. The whole gap is cold.

Two separate costs, and an earlier draft of this section conflated them:

1. **One optimizer pass is ~5-8 ms**, and that alone decides any query whose execution is
   under ~10 ms. The single-table filter row is the clean case: 9.2 ms against 2.0 ms, with no
   join and nothing to execute.
2. **`count(*)` over a filter runs the optimizer twice**, adding ~8 ms. The metadata-answer
   layer (`_answer_filtered_count_star`) optimizes the aggregate's *input* to see whether the
   surviving count is derivable from statistics, then execution optimizes the root. This is
   narrow — it is the only shape that does it — and the first draft here wrongly generalized
   it to every query.

**Where the 8 ms goes, measured per phase.** Of a 5.34 ms optimize on the simplest query, one
phase carrying **282 rules costs 4.19 ms — 78%**. The other six phases together cost 0.19 ms.
The driver is not naive about it: node-local rules already share one bottom-up traversal, and
leaf `Expr -> Expr` rules already share one expression walk within it. What remains is that
**every one of the 282 leaf rewrites is invoked on every expression node**, each opening with
its own `isinstance` check — roughly 2,800 Python calls for a two-predicate filter.

**Four attempts to remove it, and the cost is diffuse.** Each was measured on the
single-table filter, whose optimize is 5.3 ms:

| Attempt | Reasoning | Measured |
|---|---|---|
| Collapse the two memo entries | they looked like one rewrite keyed twice | no change — they optimize *different plans* |
| Cache the eligible leaf list per node type | rebuilt from every rule at every node | no change (~0.1 ms) |
| Memoize the whole fused expression pass per node | the node-rule `noop` argument applies to it too | 3% (5.32 -> 5.15 ms) |
| Index the node rules by node type per pass | ~14,000 `matches` tests a query before any rule fires | **slower** (5.44 ms) |

Instrumentation explains why: the fused expression pass runs only **4 times a query** with 427
leaf slots in total, so the 94 leaf rewrites are not the cost. The 188 *node* rules are, spread
across **19 traversals a query** with nothing dominant inside them — no single term to remove.
All four attempts were reverted.

That makes this an **architectural** cost rather than a hot spot: 282 rules over 7 phases,
iterated to a fixpoint, in Python, for a plan of four nodes. The candidate fixes are therefore
structural too — fewer passes (converge detection per rule family rather than per phase), or
moving the fixpoint loop out of Python — and neither is a tuning exercise. It remains the
largest single latency item on the board.

### 9. A small `GROUP BY` pays most of its wall time learning, not aggregating (measured 2026-09-01)

The one relational shape where Batcher loses to DuckDB in-process, and the reason is not the
aggregation.

Measured on 96 cores, 4M rows, one `int64` key and one `float64` measure, three repetitions
taking the minimum, **a fresh `Dataset` per repetition** and `collect()` rather than
`count()` (see the two traps below):

| groups | Batcher | DuckDB | ratio |
|---|---|---|---|
| 2 | 59.5 ms | 50.3 ms | 1.18x |
| 10 | 60.8 ms | 48.1 ms | 1.26x |
| 100 | 70.8 ms | 47.9 ms | **1.48x** |
| 1,000 | 76.0 ms | 50.4 ms | **1.51x** |
| 10,000 | 84.0 ms | 67.1 ms | 1.25x |
| 100,000 | 91.3 ms | 148.3 ms | 0.62x |
| 1,000,000 | 92.0 ms | 275.4 ms | **0.33x** |

Read the shape of the two curves before the ratios. Batcher's cost is **nearly flat** --
59.5 ms to 92.0 ms across five hundred thousand times more groups -- where DuckDB's climbs
50 ms to 275 ms. That is the mergeable algebra doing what it is for, and it is why the
crossover exists at all. Holding groups at 100 and varying rows instead, the crossover is
between 4M and 16M rows: 74.7 ms against 47.6 ms at 4M, and **93.8 ms against 147.6 ms at
16M**.

**Where the time goes.** Profiling a 100,000-row grouped aggregation (`cProfile`, five
iterations, cumulative):

| call | share |
|---|---|
| `_close_learning_loops` → `learn_column_stats` → `_native.heavy_hitters` | **49%** |
| `_optimize` → `column_statistics` → `_native.column_stats_full` | 14% |
| `seed_column_ndv` → `_native.column_ndv` | 7% |

So roughly **70% of a small grouped query is the cross-query learning loop and the
optimizer's own statistics**, and the aggregation it is timing is the remainder. That is
not a defect in the aggregate kernel and no amount of work on it would move this number.

**It is, however, the rule in `.claude/rules/performance.md` being broken by its own moat:**
"Don't add per-query setup cost ... that hurts the small case to help the large one. Make it
adaptive." The learning loop is already bounded by *columns* -- `learn_column_stats` sketches
only the columns the estimator can consult, which is what took a 20M-row scan's overhead from
22.9 s down -- and it is not bounded by *rows*. The adaptive re-optimization loop next to it
is (`api/adaptive/gating.py`), for exactly this reason.

**Recorded rather than changed.** `api/terminal/_metadata.py` and `core/stats.py` were both
another session's uncommitted work at the time of measurement, and that work
(`_sketch_shards`, re-slicing the input so the sketch parallelizes) is aimed at this same
cost. A second edit there would collide with it.

**Two measurement traps, both of which inflate Batcher, both hit before the numbers above
were believed.** They are recorded because anyone re-running this will hit them:

1. **`count()` is answered from metadata.** A sort does not change the row count, so
   `ds.sort(...).count()` never executes the sort -- the first run of this probe reported a
   4M-row sort in **0.2 ms**. Materialize with `collect()`.
2. **The result cache serves repetitions 2..n.** A min-of-three over one `Dataset` measures a
   cache hit; it reported a 4M-row full sort at 43 ms against DuckDB's 608 ms. Build a fresh
   `Dataset` each repetition so the source identity differs.

**Every other shape measured here Batcher wins**, verified equal-result on the same Arrow
input: filter 0.24x, high-cardinality group-by 0.41x, distinct 0.20x, sort+limit 0.78x, full
sort 0.14x, string group-by 0.42x. So did every string and expression case -- `upper` 0.06x,
`contains` 0.12x, `replace` 0.03x, `split`+`len` 0.11x, `regexp_extract` 0.43x, three-term
arithmetic 0.18x, `CASE WHEN` 0.15x -- with the outputs checked element-for-element against
DuckDB rather than assumed.

## Claims to retire

These are asserted in the repo and contradicted by its own code.

1. **`CLAUDE.md:7-9`: adaptive re-optimization "which Spark AQE's stage-boundary adaptation cannot
   match."** The loop is real (`api/adaptive.py:342-410`, genuinely re-optimizing on *measured*
   cardinalities) — but it **is** stage-boundary adaptation. `api/adaptive.py:1` says so verbatim
   in its own first line: *"Adaptive (intra-query) execution: stage-boundary re-optimization."*
   Same mechanism, same granularity as AQE. It is also **off below a size floor**
   (`_ADAPTIVE_MIN_ROWS_PER_STAGE`, 5M input rows for each pipeline breaker the loop would
   cut at — about 10M for the simplest joined shape), so most queries never touch it.
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
4. ~~**`core/executor.py:1-11` references "the `bc-adapt` re-optimization loop."**~~ **Fixed
   2026-07-29.** There is no `bc-adapt` crate and the Rust side has no adaptivity at all; both
   `core/executor.py` and `core/__init__.py` now say so and point at `api/adaptive/`, which is
   where the stage-boundary loop actually lives (it must, since re-planning re-runs Kyber and
   Core may not import it).

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


### 10. `sort().limit(n)` costs a whole scan; DuckDB's costs about 40% of one (measured 2026-09-01)

Batcher wins the two shapes either side of this one and loses this one, which is what makes it
worth writing down rather than folding into a general "sorting is fine".

Release build, 2,000,000 rows, six columns, local Parquet, min of five runs, DuckDB at
`PRAGMA threads=8`:

| Shape | Batcher | DuckDB | Ratio |
|---|---|---|---|
| bare scan, all columns | 64.2 ms | 165.5 ms | **0.39x** |
| full sort, all columns | 95.9 ms | 419.1 ms | **0.23x** |
| `ORDER BY x LIMIT 100`, all columns | 70.1 ms | 42.2 ms | **1.66x** |
| `ORDER BY x LIMIT 100`, one column | 16.5 ms | 13.1 ms | 1.26x |

The plan is not the problem. `explain()` shows the limit fused into the sort and pushed into
the scan -- `sort [top 100 by x]` over `scan ... pushed[top 100 by x]` -- so the optimizer is
doing what it should.

What the numbers say is narrower and is about *cost relative to a scan*. Batcher's top-N costs
essentially what materializing the whole relation costs; DuckDB's costs a fraction of it. Over
1,000,000 rows at three widths:

| Columns | Batcher top-N | vs its own scan | DuckDB top-N | vs its own scan |
|---|---|---|---|---|
| 2 | 45.9 ms | 2.49x | 15.3 ms | 0.78x |
| 10 | 87.1 ms | 1.12x | 41.8 ms | 0.44x |
| 40 | 361.3 ms | 1.05x | 167.2 ms | 0.39x |

So the gap is not a fixed overhead and it does not close as the query gets bigger: at ten
columns and above, Batcher pays a full materialization for a hundred rows and DuckDB pays
about 40% of one.

**Batcher's side of it is not an inference.** The engine's own counters, over the 40-column
table, say what it reads. A full scan and a `LIMIT 100` report the *same* figures:

| | rows scanned | bytes scanned | rows returned |
|---|---|---|---|
| full scan | 1,000,000 | 320,000,000 | 1,000,000 |
| `ORDER BY x LIMIT 100` | 1,000,000 | 320,000,000 | 100 |

Byte for byte identical, to return a hundred rows out of a million. The top-N materializes
every column of every row and then discards 99.99% of it, which is why its cost tracks the
full scan and why widening the projection widens the gap. The shape of a fix follows from
that and needs no further measurement: read the sort key, resolve the surviving row ids, and
fetch the other columns only for those hundred rows.

**The other half of the explanation was hypothesized, tested and rejected**, and that is
recorded because the hypothesis is the obvious one and someone will have it again. The guess was that DuckDB
late-materializes -- reads the sort key, picks the surviving row ids, then fetches only those
rows' remaining columns -- which predicts its top-N is roughly *flat* in the projected width.
It is not: 15.3 -> 41.8 -> 167.2 ms as the width goes 2 -> 10 -> 40, growing at much the same
rate as Batcher's. Whatever DuckDB is doing costs a stable ~40% of its own scan rather than
avoiding the columns altogether. So the asymmetry is real and measured on both sides, and
only *Batcher's* half of it has a mechanism attached: DuckDB's remains unidentified, and
nothing here should be read as saying otherwise.

Worth keeping in proportion. The same sweep has Batcher at 0.23x on a full sort, 0.31x on
`DISTINCT`, 0.34x on `COUNT(DISTINCT)` and 0.36x on a self-join. The only other losses were
`GROUP BY` shapes, and following them up folded them into section 9 rather than adding a
finding: **Batcher's `GROUP BY` is nearly flat in cardinality and DuckDB's is not.**

Over 2,000,000 rows, the same key drawn as an integer and as a string:

| Key | Groups | Batcher | DuckDB | Ratio |
|---|---|---|---|---|
| int | 10 | 26.3 ms | 12.0 ms | 2.20x |
| string | 10 | 30.4 ms | 11.7 ms | 2.59x |
| int | 1,000 | 25.3 ms | 10.9 ms | 2.33x |
| string | 1,000 | 34.1 ms | 11.9 ms | 2.86x |
| int | 200,000 | 46.6 ms | 94.5 ms | **0.49x** |
| string | 200,000 | 61.6 ms | 143.9 ms | **0.43x** |

Batcher goes 26 ms to 47 ms across four orders of magnitude of group count; DuckDB goes 12 ms
to 95 ms. That is section 9 again at a different scale, and the two together say what the
floor is made of. It is **flat in the group count and not flat in the row count**: ~25 ms
here at 2M rows against ~60 ms in section 9 at 4M, which is what a floor made of
`heavy_hitters` and `column_stats_full` should do, since both walk the column. Calling it a
fixed per-query cost would be wrong and would point at the wrong fix -- it is per-row work
that the group count does not change, sitting under an aggregation that scales better than
DuckDB's.

The string key is worth naming only to rule it out. It adds 4-15 ms over the integer key at
matched cardinality and does not change the shape of either curve, so the 2.54x on a
string-keyed `GROUP BY` in the sweep was the fixed overhead with a modest string cost on top,
not a string-hashing problem. Reporting it as one would have sent someone to the wrong file.

A second sweep, run to see whether top-N was one of many losses or one of few, says few. Same
release build, 2,000,000 rows, seven columns including a string and a timestamp, row counts
checked against DuckDB's before any time was reported:

| Shape | Batcher vs DuckDB |
|---|---|
| `ORDER BY g, x` | **0.23x** |
| `replace()` | **0.25x** |
| `upper()` + `length()` | **0.27x** |
| `split_part()` | 0.33x |
| `CASE WHEN` | 0.42x |
| `regexp_extract()` | 0.46x |
| `sum() OVER (PARTITION BY)` | 0.48x |
| semi-join (`EXISTS`) | 0.49x |
| `GROUP BY` two keys | 0.52x |
| `median()` | 0.72x |
| `UNION ALL` | 0.99x |
| `sum/avg/min/max` | 1.03x |
| `stddev()` | 1.40x |
| `GROUP BY ... HAVING` | 1.51x |

Batcher wins twelve of the fourteen, several by 3-4x, and the two it loses are the two that
return almost nothing -- `stddev` returns one row and the `HAVING` returns 2,412. Both are
19 ms against 13 ms, which is the floor above and not a property of either operator.

That is the useful conclusion from twenty-six measured shapes across the two sweeps: apart
from top-N, **every shape Batcher loses is a small result, and every one of them is the same
floor.** There is no scattered set of slow operators to go and optimize. There is one
cost, it is the learning loop and the optimizer's own statistics, and section 9 already says
where it lives.


## The roadmap that would make the claim true

In dependency order. (1) and (2) are the ones that change what Batcher *is*.

1. **Implement the streaming executor RFC.** Removes the sf100 OOM, the gather tax, and the
   window/materialization losses. Without it, "any data scale" is false on one node.
2. **Adopt `StringView` + preserve dictionaries through the kernels.** The other half of the
   single-node gap, and it compounds with (1). *Comparisons are done* (`try_dict_compare`,
   19.6x on the low-cardinality string filter); what remains is `StringView`, keeping the
   dictionary alive through project/join rather than decoding at the leaf, and a dict-aware
   string join key.
3. **~~Wire the shuffle replication that already exists~~ (done — all four shuffles) and
   ~~call `clear_plan` on the batch path~~ (done).** Spot preemption is now a re-fetch rather
   than a recompute for aggregate, join, sort and window, and long-lived fleets no longer leak
   buckets. ~~Interior combiner-tree outputs~~ are replicated too (done —
   `shuffle_replication.replicate_interior_outputs`, called per level from
   `flight_aggregate._tree_reduce`), so losing one combiner no longer discards every level
   built beneath it. Remaining on this ceiling: a **disk tier / external shuffle service**
   (replicas are RAM today, so they cost memory and a whole-node loss taking every copy still
   recomputes).
4. **~~Default `stream_inference=True`~~ (done), generalize to N stages with per-stage
   autoscaling.** The AI moat is now on out of the box. Remaining: the N-stage topology,
   per-stage autoscaling and keeping data resident on-device across ops. (The variable-shape
   tensor type this item used to list is done.)
5. **~~Per-partition watermarks~~ (done) + ~~object-store checkpoints~~ (done) +
   ~~incremental state checkpoints~~ (done for the running aggregate); a real state backend
   remains.** The frontier is now a minimum over per-partition maxima with idleness, a
   checkpoint can live on `s3://`/`gs://`/`hdfs://`, the running aggregate writes a changelog
   instead of rewriting its whole state every epoch, and the *windowed* aggregate spills its
   cold windows instead of raising. What is left is the state the ordered-eviction trick does
   not reach: a keyed store with point lookups, so the **unwindowed** aggregate and keyed state
   degrade instead of raising; a changelog for the evicting operators, which needs a tombstone
   the current one has no way to express; and a driver that is not a single point of failure. Until those land, do not claim Flink parity —
   and refuse, loudly, the shapes that cannot be honoured (the distributed watermark gate and the
   unwindowed-watermark refusal both do).
6. **Decouple tasks from nodes**; turn on speculation and skew splitting by default.
7. **~~Plan on the layout a table already has~~ (done for aggregate, distinct and window —
   4.2x).** A partitioned table — a Hive Parquet tree, a Delta table — has every group of its
   partition columns wholly inside one worker, so a `GROUP BY day` over a directory-per-day
   table needs no exchange at all. It used to hash-shuffle every row to rediscover a
   partitioning the layout already stated. `kyber/properties.py::clustered_on` propagates the
   clustering; `dist/executor.py::_partition_aligned_aggregate` / `_partition_aligned_distinct`
   / `_partition_aligned_window` schedule against it. 8M rows over 16 partitions on 8 workers:
   850 → 200 ms on Parquet (**4.2x**), 780 → 310 ms on Delta (**2.3x**), and 1,020 → 490 ms
   for `COUNT(DISTINCT)` (**2.1x**, growing with the column's cardinality),
   `benchmarks/internals/partition_aligned.py`.

   The failure mode this item warned about — reporting partial groups as final, a wrong answer
   at cluster scale and green everywhere else — is closed by *verifying* rather than declaring.
   `io/splits/clustering.py` checks that every split of the set the read will actually use
   declares the same columns, and **establishes** the half no single split can promise — that a
   value is on one worker — by grouping splits by value and assigning whole groups. That
   grouping is what makes it work for a file-per-split lakehouse reader, where one partition is
   hundreds of splits and the split set alone proves nothing. Values are compared typed, so
   `x=01` and `x=1` are not read as two partitions of one value, and the executor re-checks the
   splits it is about to assign, **raising** rather than silently falling back to a plan that no
   longer has a combine in it.

   Hive Parquet trees, Delta and Iceberg all declare a clustering. Iceberg needs one extra
   guard the others do not, because it is the only format whose partitioning can evolve under an
   existing table: a file written before a spec change carries the *old* spec's partition
   fields, so any file whose `spec_id` is not the current one declares nothing and the whole set
   is refused.

   A nested `year=/month=` tree is clustered on the year, which is the *complete* guarantee
   and not a partial one: `GROUP BY year, month` is aligned through the containment, and
   `GROUP BY month` alone is not co-located by any split granularity, since `month=1` lives
   under every year.

   **Remaining on this ceiling: joins.** A join whose two sides are partitioned by the join key
   is co-partitioned on disk and could skip its shuffle on the same argument, but it needs both
   sides to agree on the layout — a materially stronger condition than either single-input
   operator has to meet, and it needs a co-assignment that pairs the two sides' groups rather
   than bin-packing each independently.

Until 1–2 land, the defensible positioning is narrower than the current one, and *still strong*:

> The fastest **distributed** Arrow engine, with an optimizer that learns across runs — beating
> Ray Data by 50–450×, Spark by 13–197×, and Polars at most shapes; on a single node it wins to
> at least 60M rows and cedes to DuckDB at sf100.

That is a claim the benchmarks in this repo actually support.

**The single-node half of that sentence said "wins below ~10M rows and cedes to DuckDB above
it" until 2026-09-01.** It was stale against this file's own body, which retired the sf10 half
of that row on 2026-08-25: TPC-H at sf10 is a 60M-row `lineitem` and it is a **win** at 0.963x.
The loss is at sf100 — 600M rows, with q3/q4/q5 OOM — so the crossover is somewhere between
60M and 600M rows and is not near 10M.

A fresh four-shape check at 16M rows on the 96-core node agrees, and is recorded because it
was run to test the retired sentence rather than to support it: `ORDER BY` 0.14x, filter
0.27x, join 0.35x, and a 100-group `GROUP BY` at 2.40x. Three decisive wins and one loss, and
the loss is the low-cardinality `GROUP BY` floor of sections 9 and 10 rather than anything
about scale. Batcher's position *improves* with size across those shapes -- filter goes 1.22x
at 1M to 0.49x at 4M to 0.27x at 16M -- which is the opposite of what the old sentence
predicts.
