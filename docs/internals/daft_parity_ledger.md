# Daft parity ledger (and the single-node competitive picture)

**Status:** open, started 2026-07-24. Read `competitive_architecture.md` for the scorecard and
`competitor_technique_review.md` for the mechanism-level parts list; this file is narrower than
either.

It started with one question — what can a Daft user do that a Batcher user cannot, and what did
we do about it — and the capability half below is still exactly that. The *measurement* half
outgrew it: chasing the one place Batcher measurably lost to Daft led into DuckDB and Polars, so
"Measured" now covers all three single-node engines under one set of conditions. The filename is
kept because several files point at it.

Daft was read from `/mnt/shared_storage/ref/Daft` (the Python surface under `daft/`, the
Rust crates under `src/`).

## Method, and what it is worth

The gap list is mechanical, not impressionistic. Daft's callable surface is extracted from
`daft/functions/**`, `daft/expressions/expressions.py`, and `daft/dataframe/dataframe.py`;
Batcher's is introspected from the *live* objects (`dir()` over `bt`, `Expr`, every accessor
namespace, `Dataset`, `bt.read`), not from source text, so a generated accessor counts.
Set-difference gives 398 Daft names against a 1,377-name Batcher vocabulary, with 220 Daft
names absent.

Two things that number is not:

1. **It is not 220 missing capabilities.** Most of the 220 are names for something Batcher
   already does under a different spelling — `where`/`filter`, `avg`/`mean`,
   `write_parquet`/`to_parquet`, `list_sum`/`.list.sum()`, `length_bytes`/`.str.len_bytes()`.
   Batcher's contract forbids adding a second spelling of an existing capability
   (`python-control-plane.md`, "one obvious way"), so those close in the
   `migrate-from-daft` skill, not in the API.
2. **A name-level diff cannot see behaviour.** Two engines can share a function name and
   disagree about nulls, units, or overflow. Where that mattered it is called out below.

The triage below therefore separates *real capability gaps* from *spelling gaps*, and only
the first kind produces work.

**A measurement of the method itself.** Re-running the same diff after the work in the
closed table below: Batcher's vocabulary went 1,377 → 1,404 names, and Daft names absent
went 220 → 212. Roughly fifty capabilities landed and the headline number moved by eight.
Both halves of that are the point. Most of what landed is *parameterized* — one
`to_case(style)` covers seven Daft functions, one `from_epoch(unit)` covers three, one
`compress(codec)` covers four — so closing a capability gap usually adds one name, not
seven. And most of the residual 212 were never capability gaps at all. Do not track this
number as progress; track the closed table.

## Closed

| # | Gap | What landed | Verified by |
|---|---|---|---|
| 1-10 | Daft's seven case converters (`to_snake_case`, `to_camel_case`, …) | `.str.to_case(style)` with ten styles — Daft's seven plus `sentence`, `dot`, `train` | `tests/differential/test_diff_str_to_case.py` (38 cases, 5 anchored to DuckDB); `crates/bc-expr/src/eval/str/case.rs` unit tests |
| 11-12 | `json_array_length`, `json_object_keys` | `.json.array_length(path)`, `.json.keys(path)` | `tests/differential/test_diff_json_shape.py` vs DuckDB `json_array_length` / `json_keys` |
| 13-15 | *(beyond Daft)* no way to inspect a document's shape | `.json.values(path)` (JSON array to list column, so `explode` applies), `.json.type_of(path)`, `.json.exists(path)` | same file, vs DuckDB `json_type` / `json_exists` |
| 16-17 | `make_date`, `make_timestamp` | Same, as `Expr::MakeTemporal` | `tests/differential/test_diff_temporal_constructors.py` vs DuckDB `make_date` / `make_timestamp` |
| 18-21 | `timestamp_seconds`/`millis`/`micros` | `from_epoch(expr, unit)` covering `s`/`ms`/`us`/`ns` — one unit more than Daft | same file, vs DuckDB `to_timestamp` / `epoch_ms` |
| 22 | `date_from_unix_date` | `from_unix_date(expr)` | same file |
| 23 | *(structure)* `eval/` hit its 12-file cap | `eval/temporal/{date,timezone,make}.rs` subpackage | `just lint-structure`, `just surface-diff` |
| 24-35 | `compress`, `decompress`, `try_compress`, `try_decompress` over deflate/gzip/zlib | `.str.compress(codec)` / `.str.decompress(codec)` over six codecs — Daft's three plus zstd, brotli, lz4. Decompression is lenient, so no `try_` variants are needed | `tests/differential/test_diff_str_compress.py`, both directions against Python's own codec implementations |
| 36 | *(interop)* zstd frames were unreadable by one-shot decoders | `bulk::compress` records the content size in the frame header | same file, against `zstandard` |
| 37 | `regexp_split` | `.str.regexp_split(pattern)` | `tests/differential/test_diff_regexp_split_and_geo.py` vs DuckDB `regexp_split_to_array` |
| 38-41 | `great_circle_distance` (km) | `bt.great_circle_distance(..., unit)` in km/m/mi/nm | same file, vs an independent haversine and six geodesics fixed by geometry |
| 42-43 | `image_channel`, `image_mode` | `.image.decode()` now returns `{width, height, channels, mode}` from **one** header read, where Daft spends a call per fact | `crates/bc-expr/.../image.rs`, `tests/integration/test_image_expr.py` |
| 44 | `crop` | `.image.crop(x, y, w, h)` → PNG bytes, clipped at the edge rather than padded | same files |
| 45-48 | `encode_image` | `.image.encode(format)` over png/jpeg/bmp/gif | same files |
| 49 | *(benchmark)* Daft was measured only in the multi-node lineup | Daft added to the single-node default lineup | `benchmarks/engines/lineup.py`; results in this file |
| 50 | *(process)* the debug-build guard was a timing heuristic that passed silently | `bc_py` exports `__engine_profile__`; the suite hard-stops on a debug engine, and `bt.versions()` reports it | `benchmarks/run.py::_check_build_profile`; `tests/unit/test_toplevel_namespace_ergonomics.py` |
| 58 | *(perf)* a light `map_batches` callback was batched 2.7x below its measured optimum | `thread_batch_target` targets 1,048,576 rows for a light `fn` (was a 131,072 floor under a 262,144 cap), bounded by bytes so a wide-row `fn` shrinks instead | 6 M rows 22.3 → 9.7 ms, 12 M rows 39.5 → 15.2 ms, best-of-5; `tests/unit/test_learned_udf_strategy.py` |
| 56 | WARC (`daft-warc`) | `bt.read.warc(path)` — one row per crawl record, named headers typed, the rest as JSON in `warc_headers`, `.warc.gz` including per-record gzip members. No third-party dependency | `tests/io/test_warc.py` (18 tests) |
| 57 | *(guidance)* the IO category list omitted `robotics` | `maintainability.md` and the connector skill now list ten categories | `just lint-guardrails`; `io/formats/robotics/` |
| 52-55 | `convert_image` | `.image.convert(mode)` over `L`/`LA`/`RGB`/`RGBA`. **Batcher's image namespace now covers Daft's completely.** Writing it surfaced a divergence: `image`'s `into_luma8` weights Rec. 709 while `to_grayscale` and `dhash` weight Rec. 601, so the three would have disagreed about grey (147 vs 124 on RGB(10,200,30)). One `rec601` now serves all three, asserted | `crates/bc-expr/.../image/`, `tests/integration/test_image_expr.py` |
| 51 | *(perf)* media reads issued a stat per file that the fetch already answered | `read()` fetches first and chunks from the returned sizes; small-corpus image ingest went from 1.8-2.2x behind Daft to 1.14-1.25x | `benchmarks/run.py --benchmark images --scale 10`; numbers below |
| — | `audio_metadata` | **Already present**, found while triaging: `.audio.decode()` returns `{sample_rate, channels, num_frames, duration_secs}`. Listed here because the ledger claimed it as a gap and was wrong. |

### Note on the epoch gap

`from_epoch` is the one item so far that closed a *correctness* trap rather than a missing
convenience. An `Int64` column of epoch counts carries no record of its unit, so
`cast("timestamp")` has to assume one, and Arrow assumes microseconds. A column of epoch
seconds cast that way silently became January 1970 — a plausible-looking timestamp, no
error. The behaviour is pinned in
`test_a_bare_cast_would_have_read_epoch_seconds_as_microseconds` so the function cannot be
refactored back into a cast.

### A note on where these commits live

Several of the code changes recorded here were committed by other agents working in the
same tree, which staged them alongside their own work before this session could. The code
is in `HEAD` and correct; what would otherwise be lost is the reasoning, which is why this
ledger carries it at more length than a ledger normally would. Specifically the media
read-path work (items 51 and the header-parse change before it) has no commit message of
its own.

## Open — real capability gaps

Ranked by value. "Daft" names the Daft function the gap was found from.

Every row below has been checked against the code, not just against the name diff. That
distinction is not pedantry: four rows that started here (`audio_metadata`, MCAP,
HuggingFace, the tokenizer) turned out to be present already, and one more (Iceberg
partition transforms) turned out to matter far less than its name suggested. A row that
survives here is a gap someone read the code to confirm.

| Gap | Daft | Notes |
|---|---|---|
| Group-wise Python apply | `map_groups` | **Single-node closed; distributed unmeasured.** `GroupBy.map_groups(fn)` in `api/group_apply.py`. The shuffle this row asked for turned out to be one Batcher already has: `array_agg` is a **mergeable** aggregate, so it emits exactly one row per key however many partitions the input has, and that row carries the whole group as list columns. `map_groups` lowers to that aggregation followed by an ordinary `map_batches` over one-row-per-group batches, and rebuilds each group by slicing the list's child array — no per-row Python and no new operator. So the *grouping* is correct at any partition count by the aggregation's own guarantee. `batch_format="pandas"` converts per group, which is the `applyInPandas` shape. **What is left open**: the plan is a `map_batches` above a relational breaker, which `is_map_prefix` excludes from the embarrassingly-parallel route, so whether `collect(distributed=True)` runs it is the same question as for `agg(...).map_batches(fn)` — pre-existing, and not measured when this landed (the shared cluster was saturated). Also still open: one group is materialized at a time, which the streaming carry-over operator described below would remove. |
| Iceberg partition transforms | `partition_iceberg_bucket`, `partition_iceberg_truncate`, `partition_{years,months,days,hours}` | **Lower value than it first looks.** Batcher's Iceberg sink hands each shard to `pyiceberg`, which applies the table's own partition-spec transforms, so a partitioned write is already correct without these. What is left is manual bucketing and bucketed joins. The temporal four compose from existing expressions; only `bucket` needs an exact `murmur3_x86_32`, which is not in the tree. |
| Row-identity generators | `monotonically_increasing_id`, `uuid`, `random_int` | **Two are spelling, one is a decision.** `monotonically_increasing_id` is `ds.with_row_index()` and `random_int` is `ds.with_random()` scaled and cast — both exist at `Dataset` level rather than as expressions, so they belong in the migration table. `uuid()` is genuinely absent and should stay absent in its usual form: a random UUID per row is non-deterministic, which breaks the interpreter-as-oracle property the whole test strategy rests on (sequential, parallel and distributed runs would disagree). `with_random` is seed-keyed for exactly that reason. A deterministic v5-style UUID over a key column would be admissible; nobody has asked, and `hash64`/`xxhash64` already cover surrogate keys. |
| Video frame access | `video_frames`, `video_keyframes`, `get_video_frame_by_idx` | `.video.decode()` already covers `video_metadata` (it returns `{width, height, num_frames, duration_secs, fps}`), so the real gap is frame *extraction*, not metadata. Needs the `video` cargo feature (system FFmpeg). |
| File metadata | `file_size`, `file_exists`, `file_path`, `guess_mime_type` | Path-level facts as expressions. **Costed and deferred:** each is a `stat` per row, which means object-store access from `bc-expr` — a crate that sits under everything and deliberately links no IO. The right home is `bc-io` or the UDF plane, and that is a design decision rather than a function to add. |
| HDF5 accessors | `hdf5_attrs`, `hdf5_keys`, `hdf5_metadata` | Batcher reads HDF5 as a *format* (`io/formats/ml/hdf5.py`, via h5py). The per-row accessors would need an HDF5 parser in the data plane; routing them through h5py instead would be per-row Python, which the architecture rule forbids on a hot path. Costed, not scheduled. |
| Tokenizer in the data plane | `tokenize_encode`, `tokenize_decode` | **Capability present, placement differs.** `batcher.ml.Tokenizer(column, tokenizer, max_length=, truncation=, padding=, attention_mask_column=)` takes any callable, including a HuggingFace fast tokenizer, and `decode` is the same shape in reverse — so real BPE ids are producible and consumable today (an earlier version of this row said otherwise and was wrong). The genuine difference is *where it runs*: Daft's is a Rust expression, Batcher's is a per-batch Python callback. Closing that means a tokenizer dependency in `bc-expr`, which is a weight decision on a crate everything links, not a missing function. |
| `jq` | `jq` | A full jq engine. Recorded so the gap is honest; the shape accessors above cover the common cases. |
| Connectors | `write_turbopuffer`, `write_bigtable`, `write_paimon` | Batcher's connector surface is wider overall (ten categories, 77 registered formats); these three remain Daft-only, all of them *sinks* into hosted services. **Two entries were listed here in error**: MCAP exists in `io/formats/robotics/` (a category the agent guidance itself had omitted), and HuggingFace is largely covered by `bt.from_huggingface(ds)` — what Daft additionally has is a Hub-name reader that streams the Hub's Parquet exports, where Batcher asks you to load the dataset first. WARC is now closed. |

### `map_groups`: what a sort buys, and what `repartition` does not

Measured over 50k rows and 20 keys, counting how many `map_batches` calls each key's rows
were spread across:

| Plan | calls | keys spanning >1 batch | worst key |
|---|---|---|---|
| `map_batches(fn)` | 4 | 20 / 20 | 4 batches |
| `repartition(by="k").map_batches(fn)` | 4 | 20 / 20 | 4 batches |
| `sort("k").map_batches(fn)` | 4 | **3 / 20** | **2 batches** |

Two things worth having before anyone starts:

1. **`repartition(by=)` is not a shuffle.** Its docstring is explicit — "set how the next
   `write` lays out its files (the data is unchanged)" — and the measurement confirms it:
   byte-identical spread to no repartition at all. An earlier version of this row cited it
   as evidence that composition cannot work, which was the right conclusion from the wrong
   experiment.
2. **A sort nearly gets there.** Sorting by the key makes each group *contiguous*, so a
   group can only be split where it straddles a batch boundary — at most two batches, and
   only for 3 of 20 keys here. That makes `map_groups` a **streaming operator with a
   one-group carry-over buffer**, not the shuffle-and-materialize this row first assumed.
   Cost is the sort; correctness distributed still needs each key's rows on one worker,
   which the existing hash shuffle provides.

That is a materially smaller piece of work than "a distributed-execution change", and it is
still not a `GroupBy` method — it needs an operator that can hold one group across a batch
boundary.

**What shipped instead, and what is still open.** `map_groups` took the third route: reduce
each group to one row with `array_agg`, which is mergeable and therefore already produces
one row per key at any partition count, then run the callback over those rows. It needs no
new operator and no sort, and the grouping is right for the same reason every other
aggregate's is.

Two things it does not get. It does not get the sort's memory profile: a group is
materialized whole, so one enormous key has to fit in a worker, and the streaming
carry-over operator above is the fix for that. And it does not, on its own, answer whether
the distributed executor's *multi-worker* path will run the plan — a `map_batches` above a
breaker is outside `is_map_prefix`, so that is the same open question as
`agg(...).map_batches(fn)`, and closing it closes both.

**A bug this shook out.** Probing the shape under `distributed=True` failed — and so did
plain `ds.map_batches(fn).collect(distributed=True)`, which plainly should not. The cause was
not the plan shape: `ray_runtime.lifecycle._single_node`, the fallback every distributed run
lands on when Ray is unavailable or the cluster is one node, ran `kyber.optimize` and so
lowered the plan to IR — which `MapBatches.to_ir()` refuses by design. The fallback therefore
could not execute **any** UDF pipeline, failing with an internal message about the wire
contract for exactly the batch-inference workload most likely to be run distributed. It now
routes a UDF plan through `core.execute_with_udfs` after a logical-only optimize, the same
way `executors.write` already routed one away from the IR.

## Open — spelling gaps only (no API work)

These close in `.claude/skills/migrate-from-daft/SKILL.md`, which is the translation
table a porting user reads. Adding the Daft spelling to the API would violate "one obvious
way".

`where`/`filter`, `avg`/`mean`, `stddev`/`std`, `count_rows`/`count`, `union_all`/`union`,
`drop_null`/`drop_nulls`, `write_*`/`to_*` and `ds.write.*`, `list_*(col)`/`col.list.*()`,
`length_bytes`/`.str.len_bytes()`, `eq_null_safe`/`eq_missing`, `is_inf`/`is_infinite`,
`not_null`/`is_not_null`, `power`/`pow`, `columns_sum`/`sum_horizontal`, `date_format`/
`strftime`, `datepart`/`date_part`, `date_trunc`/`.dt.truncate()`, `day_of_month`/`.dt.day()`,
`to_struct`/`struct`, `dot_product`/`.list.dot()`, `jaccard_similarity`/`.list.jaccard()`,
`pearson_correlation`/`corr`, `over`/`.over()`.

## Where Batcher is already ahead

Recorded so the ledger is not one-sided, and so nobody "closes" a gap that runs the other
way. Checked against Daft's tree, not its documentation.

- **Optimizer.** Kyber is 317 registered rules with sketch-backed cardinality; Daft's
  `daft-logical-plan` optimizer is a small rule set with no learned statistics.
- **Adaptive re-optimization.** Batcher re-optimizes at pipeline breakers on measured
  cardinalities. Daft has none.
- **Expression breadth.** 1,377 callable names against 398. The `.str` namespace alone is
  174 methods against Daft's 59, and the statistics / model-metric / text-metric function
  families have no Daft counterpart at all.
- **Connectors.** 76 registered formats across nine categories against Daft's ~20.
- **Governance.** Row filters and column masks as a plan rewrite, with lineage. Absent from
  Daft.
- **Data quality.** `ds.dq` with fail/drop/quarantine. Absent from Daft.
- **SQL.** Batcher has a SQL front-end over the same plan; Daft's `daft-sql` is narrower.

## Measured: the single-node competitive picture

This section began as a Daft comparison and now carries all three single-node engines,
because the question it answers moved. Everything below was taken at TPC-H sf1 on a 96-core
node with a **release** build, with the engine's mtime checked across each run — see
`benchmarks/run.py::_check_build_profile` for why that check exists.

`benchmarks/engines/daft.py` had existed for a while, but Daft sat only in the *multi*-node
default lineup, so the run everyone actually types (`python benchmarks/run.py`) never included
it. It is now in both tiers, because a claim no default run measures is a claim nobody checks.

Summary first, then per-competitor detail, then the per-query diagnoses.

### The full single-node picture, TPC-H sf1, 96 cores, release

All three engines, same run conditions, engine mtime verified unchanged. `b/x` below 1 means
Batcher is faster.

| Competitor | Batcher ahead on | Batcher behind on | Worst |
|---|---|---|---|
| Daft | 18 of 19 answered | q4 | 1.41x (q4) |
| Polars | 18 of 22 | q17, q14, q22, q5 | **2.41x (q17)** |
| DuckDB | 10 of 22 | 12 queries | 1.42x (q5) |

So the remaining single-node work is **DuckDB across the board** and **Polars on q17**. q5 is
the only query lost to two competitors at once.

Both DuckDB leads turn out to be already-known architectural items rather than unexplored
optimizations, which is worth recording so nobody re-derives them:

- **The JIT does not compile filters on the streaming path**, and that is deliberate:
  `crates/bc-interp/src/stream/mod.rs` records that wiring Tier-1 in "was tried and measured
  1.01x over TPC-H in an interleaved A/B, with five queries slower", because Arrow's
  compare/boolean kernels are already SIMD. Verified independently here — `bc-codegen`
  *accepts* both `int > int` and `date > date`, so the `interp` label in `explain(analyze=True)`
  is a choice, not a fallback. The `optimize-a-slow-query` heuristic "`backend` reads `interp`
  on a shape the JIT should compile → a gap worth closing" leads nowhere on this path.
- **The gather tax on high-selectivity filters** is analyzed with `perf` in
  `rfc-streaming-executor.md`: q1 at sf10 spends ~22% in the filter gather because the
  predicate passes ~98% of rows and the engine materializes them, where DuckDB carries a
  selection vector. That RFC is written and unaccepted; it is the lever, not a missing
  micro-optimization.

### TPC-H: the documented loss has reversed on a large machine

`docs/benchmarks/vs-daft.md` has a section titled "Where Daft wins: multi-join SQL",
reporting Batcher behind on q20 (2.03x), q3 (1.55x), q4 (1.51x), q17 (1.35x) and q5
(1.13x) — measured on a **16-core** node. Re-run here on **96 cores**, release build,
engine mtime verified unchanged across the run:

| Query | 16-core (that page) | 96-core (this run) |
|---|---|---|
| q20 | 2.03x behind | **0.80x — ahead** |
| q3 | 1.55x behind | **0.52x — ahead** |
| q4 | 1.51x behind | 1.41x behind |
| q17 | 1.35x behind | **0.20x — ahead** |
| q5 | 1.13x behind | **0.80x — ahead** |

Across the whole suite Batcher is ahead of Daft on 18 of the 19 queries both engines
answer, from 0.13x (q10) to 0.80x, with q4 the single remaining loss at 1.41x.

**This does not simply overwrite that page**, because it is different hardware and the two
can both be true. But it does contradict the *explanation* the page gives for the gap:
"single-node parallelism plateaus after about 8 cores where Daft uses effectively all 16".
If that were the whole story, 96 cores would make Batcher's position worse, not reverse it
on four of five queries. Either more has landed since, or the diagnosis needs revisiting.
Either way, quoting "Daft wins join-heavy SQL by 2-12x" is no longer defensible, and the
`migrate-from-daft` skill said exactly that.

### Daft fails or errors on 5 of 22 TPC-H queries

The harness gates timing on agreement with DuckDB, so these are its verdicts, not ours:

| Query | Daft |
|---|---|
| q6 | 75,207,768 where DuckDB says 123,141,078 — the `0.06 + 0.01` float fold `vs-daft.md` already documents |
| q15 | 0 rows where DuckDB returns 1 |
| q18 | wrong columns: `l_quantity` where the query asks for `sum(l_quantity)` |
| q21 | `DaftError::InternalError: Outer reference columns cannot be bound` |
| q22 | `InvalidSQLException: Unsupported SQL: SUBSTRING(expr FROM start FOR len)` |

Worth stating plainly because a parity ledger that only tracks *our* gaps is half a
picture: on the SQL surface both engines claim, Batcher answers 22 of 22 correctly and
Daft 17.

### A correctness divergence in Daft, found by running it

The harness refuses to time a query whose result disagrees with the oracle, and it caught
one: `op-window-runsum` reported `duckdb != daft`. Batcher agreed with DuckDB. Reduced to
three rows, and reproducible directly:

```sql
SELECT k, o, sum(v) OVER (PARTITION BY k ORDER BY o) AS rs FROM t ORDER BY o
```

with `k = [1,1,1]`, `o = [1,2,3]`, `v = [10, 20, 30]`:

| Engine | `rs` |
|---|---|
| DuckDB | 10, 30, 60 |
| Batcher | 10, 30, 60 |
| Daft 0.7.21 | 60, 60, 60 |

SQL's default frame for an aggregate with an `ORDER BY` is `RANGE BETWEEN UNBOUNDED
PRECEDING AND CURRENT ROW` — a *running* aggregate. Daft appears to apply the whole
partition, which is the frame you get with no `ORDER BY` at all. So Daft's `op-window-runsum`
timing is for a different (and cheaper) computation than the other two engines ran, and the
`0.01x` in that row should not be quoted.

This is recorded here rather than filed anywhere, because the point for Batcher is narrow:
the differential harness catches this class of thing, and it caught it in a competitor
before it could ever have caught it in us.

### Why the DuckDB gap is one item, not twelve

Profiled the three worst DuckDB losses individually, expecting three causes. They are mostly
one:

| Query | Bottleneck | Reading |
|---|---|---|
| q21 (1.37x) | `filter` 90 ms, 6 M rows in, 3.79 M out, 87 MB materialized | gather |
| q5 (1.42x) | `hash_join` 204 ms, 6 M probe emitting 1.2 M rows | gather |
| q17 (2.41x vs Polars) | two 6 M probes where one suffices | duplicated work |

The q5 number is the clearest evidence. The *same* 6 M-row probe costs 204 ms when it emits
1.2 M rows and 103 ms when it emits 6 k (q17's first join) — so the cost tracks the *output*
size, not the probe. That is materialization, which is exactly what
`rfc-streaming-executor.md` measured with `perf` on q1 (~22% of wall time in the filter
gather, because the predicate passes 98% of rows and the engine copies them where DuckDB
carries a selection vector).

So chasing individual queries is the wrong shape of work here: q5 and q21 are the same item,
and it already has an RFC. q17 is the exception and has its own cheap fix below.

### q17 vs Polars (2.41x): the 6 M-row probe runs twice

The largest single loss to any competitor, localized. `explain(analyze=True)` on q17 at sf1:

```
hash_join   103.2ms   lineitem(6,001,215) x filtered_part(204) -> 6,088 rows
  aggregate   3.0ms   -> 204 per-part averages
hash_join    96.2ms   lineitem(6,001,215) x those_204          -> 6,088 rows
```

**The same 6 M-row probe is paid twice**, and the second one reproduces the 6,088 rows the
first already had. That is the whole 2.41x: two probes where Polars does one.

Two ways out, and the cheap one is not the obvious one:

- **Plan-level CSE is the obvious answer and the wrong tool here.** The two joins are
  identical, so eliminating the duplicate looks like a job for common sub-*plan* elimination.
  But `kyber/rules/extra/cse.py` is deliberately expression-level (repeats inside one
  `Project`), and sharing a sub-plan needs the executor to feed one materialized intermediate
  to two consumers — a DAG where `bc_ir::RelOp` is a tree. That is an IR shape change, i.e. a
  wire-contract change, for one query shape.
- **Decorrelating differently costs nothing at the IR level.** The subquery correlates on
  `l_partkey`, which is *also* the join key, so the per-part average can be computed as a
  window over the already-joined 6,088 rows instead of by re-joining:

  ```
  lineitem x filtered_part -> 6,088 rows
    -> avg(l_quantity) OVER (PARTITION BY l_partkey)
    -> filter l_quantity < 0.2 * avg
    -> sum
  ```

  One probe, and every operator already exists. This is a rewrite in the SQL decorrelation
  path (or a Kyber rule over the join-aggregate-rejoin shape it produces), gated on the
  correlation key being a subset of the join key. The care needed is in scalar-subquery
  semantics — what an empty group yields — not in new machinery.

Not attempted here: it is a correctness-sensitive rewrite of subquery translation, and the
measurement above is what it needs to start from rather than a rushed patch.

### Where Batcher was behind: small-corpus image ingest (mostly closed)

The multimodal suite at scale 10 (100 JPEGs from S3), release build, `b/daft` above 1
meaning Batcher is slower:

| Query | before | after | daft_ms | b/daft before | b/daft after |
|---|---|---|---|---|---|
| img-list | 239.7 | 163.5 | 143.2 | 1.83x | **1.14x** |
| img-decode | 285.5 | 174.2 | 145.8 | 2.18x | **1.20x** |
| img-resize | 253.2 | 179.3 | 143.7 | 1.60x | **1.25x** |

**The cause was a stat per file that the fetch already answers.** `_chunks()` called
`probe_sizes` to size every file before packing them into batches — one object-store HEAD
per file — and then the read issued a GET per file that returns the length anyway. Two
round trips against the same latency where one would do. Instrumented on this corpus it
was **86 ms of a 272 ms read, 32%**, and the docstring justifying it ("one stat per file is
negligible next to what a media source already does per file") was reasoning about *bytes
transferred* when the binding cost is *round trips*.

`read()` now fetches first and fills a size cache from what came back, so `_chunks()` needs
no stat. `iter_batches` still probes and must: it bounds memory before fetching, which is
the whole point of a byte bound on the streaming path. Chunk boundaries are unchanged —
only the source of the sizes moved — so `read`, `iter_batches` and `splits` still share the
one chunk definition, which the module docstring requires.

Batcher remains ~1.2x behind, so this is improved rather than won. What is left is inside
the fetch itself (163 ms for 100 GETs against Daft's 143 ms).

The diagnosis came from the *shape*, not the totals. `img-list` does no image work at all —
it lists and fetches bytes — and Batcher was 1.8x behind there. Decode and resize then cost
Batcher almost nothing on top (`img-resize` is *faster* than `img-decode`, the DCT-scaled
JPEG path doing its job), while Daft's resize adds ~30 ms. That located the whole loss in
per-file fetch and none of it in the image kernels, which are the better ones.

Also checked along the way, and recorded so nobody re-chases them:

1. **Serial header parsing** — metadata extraction ran in a Python loop after the
   concurrent fetch, one `PIL.Image.open` per file on one thread. Fusing it into the pool
   task is right on its own terms but moved the number only a few percent, inside the
   spread. It was not the cause.
2. **Benchmark fairness** — the obvious excuse is that Batcher expands a glob while Daft is
   handed a URI list. It does not hold: `ImageCorpus.uris()` calls `open()`, which calls
   `_list_corpus` with no caching, inside the timed function. Both engines list.

The lesson for the remaining 1.2x: the first two attempts were hypotheses and both missed;
the third came from instrumenting the read and reading the clock. Profile first.

This does **not** contradict `docs/benchmarks/vs-daft.md`, which reports Batcher well ahead
on multimodal ingest: that measurement is 2,000 frames on a 96-core node, a regime where
per-file latency amortizes and Batcher's kernels dominate. The two results are consistent
and describe different ends of the corpus-size range. What is now known is that the small
end belongs to Daft.

### The per-batch Python callback: 2.3-2.6x recovered

`vs-daft.md` records Daft ~2x faster on a per-batch numpy UDF. Measuring the batch size
Batcher hands such a callback found the cause, and it was a constant set from a claim that
does not hold. `strategy.py` said of its 131,072-row floor: "throughput is flat from here
up, so this is the smallest batch on the plateau." Re-measured on a 12 M-row numpy
map+reduce (29 B/row, best-of-5):

| rows per call | wall |
|---|---|
| 131,072 *(old default)* | 39.9 ms |
| 262,144 *(old cap)* | 28.4 ms |
| 524,288 | 19.2 ms |
| **1,048,576** | **14.9 ms** |
| 2,097,152 | 15.6 ms |
| 4,194,304 | 40.3 ms |

Not flat — a genuine optimum. Coarser amortizes more per-call overhead until concurrency
starves: at 3 batches over 96 cores the GIL-bound per-call Arrow conversion serializes and it
is worse than the morsel. 1 M is the bottom, and it is the same figure on a 6 M-row input, so
it is a per-call row count rather than a function of the total.

The old *cap* could not simply be raised, because a row count cannot bound memory — 1 M rows
is 29 MB of narrow numerics and gigabytes of decoded frames, and the row cap was protecting
the wide case at the narrow case's expense. The target is now byte-bounded, the same rule
`udf/stream.py` already applies to its CPU chunks. A heavy `fn` keeps the per-worker split
and the old cap: the measurement is about per-call overhead on the light path, and raising the
core-filling path's footprint is a separate claim nothing here tests.

Result: 6 M rows 22.3 → 9.7 ms, 12 M rows 39.5 → 15.2 ms. Wide-row callbacks are unchanged
(still morsel-sized, 13 calls of 16,384 on the 4 KB/row case) — the byte bound is a guard
there, not a speedup.

### An oddity worth recording rather than explaining away

An earlier pass of this same suite was run before the build profile was verified, and its
numbers are within noise of the release ones above. That is not what a dev-profile build
(`opt-level = 0`, `debug_assertions` on, dependencies unoptimized) should produce, and it
is the reason the old timing-heuristic guard never fired: on these queries the profile
barely moved the clock. Why is not established here, and guessing at it would be worse
than leaving it open. The lesson taken is narrower and solid: **infer the build profile
from the build, never from a stopwatch.**

## Where DuckDB actually stands, replicated

The earlier reading of "trails DuckDB on 12 of 22" came from a single harness pass taken while
the box carried a 15-minute load average of 17.6. Two independent passes on a quieter box, both
with every correctness check passing, put it differently: **the per-query loss list is six, not
twelve, and the aggregate is a narrow Batcher lead.**

| | pass 1 | pass 2 |
|---|---|---|
| geomean ratio (batcher / duckdb) | 0.964x | 0.990x |
| total wall time | 555.7 vs 585.4 ms | 641.4 vs 631.1 ms |

The geomean is below 1.0 in both passes, so a few percent ahead overall is supportable. The
**total** flips sign between them, so it is not — quote the geomean or nothing.

Six queries lose in *both* passes, and this is the list worth working: q4 ~1.45x, q12 ~1.43x,
q6 ~1.40x, q5 ~1.30x, q22 ~1.30x, q21 ~1.29x. Six win in both: q16 ~0.59x, q2 and q8 ~0.65x,
q17 ~0.68x, q10, q13. The remainder sit at parity or swing, and a single pass cannot tell which.

### Two ways an in-process optimizer A/B lies, and both shipped a regression

These are worse than the two below, because they do not merely waste time — they produced a
green verdict for a change that cost ~9%, and it landed.

- **A monkeypatch A/B must clear the plan cache on every arm switch.** `plan_cache.cache_key`
  folds in `learning.generation()`. Stubbing an estimator's *consumer* does not bump that
  counter, so a plan optimized under one arm is served to the other and the arms stop
  differing. Call `plan_cache.clear()` in the arm switch. Without it, two separate "timing
  neutral" verdicts were measuring a contaminated mixture.
- **A flat percentage noise floor is wrong for tight distributions.** The ~30% floor below is
  calibrated for the *bimodal* queries (q19 is min 14.8ms against a median of 52ms). Applied
  to a tight distribution it hides real effects: a consistent +12-28% across 18 of 22
  queries, min and median agreeing, was labelled "noise" by a 30% rule. **Sign consistency
  across many queries is the stronger signal** — 18 of 22 one-directional is ~0.1% by chance.
  Look at that and at min-versus-median agreement, not a single magnitude cutoff.

Clearing the plan cache also changes what is being measured, and that is the point: it moves
the comparison onto **cold-plan** cost, where optimizer overhead shows up. q2 goes from 12ms
to 112ms under it. A change that only touches planning is invisible until you do this.

### Two ways this benchmark manufactures a false result

Both were hit while producing the table above, and both survive a warm-up and a best-of-N:

- **A delta the mechanism cannot have caused is noise, whatever its size.** Forcing the runtime
  join filter on appeared to cost q4 +50% and q13 +36%. Both plans show *no filter engaged* —
  q4 builds from a 3.79M-row side whose distinct keys exceed `MAX_DISTINCT_KEYS`, and q13's only
  join is a `LEFT`, which is not reducible. Check `explain(analyze=True)` for whether the thing
  under test is active *before* believing a timing about it.
- **Some queries are bimodal, so min-of-N invents winners.** q19 measures min 14.8ms against a
  median of 52ms in the *same* arm. Pairing one arm's unlucky min against the other's lucky min
  produced an apparent 72% win from byte-identical behaviour. Report the median beside the min.

### Runtime join filters: the row reductions are real, the wall clock does not move

`bc-interp`'s `stream/runtime_filter.rs` sinks a join's build-side key set down the probe pipeline
to the scan, and records as an open item that "the wall-clock effect at scale was not measurable
on the benchmark box." That reproduces, on a release build at sf1, on a quieter box.

What the filter buys is certain, because it is a count. Forcing it on in q19:

| | rows through the `lineitem` filter | bytes | that filter's CPU |
|---|---|---|---|
| off | 1,500,048 | 88 MB | 255.4 ms |
| on | 180,632 | 11 MB | 20.0 ms |

An 8.3x row reduction, 235 ms of CPU removed, and the join above it 13.2 -> 5.0 ms. And no
wall-clock effect whatsoever: 10 alternating pairs give off min 14.77 / median 52.11 against on
min 14.80 / median 50.65, and the 22-query total moves +0.1%.

The reason is in those medians. q19 is bimodal at this scale, so it is not throughput-bound, and
CPU taken off the probe is not CPU the query was waiting on. Anyone tempted to widen the
engagement gate should note that the argument for doing so is sound in kind — the economics are
"small selective build side meets large probe side", and the query-global row gate tests neither
half against the join in question — while the evidence for a *speedup* at this scale is absent.
sf10+, or hardware where the probe genuinely is throughput-bound, is where that would be settled.

### The filter-selectivity half of the feedback loop was not connected

The moat's claim is that plans improve as a query is re-run. For filter cardinality that was
not happening, and the reason was a missing join rather than a missing measurement.

Core records every filter's measured `rows_out / rows_in` under the stable signature
`annotate_ops` stamps, on every execution, profiled or not. `StatsEstimator._selectivity`
reads exactly that per-signature `selectivity` key, and a measured value there always beats
the structural guess. But the only writer of the key was `learning.record_selectivity`, which
is handed the **query's** final row count and therefore guards on `_filter_over_scan`: the
whole plan must be a filter over a single scan. Every filter beneath a join, aggregate, sort
or limit — 21 of the 22 TPC-H queries — re-derived a guess the engine had already measured,
on every run, forever.

Verified before the fix: three runs of q12 left six signed filter rows in the hub carrying
measured selectivities of 0.0869 and 0.0594, **none** of whose signatures held a `selectivity`
entry, with the estimate frozen at 0.327 across ten consecutive runs.

`kyber/measured_selectivity.py` derives the key from that history — the same shape as
`learning._cardinality_corrections`, and the only correct layering, since Kyber cannot hook
`core`'s recording. Folded in with `setdefault`, so `record_selectivity` keeps precedence.
After two observations the estimates become exact:

| | before | after |
|---|---|---|
| q12 filter | est 1,962,347 / actual 521,289 | est 521,289 |
| q12 build side | `right≈205,429 [default]` | `right≈30,988 [learned]` |
| q4 filter | est 3,040,569 / actual 3,793,296 | est 3,793,296 |
| q4 build side | `left≈386,279 [default]` | `left≈57,218 [learned]` |

Wall clock is neutral across 12 queries. Two things it cannot do: help a shape's first
execution, by construction; and model correlation, because a signature is structural — that
is a separate gap, still open, and the reason q12's chained
`l_shipdate < l_commitdate < l_receiptdate` remains mis-estimated on a cold plan.

### Reusing a `Dataset` is not the same measurement

q12 runs ~30ms when the `Dataset` is rebuilt each time and ~14ms when one `Dataset` is collected
repeatedly — a 2x gap present from the first reuse, so not convergence. It is **not** the learned
statistics loop: the `fresh` arm never trends downward across 14 iterations, and q12's build-side
decision still reports `[default]` provenance afterwards. q4, q6 and q21 show no such gap. The
cause is unidentified and deliberately not guessed at here. What matters for quoting numbers is
that the rebuilt-plan figure is the honest one, because it is what the harness and a user both
do; a repeated-collect loop measures something else and will flatter the engine by ~2x on q12.

## What this ledger does not claim

It does not claim Batcher beats every competitor. As of the last run it leads Polars on 18 of
22 TPC-H queries and Daft on 18 of the 19 it answers, and trails DuckDB on the six queries named
above — a gap whose largest component is one architectural item (`rfc-streaming-executor.md`,
Proposal 2), not a list of optimizations. Anyone quoting a number from here should quote the
machine and scale factor with it: the join-heavy comparison against Daft reverses between 16 and
96 cores, which is the whole reason this section states its conditions twice.
