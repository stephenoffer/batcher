# Daft parity ledger

**Status:** open, started 2026-07-24. A running record of the Daft-versus-Batcher gap and
what has been closed. Read `competitive_architecture.md` for the scorecard and
`competitor_technique_review.md` for the mechanism-level parts list; this file is narrower
than either. It answers one question: what can a Daft user do that a Batcher user cannot,
and what did we do about it.

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

| Gap | Daft | Notes |
|---|---|---|
| Iceberg partition transforms | `partition_iceberg_bucket`, `partition_iceberg_truncate`, `partition_{years,months,days,hours}` | **Lower value than it first looks.** Batcher's Iceberg sink hands each shard to `pyiceberg`, which applies the table's own partition-spec transforms, so a partitioned write is already correct without these. What is left is manual bucketing and bucketed joins. The temporal four compose from existing expressions; only `bucket` needs an exact `murmur3_x86_32`, which is not in the tree. |
| Group-wise Python apply | `map_groups` | **The largest genuine gap found.** `GroupBy` exposes 24 reducers and no way to hand a whole group to a Python function — pandas `groupby().apply()`, Polars `group_by().map_groups()`, Spark `applyInPandas`. `map_batches` operates on arbitrary batches, which is not the same thing: a group can span batches. Correct here means shuffling so each group lands whole in one partition before the callback runs, which is a distributed-execution change, not a `GroupBy` method. Deliberately **not** attempted in a hurry: a version that works single-node and silently splits groups across workers is exactly the failure CLAUDE.md's mergeability guard names — wrong results at cluster scale, no error. |
| Row-identity generators | `monotonically_increasing_id`, `uuid`, `random_int` | Batcher has `with_row_index`; the expression-level forms are absent. |
| Image color conversion | `convert_image` | An explicit mode conversion (RGB to L, RGBA to RGB). `.image.to_grayscale` covers the common case; the general form is open. |
| Video frame access | `video_frames`, `video_keyframes`, `get_video_frame_by_idx` | `.video.decode()` already covers `video_metadata` (it returns `{width, height, num_frames, duration_secs, fps}`), so the real gap is frame *extraction*, not metadata. Needs the `video` cargo feature (system FFmpeg). |
| File metadata | `file_size`, `file_exists`, `file_path`, `guess_mime_type` | Path-level facts as expressions. **Costed and deferred:** each is a `stat` per row, which means object-store access from `bc-expr` — a crate that sits under everything and deliberately links no IO. The right home is `bc-io` or the UDF plane, and that is a design decision rather than a function to add. |
| HDF5 accessors | `hdf5_attrs`, `hdf5_keys`, `hdf5_metadata` | Batcher reads HDF5 as a *format*; the per-row accessors are absent. |
| Tokenizer round trip | `tokenize_encode`, `tokenize_decode` | Batcher estimates token counts; it cannot produce or consume real BPE ids. Needs a tokenizer dependency in the data plane — costed, not yet decided. |
| `jq` | `jq` | A full jq engine. Recorded so the gap is honest; the shape accessors above cover the common cases. |
| Connectors | `write_turbopuffer`, `write_bigtable`, `write_paimon`, WARC, MCAP, HuggingFace | Batcher's connector surface is wider overall (nine categories, 76 registered formats); these six are Daft-only. |

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

## Measured: Batcher against Daft

`benchmarks/engines/daft.py` has existed for a while, but Daft was only in the *multi*-node
default lineup, so the run everyone actually types (`python benchmarks/run.py`) never
included it. It is now in both tiers, because a claim no default run measures is a claim
nobody checks.

The operator suite at TPC-H scale factor 1, best-of-5, on this machine
(`python benchmarks/run.py --benchmark operators --engines batcher,duckdb,polars,daft
--scale 1`). `b/daft` below 1 means Batcher is faster by that factor.

The run below printed `release` from `bt.versions()["engine_profile"]` before starting, and
the suite now refuses to run otherwise (`benchmarks/run.py::_check_build_profile`).

| Query | batcher_ms | duckdb_ms | polars_ms | daft_ms | b/daft |
|---|---|---|---|---|---|
| op-groupby-sum | 3.8 | 3.2 | 15.9 | 26.8 | 0.14x |
| op-groupby-2key | 7.7 | 5.0 | 19.7 | 36.8 | 0.21x |
| op-global-sum | 0.1 | 1.5 | 1.2 | 4.0 | 0.04x |
| op-filter-count | 0.2 | 1.3 | 11.5 | 6.2 | 0.04x |
| op-join-agg | 39.5 | 43.4 | 50.1 | 238.8 | 0.17x |
| op-sort-limit | 7.4 | 6.0 | 230.9 | 46.1 | 0.16x |
| op-filter-project | 9.4 | 7.1 | 16.0 | 13.2 | 0.71x |
| op-window-rank | 42.4 | 117.9 | 851.7 | 6903.9 | 0.01x |
| op-window-runsum | 39.4 | 237.9 | 861.9 | 7480.2 | 0.01x |
| op-window-lag | 61.1 | 150.8 | 2655.5 | 6958.8 | 0.01x |
| op-window-sum-partition | 35.9 | 44.7 | 49.0 | 2662.5 | 0.01x |

Batcher is ahead on all eleven, from 1.4x on the narrowest (`op-filter-project`) to two
orders of magnitude on the window functions. Discount the window column, though: one of
those four is not computing the same thing on both sides (below), and the other three
share its kernel.

Three limits on what this supports: one machine, one scale factor, and the operator suite
rather than full TPC-H. `docs/benchmarks/vs-daft.md` remains the scorecard, and it reports
Daft ahead on join-heavy TPC-H — nothing here contradicts that, because none of these
eleven is a multi-join query.

### An oddity worth recording rather than explaining away

An earlier pass of this same suite was run before the build profile was verified, and its
numbers are within noise of the release ones above. That is not what a dev-profile build
(`opt-level = 0`, `debug_assertions` on, dependencies unoptimized) should produce, and it
is the reason the old timing-heuristic guard never fired: on these queries the profile
barely moved the clock. Why is not established here, and guessing at it would be worse
than leaving it open. The lesson taken is narrower and solid: **infer the build profile
from the build, never from a stopwatch.**

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

## What this ledger does not claim

It does not claim Batcher is faster than Daft in general. It reports one suite, on one
machine, at one scale factor, on a verified release build — eleven operator queries, none
of them a multi-join. `docs/benchmarks/vs-daft.md` is the scorecard, and it says Daft leads
on join-heavy TPC-H; this ledger does not amend that and should not be quoted as if it
did.
