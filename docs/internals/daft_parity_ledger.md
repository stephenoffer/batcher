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

### Note on the epoch gap

`from_epoch` is the one item so far that closed a *correctness* trap rather than a missing
convenience. An `Int64` column of epoch counts carries no record of its unit, so
`cast("timestamp")` has to assume one, and Arrow assumes microseconds. A column of epoch
seconds cast that way silently became January 1970 — a plausible-looking timestamp, no
error. The behaviour is pinned in
`test_a_bare_cast_would_have_read_epoch_seconds_as_microseconds` so the function cannot be
refactored back into a cast.

## Open — real capability gaps

Ranked by value. "Daft" names the Daft function the gap was found from.

| Gap | Daft | Notes |
|---|---|---|
| Binary compression codecs | `compress`, `decompress`, `try_*` | gzip / zlib / deflate / brotli / zstd, both directions. Batcher has base64/hex only. |
| Iceberg partition transforms | `partition_iceberg_bucket`, `partition_iceberg_truncate`, `partition_{years,months,days,hours}` | Needed to write a table another engine's Iceberg reader will prune correctly. |
| Row-identity generators | `monotonically_increasing_id`, `uuid`, `random_int` | Batcher has `with_row_index`; the expression-level forms are absent. |
| Image ops | `crop`, `convert_image`, `encode_image`, `image_{width,height,mode,channel}` | Batcher has decode/resize/center_crop/to_tensor/dhash; the attribute readers and the encode direction are missing. |
| Video frame access | `video_frames`, `video_keyframes`, `get_video_frame_by_idx`, `video_metadata` | Batcher decodes video but exposes no frame-level accessor. |
| Audio metadata | `audio_metadata` | Batcher has decode/resample/mel/mfcc but no metadata reader. |
| File metadata | `file_size`, `file_exists`, `file_path`, `guess_mime_type` | Path-level facts as expressions. |
| HDF5 accessors | `hdf5_attrs`, `hdf5_keys`, `hdf5_metadata` | Batcher reads HDF5 as a *format*; the per-row accessors are absent. |
| Great-circle distance | `great_circle_distance` | Batcher has L1/L2/cosine/hamming/jaccard, no geodesic. |
| Regex split | `regexp_split` | Batcher's `split` is literal-delimiter only. |
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

| Query | batcher_ms | duckdb_ms | polars_ms | daft_ms | b/daft |
|---|---|---|---|---|---|
| op-groupby-sum | 3.7 | 3.2 | 15.8 | 27.0 | 0.14x |
| op-groupby-2key | 5.4 | 5.1 | 21.9 | 37.2 | 0.15x |
| op-global-sum | 0.1 | 1.2 | 1.3 | 3.2 | 0.04x |
| op-filter-count | 0.2 | 1.3 | 10.8 | 5.6 | 0.04x |
| op-join-agg | 39.7 | 41.8 | 52.5 | 240.9 | 0.16x |
| op-sort-limit | 8.6 | 6.1 | 230.8 | 45.8 | 0.19x |
| op-filter-project | 8.6 | 8.0 | 12.0 | 13.4 | 0.64x |
| op-window-rank | 43.5 | 121.7 | 908.3 | 7322.4 | 0.01x |
| op-window-runsum | 43.1 | 251.1 | 821.6 | 6867.1 | 0.01x |
| op-window-lag | 61.5 | 156.3 | 2679.7 | 6794.5 | 0.01x |
| op-window-sum-partition | 38.2 | 49.9 | 52.0 | 2535.8 | 0.02x |

Batcher is faster on all eleven, by 1.6x on the narrowest (`op-filter-project`) and ~100x
on the window functions. Read the window column with the caveat below, though — one of
those four is not measuring the same computation on both sides.

Two limits on what this table supports: it is one machine and one scale factor, and it is
the operator suite rather than full TPC-H. It is enough to say Batcher is not *behind*
Daft on these shapes; it is not enough for a headline multiplier.

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

It does not claim Batcher is faster than Daft beyond the table above: one machine, one
scale factor, one suite. `competitive_architecture.md` is the file that carries competitive
claims; this one records measurements and the conditions they were taken under.
