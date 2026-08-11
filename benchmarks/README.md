# Batcher benchmark suite

A correctness-first, multi-engine comparison of the Batcher data engine on the
workloads the industry actually cites — **TPC-H** (all 22 queries), **TPC-DS** (all 99),
**ClickBench** (43 queries), the **Join Order Benchmark** (all 113, over the real IMDb
database), the **H2O.ai db-benchmark** (its 10 groupby and 5 join questions), an
**operator-mix** of single relational operators, a **JSON** suite for semistructured
parsing, a **scan** benchmark over three parquet file layouts, and an **images** benchmark
for unstructured multimodal ingest — against the engines Batcher claims to beat:

| Tier            | Engines                                              |
|-----------------|------------------------------------------------------|
| **Single-node** | batcher, duckdb, `duckdb_arrow`, polars, pyarrow, **pyspark** (opt-in) |
| **Multi-node**  | batcher (distributed), daft, **pyspark** (opt-in) |

`duckdb` runs on its compressed *native* store (DuckDB at its best); `duckdb_arrow` runs the
same query on the *same zero-copy Arrow* Batcher consumes — the like-for-like execution bar.
Reporting both separates DuckDB's storage engine from its execution engine (see
`TPCH_FINDINGS.md`).

Correctness is checked before any timing is trusted: a query is only timed once its
result matches the reference engine, so a fast wrong answer can never be reported as
a win.

## No invented data — public sources, or the benchmark's own generator

The suite never invents a substrate to benchmark on. Every table is either read from a
canonical public parquet location and normalized once (`sources/`), or produced by the
**benchmark's own published generator** — which is how those benchmarks are specified, and
what every published result for them does:

| Dataset    | Default source                                                                 | Access |
|------------|--------------------------------------------------------------------------------|--------|
| TPC-H      | `s3://ray-benchmark-data/tpch/parquet/sf{scale}/{table}/`                       | S3 (creds/region may be needed) |
| ClickBench | `https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_*.parquet` | anonymous HTTPS |
| TPC-DS     | `~/bench-data/tpcds/sf{scale}/` — materialized once by the spec's own `dsdgen` (DuckDB's `tpcds` extension); no public parquet mirror exists | local, generated on first run |
| Scan       | `s3://ray-benchmark-data/{parquet,parquet/128MiB-file,small-parquet}/{size}/`   | S3 (creds/region may be needed) |
| JOB (IMDb) | `https://event.cwi.nl/da/job/imdb.tgz` — the archive the reference implementation distributes, converted to parquet at `~/bench-data/job/parquet/` on first run | HTTPS, ~1.2 GiB once |
| H2O.ai     | built in-process to the benchmark's `groupby-datagen.R` / `join-datagen.R` spec (`datagen/h2o_tables.py`); db-benchmark ships no data | fixed seed, in memory |
| JSON       | built in-process (`datagen/json_events.py`); no public nested-JSON corpus exists | fixed seed, in memory |

The generated datasets are fixed-seed and shared as one Arrow table across every engine, so
the correctness gate compares engines on byte-identical input. They are not byte-identical
to another project's copy of the same generator, so their absolute times are comparable
**across the engines in a run**, not against a published leaderboard.

Loading uses DuckDB's `httpfs` (already a core dependency), which reads local paths,
`s3://`, and `https://` directly. Override the base URI, scale, or ClickBench
partition count without touching code:

```bash
export BENCH_TPCH_BASE=s3://my-mirror/tpch/parquet     # or --source on the CLI
export BENCH_CLICKBENCH_PARTS=10                        # read 10 hits partitions
export BENCH_SCAN_BASE=s3://my-mirror                   # scan corpus bucket root
export BENCH_S3_REGION=us-east-1                        # for S3 sources
```

### Standard benchmarks deliberately not wired up

Each of these was evaluated and rejected for a specific, checkable reason. Re-check the
reason before re-treading one; do not add a suite whose data cannot be obtained, because a
benchmark that never runs is worse than an absent one — it reads as coverage.

| Benchmark | Why not |
|---|---|
| Nexmark (22 streaming queries) | Written in Flink SQL (`TUMBLE`/`HOP`/`MATCH_RECOGNIZE` over a `datagen` connector), and its generator is Java with no published spec to follow. Roughly half the queries would not translate, and reproducing the generator from its source is the kind of guess this suite does not make. |
| Star Schema Benchmark (13 queries) | Needs `ssb-dbgen`, a C program with no packaged build here and no public parquet mirror. Deriving `lineorder` from TPC-H is a transformation the SSB paper describes but does not specify precisely enough to reproduce faithfully. |
| TPCx-BB / BigBench, TPCx-AI | Data comes from PDGF, a Java generator that is not redistributable. |
| TSBS (time series) | Go generator, no published data. Worth revisiting if the generator's spec is documented. |

## The scan benchmark: one table, three file layouts

TPC-H and ClickBench read a handful of well-sized files, so they never price the work
an engine does *before* it reads a value: listing the files, opening each footer, and
planning the scan. Real lakehouse tables are frequently thousands of small files, and
that fixed per-file cost is where engines diverge by orders of magnitude.

The Ray bucket stores one 16-column `int64` dataset three ways, which makes the file
layout the only variable:

| Family            | Source                             | Files at scale 1 | File size |
|-------------------|------------------------------------|------------------|-----------|
| `scan-one_big`    | `parquet/{size}/`                  | 1                | ~1 GiB    |
| `scan-ideal`      | `parquet/128MiB-file/{size}/`      | 8                | ~132 MiB  |
| `scan-many_small` | `small-parquet/{size}/`            | 1,024            | ~1.2 MiB  |

At scale 1 and 10 all three hold an **identical row count** (8,388,608 and 83,886,080),
so the `_ms` columns are comparable across families. At scale 100 and above the
many-small corpus is *not* row-count-equivalent (it mixes in a few ~133 MiB files, which
`sources/` filters to keep the layout genuinely many-small), so there the cross-family
comparison is indicative and the per-engine one within a family still exact.

Nine shapes run against each layout, chosen to separate the costs a layout moves:
`count`/`minmax` (listing + footer metadata only), `sum1` (projection pushdown, 1 of 16
columns), `sumwide` (I/O-bound, all 16), `filter`/`filter_agg` (~1% selectivity, row-group
skipping), and `groupby`/`distinct`/`topn` (cost past the scan, in the operators).

Each engine's reader is rebuilt **inside** the timed call, so listing and metadata are
measured rather than amortized away — the opposite of the `--scan` mode used for TPC-H,
where the scan is a fixed setup cost shared across 22 queries.

Every SQL engine (Batcher, DuckDB, Polars, Daft, Spark) runs all nine shapes, and PyArrow
runs all nine through Acero. An engine whose API does not express a shape directly reports
`n/a` for it rather than a hand-rolled reimplementation that would benchmark the benchmark
instead of the engine.

```bash
python3 benchmarks/run.py --benchmark scan                        # all three layouts
python3 benchmarks/run.py --benchmark scan --family scan-many_small
python3 benchmarks/run.py --benchmark scan --only scan-count      # one shape, 3 layouts
python3 benchmarks/run.py --benchmark scan --scale 10             # 10 GiB corpus
```

Because every repeat re-reads the corpus from object storage, this benchmark runs
best-of-2 (best-of-1 above scale 10) rather than the best-of-5 the in-memory suites use.
For the same reason `scan` is **excluded from `--benchmark all`** and must be asked for
by name: at scale 1 the many-small-files family alone takes tens of minutes.

## The Join Order Benchmark: the optimizer test, on real data

TPC-H and TPC-DS generate their data from uniform, independent distributions, which is
precisely the assumption a textbook cost model makes — so they flatter cardinality
estimation. JOB does not. Its 113 queries run over a real 2014 IMDb snapshot where
predicates are correlated the way real data is, and Leis et al. built it to show that
estimation error, not the join-order search, is where optimizers actually lose.

That makes it the benchmark aimed most directly at Batcher's stated moat. Re-optimizing on
*measured* cardinalities at pipeline breakers is only worth its complexity if estimates are
badly wrong somewhere, and JOB is the workload where they are. A loss here is more
interesting than a win on TPC-H.

Every query has the same shape: `SELECT MIN(...) FROM a, b, ... WHERE <equi-joins and
filters>`, joining 3 to 16 tables and returning a single row. That is deliberate on the
benchmark's part — the result is trivial to compare, so what gets measured is the plan.

The data is the archive the reference implementation distributes: 21 tables, 3.6 GiB of CSV,
converted once to 1.8 GiB of parquet under `~/bench-data/job/parquet/`. Column names and
types are read from the `schematext.sql` shipped *inside* that archive, never transcribed.
The suite is **excluded from `--benchmark all`** because of the one-time 1.2 GiB download.

```bash
python3 benchmarks/run.py --benchmark job                      # all 113 queries
python3 benchmarks/run.py --benchmark job --only job-q1        # q1a-q1d (and q10-q19: substring)
python3 benchmarks/run.py --benchmark job --skip job-q7c       # complete a run around a fatal query
BENCH_JOB_LOCAL=/data/job python3 benchmarks/run.py --benchmark job   # relocate the mirror
```

> **Batcher currently cannot finish this suite.** Two full runs were **OOM-killed** — at
> `job-q7c`, and with that skipped at `job-q10a` — on a 30 GiB box, where DuckDB answers
> q7c in 0.43 s. Both are many-way joins whose predicates are all top-level equalities, so
> this is join ordering, not the disjunction problem TPC-DS q13 turned out to be. Pinning
> `--memory-bytes` does not contain it: the allocation is outside the path Carbonite bounds,
> so the process dies rather than spilling. Use `--skip` to complete a run around a fatal
> query; `BENCHMARK_RESULTS.md` has the detail.

## The H2O.ai db-benchmark: groupby and join at dataframe scale

TPC-H and TPC-DS ask a snowflake schema hard questions. `db-benchmark` asks one wide table a
different kind: aggregate it by keys whose cardinality spans three orders of magnitude (100
groups against N/100), then join it against three tables spanning six. It is where Polars,
DuckDB, data.table, pandas, Spark and Dask publish comparable numbers, and it isolates the
two things TPC-* buries inside larger plans — group-by state management, and join
build-side selection.

| Dataset       | Cases | Shape |
|---------------|-------|-------|
| `h2o-groupby` | 10    | one table `x`; keys from 100 groups to N/100, plus a median/stddev pair, a correlation, a top-2-per-group window, and a group-by on all six keys |
| `h2o-join`    | 5     | LHS `x` joined against `small` (N/1e6 rows), `medium` (N/1e3) and `big` (N), inner and left, on integer and string keys |

`--scale` selects the benchmark's own row tier: **1 → 1e7 rows** (its smallest published
size, the default), 10 → 1e8. Both suites are **excluded from `--benchmark all`**, and
`h2o-join` in particular is not a quick check: every RHS has a unique join key, so all five
questions return about as many rows as they read, and the correctness gate canonicalizes and
sorts every one of those rows for each engine before it will report a timing. A three-engine lineup at scale 1 held ~18 GiB
resident here. Drop to two engines, or to `--scale 0.1`, on a smaller box.

The generator (`datagen/h2o_tables.py`) follows the benchmark's published
`groupby-datagen.R` / `join-datagen.R` column for column, including the join key split that
gives each side 10% of keys the other lacks — so an inner join genuinely drops rows and a
left join genuinely produces nulls.

```bash
python3 benchmarks/run.py --benchmark h2o-groupby                 # 1e7 rows, 10 questions
python3 benchmarks/run.py --benchmark h2o-join --scale 0.01       # 1e5 rows, a fast check
python3 benchmarks/run.py --benchmark h2o-join --only h2o-join-q5 # one question
```

## The images benchmark: unstructured multimodal ingest

The structured suites never touch non-tabular data, yet the engine is built for multimodal
workloads. The `images` benchmark reads the `profile-pictures` corpus in the Ray bucket
(211,742 JPEGs, 110×110 RGB, ~5 KiB each) and runs the three stages of a real
image-preprocessing pipeline, across every engine with a multimodal path:

| Shape        | Work                                             | Engines |
|--------------|--------------------------------------------------|---------|
| `img-list`   | list files + read bytes (no decode)              | batcher, ray, daft, pyarrow |
| `img-decode` | decode each JPEG to an `(H, W, 3)` tensor        | batcher, ray, daft |
| `img-resize` | decode + resize to `224×224` (model-input prep)  | batcher, ray, daft |

DuckDB and Polars have no image path and sit out. Pixel-exact agreement across engines is
not sound (JPEG decoders and resize kernels differ), so each shape returns a small aggregate
the engines *do* agree on: the image **count**, the exact **total bytes** (a strong content
check for `img-list`), and the produced **height/width** — enough to prove every engine read
and processed the same corpus before its throughput is trusted, the discipline the GPU
cluster benchmarks already use. Both decode shapes pass an explicit target size to every
engine (Batcher's `read.images` requires the decode size be given), so all do the identical
operation.

`--scale` sets the image count via a filename-prefix glob (1 → 10 images, 10 → 100,
100 → 1,000, …). The default is small and the suite is **opt-in** (excluded from
`--benchmark all`) because image reads are per-file: a corpus of thousands of tiny objects is
minutes of wall-clock. `BENCH_IMAGES_BASE` overrides the corpus root (point it at a local
directory of `.jpg`s for a fast offline run).

```bash
python3 benchmarks/run.py --benchmark images                       # 10 images, all shapes
python3 benchmarks/run.py --benchmark images --scale 10            # 100 images
python3 benchmarks/run.py --benchmark images --only img-resize     # one shape
BENCH_IMAGES_BASE=/data/jpgs python3 benchmarks/run.py --benchmark images  # local corpus
```

Note: Batcher's image reader is fast on local files but currently pays a large **per-file
open cost over S3** (see the `cluster/` GPU suites for the in-memory multimodal *compute*
comparison, which is where Batcher's warm-pool / streaming moat shows). This suite measures
the read/decode path honestly, S3 penalty included.

### Physical-AI ingest scenarios (self-contained, local, no S3)

For a fast offline read of the multimodal ingest story, `benchmarks/scenarios/` holds
single-file, correctness-gated head-to-heads that synthesize their own local corpus:

```bash
python benchmarks/scenarios/image_decode.py       # JPEG decode+resize vs Daft
python benchmarks/scenarios/point_cloud_load.py   # LiDAR .npy -> torch tensors
python benchmarks/scenarios/audio_decode.py       # native audio decode vs a soundfile loop
python benchmarks/scenarios/robotics/sweep_transform.py  # LiDAR sweep -> world frame, vs NumPy
```

On a 96-core node these show batcher **2.4× faster than Daft** on image decode+resize,
and they exercise the physical-AI (camera / LiDAR / audio) ingest path. Findings and the
fix chain are in
`BENCHMARK_RESULTS.md`; the mechanism is documented in `docs/user-guide/operate/tuning/performance.md`
("Multimodal & physical-AI ingest").

### Small-query latency (the transactional shape)

Every other suite here measures throughput: one large query, timed once. That is blind to
the workload an OLTP-shaped application actually issues — thousands of tiny queries where
the result is a handful of rows and nearly all the elapsed time is control plane.

```bash
python benchmarks/scenarios/latency_bench.py                    # 100k rows, 300 iterations
python benchmarks/scenarios/latency_bench.py --engines batcher,duckdb
```

It reports wall p50/p99 **and CPU p50** per shape. Quote the CPU figure when comparing two
builds: on a shared box the wall p99 moves by more than 10x under another session's test
run while CPU p50 barely moves.

Two things it is built to expose:

- **The repeated-vs-parameterized gap.** The same query shape with a different literal each
  time misses the plan cache, because `LogicalPlan.content_key()` includes literal values.
  That gap is the prize a prepared-statement API would collect.
- **The index gap.** SQLite is included as the *transactional reference*, not as a
  competitor Batcher claims to beat — it answers a primary-key lookup from a B-tree while
  Batcher scans. It correspondingly loses the aggregate shape by more than an order of
  magnitude, which is the honest mirror image.

### Structured streaming (Batcher vs Spark Structured Streaming)

Batcher and Spark are the two real *structured-streaming* engines; this head-to-head
runs the drain trigger both support (Spark `Trigger.AvailableNow`, Batcher
`Trigger.available_now()`) over a Parquet backlog, folding a grouped aggregation, with a
per-key correctness gate against DuckDB/Polars (which appear as a batch floor — they have
no streaming engine):

```bash
python benchmarks/scenarios/streaming_throughput.py            # 4M rows, grouped agg
# the Spark comparison needs a JVM: export JAVA_HOME=<jdk17-or-21> first (else it skips)
```

At 4M rows / 1000 keys this shows batcher streaming at **~81M rows/s vs Spark Structured
Streaming's ~6.5M rows/s — 12.5× faster**, correctness-gated (both match DuckDB). Batcher's
micro-batch overhead is small enough that its *streaming* aggregation also edges out
Polars' *batch* aggregation on the same data. The distributed drain (`distributed=True`,
Spark `AvailableNow` parity) fans the same work across the cluster; see
`tests/integration/test_distributed_streaming.py`.

## `cluster/`: GPU multimodal compute (distributed)

Beyond ingest, `benchmarks/cluster/` holds the distributed **GPU** multimodal benchmarks
— batch inference (`gpu_inference`, `gpu_pipeline`), LLM/embeddings
(`gpu_llm`, `gpu_text_embed`), diffusion (`gpu_imagegen`), audio/video
(`gpu_audio`, `gpu_video`), and training-data ingest (`gpu_train_ingest`). Those run on a
live GPU cluster and measure images/sec **and GPU utilization**; they use fixed-seed
in-memory input to isolate the compute + scheduling moat from the S3 read path this `images`
suite measures. Run one directly, e.g. `python benchmarks/cluster/gpu_pipeline.py`.

Tables are materialized to in-memory Arrow once and shared across engines, which
keeps small/medium scale (the dev and CI path) exact and simple. Reading parquet
natively per engine for PB-scale multi-node runs is the documented follow-up — every
engine adapter already has a `read_parquet`.

## Layout

```
benchmarks/
  harness.py     correctness check + best-of-N timing (the measurement core)
  registry.py    the benchmark registry, the suite(...) decorator, and sql_case
  sources/       established public parquet sources; job.py fetches the IMDb database
  datagen/       the two datasets with no public corpus: h2o_tables.py  json_events.py
  context.py     loads a benchmark's tables once, serves every engine
  engines/       one adapter per engine, behind a common contract
    base.py  lineup.py  batcher.py  duckdb.py  polars.py  pyarrow.py
    spark.py  daft.py  ray.py
  suites/
    standard/    SQL-first: tpch.py (22)  clickbench.py (43)  tpcds.py (99)  job.py (113)
                 — the latter two split vendored .sql files written by
                 tools/vendor_{tpcds,job}_queries.py
    h2o/         H2O.ai db-benchmark: groupby.py (10)  join.py (5)
    operators/   dataframe-API operator-mix; where PyArrow also competes natively
    scan/        one table x three parquet file layouts; isolates scan planning
    semistructured/ JSON parsing + typed path extraction
    multimodal/  unstructured ingest: images.py (list/decode/resize) vs Daft
  cluster/       distributed GPU multimodal benchmarks (inference/LLM/audio/video)
  run.py         the CLI: select engines, load data, run, report
  internals/     benchmarks of Batcher's own subsystems, with their own reporting
    distributed.py             single-node == many-partition equivalence + timing
    optimizer_bench.py         Kyber planning latency as the rule set grows
    metadata_bench.py          metadata-answered queries vs the O(rows) computation
    shuffle_vs_object_store.py Arrow Flight shuffle vs the Ray object store
```

### Adding a benchmark

Suite modules are **auto-discovered** (`discover.py`): drop a `.py` file into
`suites/standard/` or `suites/operators/` and its cases register on import — no
`__init__` edit, no list to maintain.

TPC-H / TPC-DS / ClickBench are SQL benchmarks, so each query is written **once** as
SQL and fanned across every SQL-capable engine (batcher via `ds.sql`/`Session`,
duckdb, polars `SQLContext`, spark, daft). Adding one is a single line:

```python
# suites/standard/tpch.py  (or a new file in the same dir)
tpch.sql("tpch-q6", "SELECT sum(l_extendedprice * l_discount) ... FROM lineitem WHERE ...")
```

TPC-DS and JOB are the exceptions to writing the SQL inline, for a reason worth keeping: at
99 and 113 queries, hand-transcribing the statements is how a benchmark quietly stops being
the benchmark. Both are vendored verbatim — TPC-DS from DuckDB's `tpcds` extension (the same
extension whose `dsdgen` produces the tables), JOB from its reference implementation — into
`suites/standard/{tpcds,job}_queries.sql`, and the suite modules only split those files.
Refresh with `python tools/vendor_tpcds_queries.py` / `python tools/vendor_job_queries.py`;
do not edit the `.sql` by hand.

PyArrow has no SQL surface, so it sits out the standard suites (shown `n/a`) and competes
in the operator-mix, where a case is one SQL string for the SQL engines plus a native
callable for PyArrow (Acero):

```python
# suites/operators/aggregation.py
@agg.case("op-groupby-sum")
def groupby_sum(ctx):
    sql = "SELECT l_returnflag, SUM(l_quantity) AS s FROM lineitem GROUP BY l_returnflag"

    def pyarrow(t):  # native Acero
        a = t.group_by("l_returnflag").aggregate([("l_quantity", "sum")])
        return pa.table({"l_returnflag": a["l_returnflag"], "s": a["l_quantity_sum"]})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)
```

## Running

```bash
source .venv/bin/activate
pip install -e '.[bench]'    # duckdb, polars, pyarrow, ray, daft, pyspark
just build-release           # NOT `just build` — see below
```

**Use `just build-release`.** `just build` installs the *dev* profile, which sets no
`opt-level` and leaves `debug_assertions` on, so both Batcher and every third-party crate
it links are unoptimized. Every comparator is an installed release wheel. Timing a dev
build against them compares an unoptimized engine to optimized ones, and the resulting
ratios are not a measurement of anything — in one direction they understate Batcher, and
where a competitor still wins you cannot tell whether the gap is real. The harness cannot
detect the profile from Python, so this is on you.

### Competitor engines (comparators)

The comparison is only meaningful if the comparators actually run. `.[bench]` installs
the Python packages; two need a little more before they execute:

- **PySpark** needs a **JVM** — the wheel alone is not enough (`available()` reports
  `False` and the suite silently omits Spark). Install a JRE once:
  `python -c "import jdk; print(jdk.install('17', jre=True))"` (needs `pip install
  install-jdk`); the adapter finds it under `~/.jre`. To read the `s3://` benchmark data
  directly (the scan suite and TPC-H `--scan`), Spark also pulls the `hadoop-aws`
  connector matching its bundled Hadoop from Maven on first launch — automatic, but it
  needs network and rewrites `s3://` → `s3a://` internally. Set `BENCH_S3_REGION`.
- **Daft** (`pip install daft`) runs its native multithreaded engine with
  `DAFT_RUNNER=native`. It expresses most of the SQL suites; queries its planner handles
  differently (e.g. a couple of TPC-H `HAVING`/tie cases) surface as `FAILED`/`n/a` via
  the correctness gate rather than a wrong "win".

Confirm what will actually run before a long session:

```python
python -c "from benchmarks.engines.lineup import _ADAPTERS; \
  [print(n, e.available()) for n, e in _ADAPTERS.items()]"
```

```bash
python3 benchmarks/run.py                                # TPC-H, scale 1, single-node lineup
python3 benchmarks/run.py --benchmark clickbench         # ClickBench (hits)
python3 benchmarks/run.py --benchmark tpcds --scale 1    # TPC-DS, all 99 queries
python3 benchmarks/run.py --benchmark job                # Join Order Benchmark, 113 queries
python3 benchmarks/run.py --benchmark operators          # operator-mix (incl. PyArrow/Ray)
python3 benchmarks/run.py --benchmark h2o-groupby        # H2O.ai db-benchmark, 10 groupby
python3 benchmarks/run.py --benchmark h2o-join           # H2O.ai db-benchmark, 5 joins
python3 benchmarks/run.py --benchmark all                # every dataset but scan/images/h2o-*

python3 benchmarks/run.py --engines batcher,duckdb,spark # opt in to PySpark
python3 benchmarks/run.py --tier multi                   # batcher, ray, daft
python3 benchmarks/run.py --benchmark tpch --only q1     # one query
python3 benchmarks/run.py --benchmark tpcds --only q17,q72  # a subset, in ONE process
python3 benchmarks/run.py --list                         # list, do not run
```

`--only` matches on substring and takes a comma-separated list, so an arbitrary subset runs in
one process. That matters for an A/B over a handful of queries: a process per query re-loads
the whole table set, which on TPC-DS costs more wall time than the measurement and is what gets
a run `SIGKILL`ed on a box that is short of memory.

`run.py` is the **single entrypoint**: besides the engine-comparison datasets it also
dispatches the standalone benchmarks —
`--benchmark distributed` (single-node == many-partition equivalence),
`--benchmark optimizer` (Kyber planning latency), and
`--benchmark shuffle` (Arrow Flight vs the Ray object store).

`just` shortcuts: `bench`, `bench-tpch`, `bench-clickbench`, `bench-tpcds`, `bench-job`,
`bench-h2o-groupby`, `bench-h2o-join`, `bench-ops`, `bench-scan`, `bench-images`,
`bench-multi`, `bench-all`, `bench-list`, `bench-dist`, `bench-aux <which>`.

The harness (`harness.py`):

1. **Verifies correctness first.** Every engine's output is compared to a reference
   as a sorted row multiset (row and column order normalized away), tolerant of float
   rounding and of DuckDB's `Decimal` sums vs. float. A mismatch marks the row
   `FAILED` and prints a diff; it does not abort the suite.
2. **Times best-of-N** wall-clock in milliseconds after one warm-up. An engine that
   cannot express a query is marked `n/a` (`PARTIAL` overall); one that errors records
   the error rather than crashing the run.
3. **Reports an aligned table** whose columns adapt to the selected lineup:
   `query | <engine>_ms ... | b/<engine> ratios | status`.

## Reading the numbers

`b/<engine>` is `batcher_ms / engine_ms` (lower means Batcher is faster). Timings vary
run to run; treat them as order-of-magnitude. The status column is the gate: only `OK`
rows have been verified to match the reference engine. `PARTIAL` means some engine in the
lineup could not express that query, while the ones that did still agreed.

Read `PARTIAL` carefully rather than as a footnote, because on TPC-DS it is usually
**Batcher's** gap, not a comparator's. The full 99 include shapes the SQL front-end does
not yet cover, and each one prints the reason beneath the table. That is deliberate: a
suite that registered only the queries Batcher already runs would report 100% and measure
nothing. `python benchmarks/internals/tpcds_coverage.py` gives the same gaps grouped by
cause in seconds, without data or a scale factor.

## internals/distributed.py: single-node vs many-partition equivalence

```bash
python3 benchmarks/run.py --benchmark distributed        # TPC-H scale 1, 8 partitions
python3 benchmarks/internals/distributed.py 10 16        # scale 10, 16 partitions
```

Each query runs single-node and again across several partitions via
`collect(distributed=True, num_partitions=...)`. The mergeable algebra
(`partial / combine / finalize` over a hash shuffle) guarantees the two results are
identical, so the benchmark asserts that equivalence first and only then reports
timings. A divergence is a correctness bug and fails the run. Multi-node throughput at
large scale depends on network and cluster size; the engine keeps per-node memory
bounded through the mergeable algebra and spill, and moves batches over Arrow Flight
with credit-based backpressure rather than through the Ray object store.
