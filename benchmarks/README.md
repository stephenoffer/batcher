# Batcher benchmark suite

A correctness-first, multi-engine comparison of the Batcher data engine on the
workloads the industry actually cites — **TPC-H** (all 22 queries), **ClickBench**
(43 queries), a **TPC-DS** subset, an **operator-mix** of single relational
operators, a **scan** benchmark over three parquet file layouts, and an **images**
benchmark for unstructured multimodal ingest — against the engines Batcher claims to beat:

| Tier            | Engines                                              |
|-----------------|------------------------------------------------------|
| **Single-node** | batcher, duckdb, `duckdb_arrow`, polars, pyarrow, **pyspark** (opt-in) |
| **Multi-node**  | batcher (distributed), ray data, daft, **pyspark** (opt-in) |

`duckdb` runs on its compressed *native* store (DuckDB at its best); `duckdb_arrow` runs the
same query on the *same zero-copy Arrow* Batcher consumes — the like-for-like execution bar.
Reporting both separates DuckDB's storage engine from its execution engine (see
`TPCH_FINDINGS.md`).

Correctness is checked before any timing is trusted: a query is only timed once its
result matches the reference engine, so a fast wrong answer can never be reported as
a win.

## No generated data — established public sources only

The suite **never generates data**. Every table is read from a canonical public
parquet location and normalized once (`sources.py`) so all engines see identical
inputs:

| Dataset    | Default source                                                                 | Access |
|------------|--------------------------------------------------------------------------------|--------|
| TPC-H      | `s3://ray-benchmark-data/tpch/parquet/sf{scale}/{table}/`                       | S3 (creds/region may be needed) |
| ClickBench | `https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_*.parquet` | anonymous HTTPS |
| TPC-DS     | `s3://ray-benchmark-data/tpcds/parquet/sf{scale}/{table}/`                      | S3 (configurable) |
| Scan       | `s3://ray-benchmark-data/{parquet,parquet/128MiB-file,small-parquet}/{size}/`   | S3 (creds/region may be needed) |

Loading uses DuckDB's `httpfs` (already a core dependency), which reads local paths,
`s3://`, and `https://` directly. Override the base URI, scale, or ClickBench
partition count without touching code:

```bash
export BENCH_TPCH_BASE=s3://my-mirror/tpch/parquet     # or --source on the CLI
export BENCH_CLICKBENCH_PARTS=10                        # read 10 hits partitions
export BENCH_SCAN_BASE=s3://my-mirror                   # scan corpus bucket root
export BENCH_S3_REGION=us-east-1                        # for S3 sources
```

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
`sources.py` filters to keep the layout genuinely many-small), so there the cross-family
comparison is indicative and the per-engine one within a family still exact.

Nine shapes run against each layout, chosen to separate the costs a layout moves:
`count`/`minmax` (listing + footer metadata only), `sum1` (projection pushdown, 1 of 16
columns), `sumwide` (I/O-bound, all 16), `filter`/`filter_agg` (~1% selectivity, row-group
skipping), and `groupby`/`distinct`/`topn` (cost past the scan, in the operators).

Each engine's reader is rebuilt **inside** the timed call, so listing and metadata are
measured rather than amortized away — the opposite of the `--scan` mode used for TPC-H,
where the scan is a fixed setup cost shared across 22 queries.

Every SQL engine (Batcher, DuckDB, Polars, Daft, Spark) runs all nine shapes. PyArrow
runs all nine through Acero; Ray Data covers `count`, `minmax`, `filter`, and
`filter_agg` — the shapes its API expresses directly, and the ones that isolate the
small-files tax — and reports `n/a` for the rest rather than a hand-rolled
reimplementation that would benchmark the benchmark instead of the engine.

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

## `cluster/`: GPU multimodal compute (distributed)

Beyond ingest, `benchmarks/cluster/` holds the distributed **GPU** multimodal benchmarks vs
Ray Data (and Daft) — batch inference (`gpu_inference`, `gpu_pipeline`), LLM/embeddings
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
  sources.py     established public parquet sources (no data generation)
  context.py     loads a benchmark's tables once, serves every engine
  engines/       one adapter per engine, behind a common contract
    base.py  lineup.py  batcher.py  duckdb.py  polars.py  pyarrow.py
    spark.py  daft.py  ray.py
  suites/
    standard/    SQL-first: tpch.py (22)  clickbench.py (43)  tpcds.py (subset)
    operators/   dataframe-API operator-mix; where PyArrow + Ray Data also compete
    scan/        one table x three parquet file layouts; isolates scan planning
    multimodal/  unstructured ingest: images.py (list/decode/resize) vs Ray Data + Daft
  cluster/       distributed GPU multimodal benchmarks (inference/LLM/audio/video) vs Ray
  run.py         the CLI: select engines, load data, run, report
  distributed.py             single-node == many-partition equivalence + timing
  optimizer_bench.py         Kyber planning latency as the rule set grows
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

PyArrow and Ray Data have no SQL surface, so they sit out the standard suites
(shown `n/a`) and compete in the operator-mix, where a case is one SQL string for the
SQL engines plus native callables for PyArrow (Acero) and Ray Data:

```python
# suites/operators/aggregation.py
@agg.case("op-groupby-sum")
def groupby_sum(ctx):
    sql = "SELECT l_returnflag, SUM(l_quantity) AS s FROM lineitem GROUP BY l_returnflag"
    def pyarrow(t):  # native Acero
        a = t.group_by("l_returnflag").aggregate([("l_quantity", "sum")])
        return pa.table({"l_returnflag": a["l_returnflag"], "s": a["l_quantity_sum"]})
    def ray(rd):     # native Ray Data
        ...
    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)
```

## Running

```bash
source .venv/bin/activate
pip install -e '.[bench]'    # duckdb, polars, pyarrow, ray, daft, pyspark
just build                   # or: maturin develop --release
```

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
python3 benchmarks/run.py --benchmark tpcds --scale 1    # TPC-DS subset
python3 benchmarks/run.py --benchmark operators          # operator-mix (incl. PyArrow/Ray)
python3 benchmarks/run.py --benchmark all                # every dataset

python3 benchmarks/run.py --engines batcher,duckdb,spark # opt in to PySpark
python3 benchmarks/run.py --tier multi                   # batcher, ray, daft
python3 benchmarks/run.py --benchmark tpch --only q1     # one query
python3 benchmarks/run.py --list                         # list, do not run
```

`run.py` is the **single entrypoint**: besides the engine-comparison datasets it also
dispatches the standalone benchmarks —
`--benchmark distributed` (single-node == many-partition equivalence),
`--benchmark optimizer` (Kyber planning latency), and
`--benchmark shuffle` (Arrow Flight vs the Ray object store).

`just` shortcuts: `bench`, `bench-tpch`, `bench-clickbench`, `bench-tpcds`,
`bench-ops`, `bench-scan`, `bench-images`, `bench-multi`, `bench-all`, `bench-list`,
`bench-dist`, `bench-aux <which>`.

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
rows have been verified to match the reference engine. `PARTIAL` means an engine in the
lineup legitimately could not express that query (e.g. Polars' SQL subset, PyArrow on
the SQL suites) — the verified engines still agreed.

## distributed.py: single-node vs many-partition equivalence

```bash
python3 benchmarks/distributed.py            # TPC-H scale 1, 8 partitions
python3 benchmarks/distributed.py 10 16      # scale 10, 16 partitions
```

Each query runs single-node and again across several partitions via
`collect(distributed=True, num_partitions=...)`. The mergeable algebra
(`partial / combine / finalize` over a hash shuffle) guarantees the two results are
identical, so the benchmark asserts that equivalence first and only then reports
timings. A divergence is a correctness bug and fails the run. Multi-node throughput at
large scale depends on network and cluster size; the engine keeps per-node memory
bounded through the mergeable algebra and spill, and moves batches over Arrow Flight
with credit-based backpressure rather than through the Ray object store.
