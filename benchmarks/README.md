# Batcher benchmark suite

A correctness-first, multi-engine comparison of the Batcher data engine on the
workloads the industry actually cites — **TPC-H** (all 22 queries), **ClickBench**
(43 queries), a **TPC-DS** subset, and an **operator-mix** of single relational
operators — against the engines Batcher claims to beat:

| Tier            | Engines                                              |
|-----------------|------------------------------------------------------|
| **Single-node** | batcher, duckdb, polars, pyarrow, **pyspark** (opt-in) |
| **Multi-node**  | batcher (distributed), ray data, daft, **pyspark** (opt-in) |

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

Loading uses DuckDB's `httpfs` (already a core dependency), which reads local paths,
`s3://`, and `https://` directly. Override the base URI, scale, or ClickBench
partition count without touching code:

```bash
export BENCH_TPCH_BASE=s3://my-mirror/tpch/parquet     # or --source on the CLI
export BENCH_CLICKBENCH_PARTS=10                        # read 10 hits partitions
export BENCH_S3_REGION=us-east-1                        # for S3 sources
```

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
`bench-ops`, `bench-multi`, `bench-all`, `bench-list`, `bench-dist`,
`bench-aux <which>`.

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
