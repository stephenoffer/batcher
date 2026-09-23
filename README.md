# Batcher

**One data engine for SQL, DataFrames, streaming, and models — over any shape of data,
across CPU and GPU, from a laptop to a cluster, on the same code.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/batcher-engine/)
[![PyPI](https://img.shields.io/badge/pypi-batcher--engine-blue.svg)](https://pypi.org/project/batcher-engine/)
[![Docs](https://img.shields.io/badge/docs-batcher-blue.svg)](https://stephenoffer.github.io/batcher/)

[Documentation](https://stephenoffer.github.io/batcher/) ·
[Quickstart](https://stephenoffer.github.io/batcher/getting-started/quickstart.html) ·
[Benchmarks](https://stephenoffer.github.io/batcher/benchmarks/index.html) ·
[Architecture](https://stephenoffer.github.io/batcher/architecture/index.html)

Most data tools make you choose, and then make you choose again. Fast on one machine or
able to scale. Batch or streaming. Tables or images. SQL or Python. CPU or GPU. Every
answer is a different system, and the seams between them are where the time goes.

Batcher is one engine for all of it, and mostly because of one decision: every stateful
operator exists exactly once, as a mergeable `partial → combine → finalize` triple in Rust
over Arrow. One core, ninety-six cores and a cluster differ only in how that triple is
scheduled, so going bigger is a config change and not a rewrite. The same triple is the
incremental form, so a finite table is just a stream that ends. And because decode,
embedding, vector search and inference are expressions in the same algebra, a filter can
run *before* a JPEG decode and a tensor never leaves the engine.

```python
import batcher as bt

revenue = (
    bt.read("s3://events/*.parquet")
    .filter(bt.col("status") == "active")
    .group_by("region")
    .agg(total=bt.col("amount").sum())
    .sort("total", descending=True)
)
print(revenue.to_pydict())  # nothing runs until here
```

## Install

Prebuilt wheels ship for Linux (x86_64 and aarch64, glibc and musl/Alpine), macOS (Apple
silicon and Intel), and Windows x86_64 on Python 3.11+ — no Rust needed.
Batcher is on PyPI as `batcher-engine` and imported as `batcher` (the bare `batcher`
name belongs to an unrelated project):

```bash
pip install batcher-engine
```

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```

Optional features are extras, e.g. `pip install "batcher-engine[ray,cloud]"`.

Other ways to install, each with its own guide in
[Choose how to install](https://stephenoffer.github.io/batcher/getting-started/install/index.html):

| Where it runs | How |
|---|---|
| uv, Poetry, conda/mamba/pixi, Jupyter | [Python environments](https://stephenoffer.github.io/batcher/getting-started/install/python-environments.html) |
| Docker, Kubernetes, Alpine | `docker run ghcr.io/stephenoffer/batcher:latest` — [containers](https://stephenoffer.github.io/batcher/getting-started/install/containers.html) |
| Ray clusters, KubeRay, Slurm, air-gapped networks | `ghcr.io/stephenoffer/batcher:latest-ray` — [servers and clusters](https://stephenoffer.github.io/batcher/getting-started/install/clusters-and-servers.html) |

## What it does

| Area | What ships |
|---|---|
| **Read** | Parquet, CSV, JSON, Arrow, ORC, Avro · text and documents · images, audio, video · SQL databases and warehouses · Kafka, Kinesis, Pulsar, Pub/Sub |
| **Query** | SQL and a DataFrame API over the same plan · joins, windows, pivots, `MERGE INTO` · typed accessors for strings, dates, lists, structs, JSON |
| **Tables** | Delta, Iceberg, and Hudi with transactional writes, time travel, change feeds, schema evolution, compaction |
| **Stream** | Unbounded sources, triggers, watermarks, stateful windows, stream joins, checkpointing, exactly-once sinks |
| **Model** | GPU batch inference, LLM scoring, embeddings and vector search, RAG, tabular models, preprocessors, zero-copy PyTorch loaders |
| **Operate** | Out-of-core spill, caching, explain plans, a live progress UI, metrics, data-quality contracts, column masking and row-level security |

## What makes it different

Most engines decide how to run your query before they have looked at any data, then
stick with that plan even when the data turns out different — which is the usual
reason a job stalls or runs out of memory. Batcher measures each stage as it completes
and re-plans the rest of the query on the real numbers, so a query that starts on a
bad guess corrects itself instead of failing.

Being precise about that, because it is easy to overclaim: this is stage-boundary
re-optimization, the same mechanism and granularity as Spark AQE, and it engages only
on a joined query large enough to pay for the re-planning: 5M rows, or roughly 320 MB,
for each pipeline breaker the loop would cut at. That is about 10M rows for the simplest
joined shape and proportionally more for a many-join query, because each cut is what
costs. Two things about
it *are* different. It runs **single-node**, where DuckDB has no equivalent at all, and
what it measured is **kept**: cardinality sketches, cost coefficients calibrated from
measured operator times, and a bandit over join strategies all feed the *next* run, so
a query gets a better plan the more often it runs. Neither DuckDB nor Spark does that
second half. [What makes Batcher different](https://stephenoffer.github.io/batcher/architecture/differentiators.html)
covers both halves and where each one stops.

## How it compares

| Tool | Where it stops | What Batcher does instead | Measured (geomean) |
|------|----------------|---------------------------|----------|
| **DuckDB**, its own compressed store | fast, but single-node and plans once | scales out, and re-optimizes mid-query | *2026-09-13, 48 cores:* **1.4×** TPC-H sf1, **1.5×** ClickBench, **2.9×** JSON, **1.6×** H2O `join`, **1.3×** the operator mix; **loses** H2O `groupby` (0.95×) and **loses TPC-H sf100** (see below) |
| **DuckDB**, on the same Arrow | — | the like-for-like execution comparison | *2026-09-13:* **4.0×** TPC-H sf1, **6.3×** ClickBench, **3.1×** JSON, **1.7×** H2O `join`, **1.2×** H2O `groupby`, **2.1×** the operator mix |
| **Polars** | fast, but single-node | the same code runs from one core to a cluster | **2.4×** TPC-H sf1 (21/22), **3.0×** ClickBench, **8.6×** the operator mix (19/19) |
| **Daft** | scales, but plans once | adaptive re-optimization, and a correct q6 | **2.9×** TPC-H sf1 (20/20), **3.8×** ClickBench (41/41), **2.4×** cluster-vs-cluster |
| **Spark** | scales, but heavy on small jobs | runs in-process locally — no cluster to spin up | **5×–33×** on TPC-H |

The speed is measured **correctness-first**: the harness refuses to time a query whose result
doesn't match the oracle, so a fast wrong answer never counts as a win. That gate has caught
real bugs in other engines — on TPC-H q6, Daft returns `75,207,768.19` where the official
answer is `123,141,078.2283`, because it folds the bound `0.06 + 0.01` in IEEE double to
`0.06999999999999999` and drops every `l_discount = 0.07` row. Batcher returns the official
answer exactly.

**TPC-H at scale factor 10 is a win; at scale factor 100 it is a loss.** At sf10 (60M-row
`lineitem`, 2026-08-26) Batcher runs in 0.91× DuckDB's time against its own compressed store and
0.27× against DuckDB on the same Arrow. At sf100 (600M rows, scanned from Parquet) the last
recorded single-node run (2026-08-12) did not finish: `tpch-q9` was OOM-killed under a 64 GB cap
and the suite hit a 3-hour wall at 21 of 22. The crossover is somewhere between 60M and 600M
rows, and closing it is open work.

Four places Batcher does **not** yet win, stated up front, all of them against DuckDB reading
its own compressed store: **TPC-H sf100 single-node**, the Join Order Benchmark on geomean
(1.11×, 44 of 109), the H2O.ai `groupby` task (0.95×, 4 of 10), and Parquet decode
(1.4×–2.8×, which is `arrow-rs` and is slower than PyArrow too). Several wall-clock wins are
also bought with more CPU: the TPC-H sf1 win costs about 1.9× DuckDB's CPU time. High-concurrency serving is not this engine's shape at all. All are
detailed below.

Two of those are the storage format rather than the engine, and the like-for-like row
above is how you tell: `groupby` goes from a loss to a 1.2× Batcher win on the same Arrow. The Join Order Benchmark is the one that is genuinely about planning — it is built to
expose exactly the estimation errors this design claims to correct — and it is the suite that
has moved furthest, which is the direction the design predicts.

## Benchmarks

Numbers, not adjectives — and every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the *identical* result (a sorted row multiset within float
tolerance), and only then trusts the timing. A fast wrong answer is a bug, not a win. Setup:
TPC-H `lineitem` (6M rows at scale 1, 60M at scale 10), read once into Arrow and shared
byte-identically across engines; an 8-node / 128-CPU cluster; 8×T4 GPUs for the ML runs. Full
per-scale tables: [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md).

**Single-node, head-to-head on the same in-memory Arrow (TPC-H sf1).** Each engine runs the
*identical* zero-copy Arrow input Batcher runs on, so this is a like-for-like **execution**
comparison, and the first column is the same cases against DuckDB reading its own compressed
store. Cells are **how many times faster Batcher is**, so a value below 1 marks a loss. Every
row is correctness-gated against DuckDB:

| operator | vs DuckDB (own store) | vs DuckDB (same Arrow) | vs Polars | vs PyArrow | vs Daft |
|---|:--:|:--:|:--:|:--:|:--:|
| global sum&sup1; | **12×** | **62×** | **13×** | **56×** | **46×** |
| distinct &rarr; `LIMIT` | **5.3×** | **6.2×** | **7.3×** | **47×** | **5.9×** |
| window running `sum()` | **5.0×** | **4.8×** | **18×** | n/a&sup2; | &mdash;&sup3; |
| filter &rarr; count&sup1; | **3.5×** | **61×** | **28×** | **706×** | **17×** |
| window `lag()` | **2.8×** | **2.9×** | **44×** | n/a&sup2; | &mdash;&sup3; |
| window `rank()` | **2.2×** | **2.6×** | **17×** | n/a&sup2; | &mdash;&sup3; |
| distinct, low cardinality | **1.9×** | **2.7×** | **12×** | **1.2×** | **7.8×** |
| window whole-partition `sum()` | **1.5×** | **1.7×** | **1.5×** | n/a&sup2; | **66×** |
| dedup by key, unordered | **1.2×** | **1.6×** | **10×** | **40×** | **2.2×** |
| join &rarr; group-by | **1.2×** | **2.8×** | **1.5×** | **7.1×** | **5.3×** |
| dedup by key, ordered | **1.1×** | **1.8×** | **32×** | **40×** | &mdash;&sup3; |
| sort &rarr; top-N (`LIMIT`) | **1.1×** | **1.3×** | **34×** | **396×** | **7.0×** |
| sort by string | 0.91× (slower) | 0.89× (slower) | **3.5×** | **33×** | **31×** |
| group-by sum (2 keys) | 0.82× (slower) | **2.6×** | **3.0×** | **1.1×** | **6.5×** |
| filter &rarr; project | 0.83× (slower) | **2.6×** | **1.7×** | **23×** | **1.4×** |
| distinct, high cardinality | 0.81× (slower) | 0.94× (slower) | **5.7×** | **6.9×** | **1.8×** |
| sort by string, low cardinality | 0.65× (slower) | 0.89× (slower) | **3.5×** | **25×** | **10×** |
| group-by sum (1 key) | 0.62× (slower) | **2.2×** | **2.9×** | 0.85× (slower) | **6.9×** |
| sort by string &rarr; top-N | 0.59× (slower) | 0.96× (slower) | **39×** | **599×** | **8.4×** |

**Batcher wins 12 of 19 against DuckDB's own store, 15 of 19 against DuckDB on the same
Arrow, all 19 against Polars, 14 of 15 against PyArrow, and all 15 against Daft.** Against
Spark it wins all ten of the operators measured against it, by 13×–197×.

Sorting is where the losses concentrate: four of the seven are string sorts or a
high-cardinality distinct, and all four are losses on the same Arrow too, which puts them in
the kernel rather than in the storage format. That is open work, named in
`docs/architecture/internals/competitive_architecture.md`.

&sup1; These two cases are answered partly from memoized statistics rather than executed in
full, so they measure the plan, not only the kernel. They are marked in
[`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md) and excluded from the
suite geomean quoted above. &sup2; PyArrow (Acero) has no window functions. &sup3; Daft is
SIGKILLed on an ordered 6M-row window (it needs ~22 GB, and `lag()` exceeds 30 GB).
All columns re-measured 2026-08-15 on a release build, 96 cores.

**Full suites, not just an operator mix.** The first column has every engine reading the
identical zero-copy Arrow input, so it compares *execution* alone; the second lets DuckDB read
its own compressed format, so it puts DuckDB's storage engine *and* execution engine against
Batcher's execution engine. Cells are how many times faster Batcher is, so a value below 1 is a
loss:

| suite | vs DuckDB on the same Arrow | vs DuckDB's own compressed store |
|---|---|---|
| **TPC-H sf1** — 22 queries | **4.0×** | **1.4×** |
| **TPC-H sf10** — 22 queries&sup1; | **3.7×** (22 of 22) | **1.1×** (22 of 22 correct) |
| **TPC-H sf100** — 22 queries&sup2; | not run | **loss**: q9 OOM, suite unfinished at 3 h |
| **ClickBench** — 43 queries | **6.3×** | **1.5×** |
| **Operator mix** — 46 kernels | **2.1×** | **1.3×** |
| **Semi-structured JSON** — 5 queries | **3.1×** | **2.9×** |
| **H2O.ai `join`** — 5 queries | **1.7×** | **1.6×** |
| **H2O.ai `groupby`** — 10 queries | **1.2×** | 0.95× (loss) |
| **JOB** — 113 queries, real IMDb&sup3; | **2.6×** | 0.90× (loss) |

Rows without a mark are the 2026-09-13 sweep on a 48-core box, best of five, one process per
case (the same figures as the documentation landing page). &sup1; 2026-08-26, after the
`duckdb_arrow` chunking fix, 92 cores. &sup2; 2026-08-12, scanned from a local Parquet mirror,
64 GB cap. &sup3; 2026-08-16, carried; JOB reads Parquet, so the chunking defect did not
affect it.

An earlier version of this table showed 11×–27× same-Arrow wins on H2O and JSON. Those were
measured while DuckDB's Arrow scan was pinned to one thread by a single-chunk input
(`benchmarks/BENCHMARK_RESULTS.md`, 2026-08-26) and are withdrawn.

Standout queries, against DuckDB's own store and against DuckDB on the same Arrow: TPC-H q15
**4.9×** / **14×**, q19 **2.7×** / **4.3×**; ClickBench q27 **1.9×** / **88×**, q40 **1.3×** /
**26×**. Batcher also beats Spark on every TPC-H query (5×–33×).

**Correctness is a result too.** On TPC-H q6, Batcher returns the official answer
(`123,141,078.2283`); **Daft returns `75,207,768.19`**. Daft also returns no rows at all for q15, and
cannot run q21 or q22; Polars cannot parse 8 of the 99 TPC-DS queries or 3 of the 43 ClickBench
queries. Every such case is reported as `PARTIAL` and excluded from that engine's ratio rather
than counted as a Batcher win.

**Cluster vs cluster.** On an 8-node / 128-CPU cluster with *both* engines distributed and
reading the same S3 parquet, TPC-H sf10 q6: Batcher **224 ms** vs Daft **536 ms** — **2.4×
faster, and correct where Daft is not**.

**How it scales, kept honest.** Against Polars, PyArrow, and Spark, Batcher leads at every
scale we measured. Against DuckDB the story is scale-dependent, and we report it straight:

| scale (single node) | same Arrow | DuckDB's own store |
|---------------------|---|---|
| **sf1** — 6M rows, in memory   | **4.0×** (2026-09-13) | **1.4×** (2026-09-13) |
| **sf10** — 60M rows, in memory | **3.7×** (2026-08-26) | **1.1×** (2026-08-26) |
| **sf100** — 600M rows, scanned | not run | **DuckDB wins**: q9 OOM-killed at a 64 GB cap, suite unfinished at 3 h (2026-08-12) |

Nine of the 13 TPC-H shapes we timed at both scales are **sublinear** from sf1 to sf10 — ten
times the rows for less than ten times the time. The four that are not (q5, q9, q13, q18) are
join-tree shapes whose intermediate results grow faster than the scan, and they are most of the
sf10 gap.

Batcher leads DuckDB's own store at sf1 and sf10 and DuckDB leads at sf100, and the crossover
has three structural causes, none of them a tuning knob: DuckDB
**decompresses its native store on the fly** (fewer bytes off memory — Batcher's Arrow-only
contract has no compressed form to read), its **vector-at-a-time engine with selection vectors**
edges Batcher's batch-at-a-time kernels as rows grow, and it **streams** where Batcher's model
materializes each operator's output — which is what OOMs the largest single-node sf100 joins.
Batcher's answer at that scale is **distribution**: the same mergeable operators shard across a
cluster (one partition per node, bounded per-node memory), which is the regime it is built for.
Closing the single-node scale gap to DuckDB is honest, open work — a compressed or
dictionary-encoded scan path, dictionary-aware grouping, and streaming between operators. The
first of those is also what the H2O.ai `groupby` loss is: on the identical Arrow input Batcher
wins that task 10 of 10 by **9×**, and loses it 4 of 10 once DuckDB reads its own dictionary
encoding instead.

**Distributed data plane.** Execution is in-process and native, so a distributed operator costs
no per-operation task-scheduling hop and no pandas bridge. Bulk Arrow batches move worker to
worker over Arrow Flight with credit-based flow control, bypassing the Ray object store, and
streaming `map_batches`, row-exploding `flat_map`, chained multi-stage maps, Parquet reads, and
`iter_torch_batches` training ingest all run on that same path.

**GPU batch inference (8×T4)** — one of the workload families the same engine runs. Stage-overlap
streaming and session-warm pools, which load a model once per session rather than once per job,
are what produce these figures:

- **LLM batch inference** (gpt2 generate) — 814 prompt/s
- Batch inference (ResNet-50) — 2576 img/s at 78% GPU utilization
- Batch embeddings (2048-d vectors) — 2502 img/s at 80% GPU utilization
- Zero-config `map_batches(Model, num_gpus=1)` — 2451 img/s at 82% GPU utilization

Stage-overlap alone lifted a decode → ResNet-50 pipeline from **942 → 2504 img/s** and GPU
utilization from **~30% → 81%** — same result, the device just stops idling through the CPU decode.

**Why the wins happen** — they're structural, not tuning: an in-process native engine over Arrow
(no per-operation scheduler / object-store hop), a high-cardinality aggregation path that hashes
native and composite keys directly and merges its radix partitions in parallel without copying the
key column twice — so `GROUP BY` and `COUNT(DISTINCT)` over string or near-unique keys scale, and
`DISTINCT` shards across every core like the aggregate above it — warm model pools +
stage-overlapped streaming for GPU work, and adaptive re-optimization that re-tunes the plan on
measured cardinalities mid-query.

## Documentation

Full docs: **<https://stephenoffer.github.io/batcher/>**

| If you want to | Go to |
|---|---|
| Install it and run a first query | [Getting started](https://stephenoffer.github.io/batcher/getting-started/index.html) |
| Understand the one idea the API rests on | [Core concepts](https://stephenoffer.github.io/batcher/getting-started/concepts/index.html) |
| Port from Spark, pandas, Polars, or DuckDB | [Migration guides](https://stephenoffer.github.io/batcher/migration/index.html) |
| Look up a capability | [User guide](https://stephenoffer.github.io/batcher/user-guide/index.html) |
| Run models, embeddings, or a training loop | [Machine learning](https://stephenoffer.github.io/batcher/ml/index.html) |
| Copy working code | [Tutorials](https://stephenoffer.github.io/batcher/tutorials/index.html) and [examples](https://stephenoffer.github.io/batcher/examples/index.html) |
| Look up a name | [API reference](https://stephenoffer.github.io/batcher/api/index.html) |
| Connect Kafka, Snowflake, Delta, Ray, PyTorch | [Integrations](https://stephenoffer.github.io/batcher/integrations/index.html) |
| See the numbers and how they were measured | [Benchmarks](https://stephenoffer.github.io/batcher/benchmarks/index.html) |
| Know how the engine works | [Architecture](https://stephenoffer.github.io/batcher/architecture/index.html) and [deep dives](https://stephenoffer.github.io/batcher/deep-dives/index.html) |

## Under the hood

You write Python; the heavy lifting happens in Rust over [Apache
Arrow](https://arrow.apache.org/). Python builds and optimizes your query, and Rust
runs it — so you get Python's ergonomics with native speed. The same engine powers
one core and a whole cluster, so a result is the same whether it ran on your laptop or a
hundred machines, apart from four cases where the query itself leaves the answer open:
floating-point summation order, tie-breaking in `row_number()`, which rows a `LIMIT` keeps
over unordered input, and the element order of an unordered `array_agg`. The full design (and the math behind the
optimizer) is in the
[documentation](https://stephenoffer.github.io/batcher/) and `architecture.txt`.

> **Status:** young but working, not yet 1.0. Batcher runs SQL and DataFrame
> workloads, single-node and distributed, and is benchmarked for correctness and
> speed against DuckDB and Polars. Expect APIs to change, and some operators and
> large-scale paths are still landing.

## Build from source

Requires a [Rust toolchain](https://rustup.rs):

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
maturin develop           # build the Rust engine into the venv
pytest
```

Or install an unreleased revision directly:
`pip install "git+https://github.com/stephenoffer/batcher.git"`.

On an ephemeral dev box — an Anyscale workspace, a preemptible VM, a rebuilt container —
`rustup`'s default install lives in `$HOME` and does not survive the machine being replaced,
so `cargo` and `just` disappear and the repo stops building for a reason nothing names.
`tools/bootstrap_env.sh` reinstalls the toolchain onto fast local disk from a tarball cached
on durable storage and exports `CARGO_HOME`/`RUSTUP_HOME`/`CARGO_TARGET_DIR`. It is a no-op
once the toolchain is present, so it is safe to source from a shell startup file:

```bash
source tools/bootstrap_env.sh
```

## Layout

- `python/batcher/` — the Python API
- `crates/` — the Rust engine
- `docs/`, `architecture.txt` — design and documentation
- `MAP.md` — the file-level index: what every module and crate file is for, and
  where new code goes. Generated by `just map` from the code's own docstrings, so
  it stays true. Start here when finding your way around.

Apache-2.0 licensed.
