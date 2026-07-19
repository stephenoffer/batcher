# Benchmarks

Numbers, not adjectives. Every figure on this page comes from a run that was
**correctness-gated first**: the harness executes the query on every engine, checks they
return the identical result (a sorted row multiset within float tolerance), and only then
records a time. A fast wrong answer is a bug, not a win. A benchmark that disagrees with the oracle reports
`FAILED` and produces no number at all.

:::{important}
That gate is not decoration. It has caught real bugs in other engines: on TPC-H q6, **Daft and
Polars each return 75,207,768.19 where the official TPC-H answer is 123,141,078.23** — they
fold the predicate bound `0.06 + 0.01` in IEEE double, getting `0.06999999999999999`, and so
drop every `l_discount = 0.07` row. Batcher returns the official answer exactly. The harness
refuses to time a wrong result, so neither engine is quietly credited with a win. Read every
table on this site knowing that a missing number means a wrong answer, not a slow one.
:::

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` AI & GPU workloads
:link: ai-and-gpu
:link-type: doc
Ten workload families, every one at least 2× Ray Data. LLM inference 11×, text
embeddings 47×.
:::

:::{grid-item-card} {octicon}`database;1.1em` Analytics & I/O
:link: analytics
:link-type: doc
Operators, TPC-H, connectors, and the honest gap to DuckDB on join-heavy SQL.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Methodology
:link: methodology
:link-type: doc
Hardware, correctness gating, and the commands to reproduce every number.
:::
::::

## The short version

Batcher's advantage is widest exactly where the modern workload lives: AI, multimodal, and
last-mile training ingest. On classical analytics it now sweeps every suite we run against
DuckDB reading the same Arrow — **22 of 22** TPC-H, **42 of 43** ClickBench, **5 of 5** JSON.
The one place DuckDB still leads is its own compressed store on join-heavy SQL, where it never
pays an ingest and decompresses as it scans. We publish both, and label which is which.

| Workload | Compared with | Result |
|---|---|---|
| Text embeddings (MiniLM) | Ray Data | **47× faster** |
| Audio feature extraction | Ray Data | **12.5× faster** |
| LLM batch inference (gpt2) | Ray Data | **11.1× faster** |
| Image generation (diffusion) | Ray Data | **8.6× faster** |
| Image decode → tensor | Daft / Ray Data | **2.4× / 6.1× faster** |
| Training ingest (`iter_torch_batches`) | Ray Data | **3.0× faster** |
| Batch inference (ResNet-50) | Ray Data | **2.05× faster** |
| Parquet read → aggregate | Ray Data | **20.8× faster** |
| TPC-H, all 22 queries | DuckDB on the same Arrow | **won 22 of 22** (1.03–7.1×) |
| TPC-H sf10 q6, cluster vs cluster | Daft (both distributed) | **2.4× faster**, and Daft's answer is wrong |
| ClickBench, 43 queries | DuckDB on the same Arrow | **won 42 of 43**, 43/43 correct |
| Semi-structured JSON, 5 queries | DuckDB / Polars | **3.6–12× / 11–100× faster** |
| Group-by, filter, top-N | DuckDB | **1.3–5× faster** |
| Sort → top-N, window functions | Polars | **5–50× faster** |
| Join-heavy TPC-H | DuckDB on its **native store** | **~1.4× slower** (see below) |

:::{note}
Those rows were not measured on the same machine. The DuckDB and Polars comparisons ran on
a 16-core node; the Ray Data ones on a 96-core node or an 8×T4 cluster, because a GPU
benchmark needs GPUs. A ratio *within* a row is meaningful. A number lifted out of one row
and set against a number from another is not. [Methodology](methodology.md) lists the
hardware per family.
:::

## Where Batcher wins

**AI and GPU work, by a wide margin.** Across ten GPU workload families on 8×T4, using real
models and a correctness gate, every one beats Ray Data by at least 2×, and most by far
more. The wins come from engine mechanisms rather than per-workload tuning. Stage-overlapped
streaming keeps the device fed (the CPU decode of morsel *k+1* runs while the GPU forward
of morsel *k* is still in flight, which lifted a two-stage ResNet-50 pipeline from 942 to
2,504 img/s and GPU utilization from ~30% to 81%), and session-warm pools load a model
once per session instead of once per job.

**The fixed cost that isn't paid.** Batcher runs in-process and native over Arrow, so it
pays none of Ray Data's per-operation task-scheduling and block/pandas-bridge overhead,
roughly 300–4,500 ms even on a warm cluster. On small and medium queries that overhead *is*
the runtime, which is why the operator gaps reach 50–450×.

**Ray Data's own home turf.** The fair test is not SQL, where Ray Data is weakest, but its
bread-and-butter streaming `map_batches`. Batcher still leads there: a `map_batches`
transform by 2.35×, a row-exploding `flat_map` by 3.5×, a chained multi-stage map by 3.17×,
and `iter_torch_batches` training ingest by 3.0×.

**Reading data.** Parquet read → sum is 20.8× Ray Data, CSV 14.3×, JSON 5.3×. Batcher
decodes files concurrently in-process, with no per-file task scheduling and no object-store
hop.

## Where Batcher loses

Publishing only the wins would make this page marketing rather than measurement.

:::{warning}
On a 16-core single node at scale factor 1, **DuckDB is faster on 16 of the 21 comparable
TPC-H queries**, with a geometric mean of about **1.4× in DuckDB's favor**. Daft leads by
**2–12× across the multi-join queries** and by roughly **2× on a per-batch Python UDF**. If
your workload is join-heavy single-node SQL, those are the numbers that apply to you, and no
admonition elsewhere on this site changes them.
:::

**Join-heavy TPC-H against DuckDB.** Batcher wins the scan-and-aggregate shapes (q1, q6,
q12, q14) and trails on the multi-join ones (q5 ≈ 3×, q8 ≈ 2.3×, q17 ≈ 2.5×). The same gap
shows against Daft on join-heavy queries.

**Per-batch Python UDFs against Daft.** Roughly 2× behind on a tight numpy `map_batches`.

The cause is understood and is not a tuning knob: single-node parallelism currently reaches
about 1.7–3.8× on 16 cores where Daft uses effectively all of them, and Batcher does more
CPU work per query. Closing it is a runtime-parallelism and kernel-efficiency effort. It is
tracked, not hidden.

Correctness, though, is not in question: Batcher matches DuckDB on all 22 TPC-H queries.

## Scaling out

The same mergeable operators run distributed, so scale-out is a scheduling decision rather
than a rewrite, and the distributed result is bit-identical to the single-node one. At small
scale, distribution correctly loses to a single node (the network shuffle costs more than it
saves). Batcher stays within ~7% rather than falling off a cliff:

| Engine | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed (4 workers) | 92 ms |
| Ray Data (cluster) | 4,284 ms |

Even distributed-against-distributed, that is **46× Ray Data**.

## Reproduce it

Every number here is regenerated by the harness in `benchmarks/`. The full run log lives in
`benchmarks/BENCHMARK_RESULTS.md`, failures and regressions included. See
[methodology](methodology.md) for the hardware each family was measured on and the exact
commands.

```bash
python benchmarks/run.py --benchmark tpch --tier single    # vs DuckDB / Polars
python benchmarks/run.py --benchmark operators --tier multi # vs Ray Data / Daft
python benchmarks/scenarios/image_decode.py                 # multimodal ingest
```

## Head to head

One page per engine, each with the losses stated as plainly as the wins.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` DuckDB
:link: vs-duckdb
:link-type: doc
We take the operators; DuckDB takes join-heavy SQL.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Polars
:link: vs-polars
:link-type: doc
50× on top-N; Polars takes high-cardinality hashing by 2×.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Ray Data
:link: vs-ray-data
:link-type: doc
The widest margin, on Ray Data's own data plane.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Daft
:link: vs-daft
:link-type: doc
2.4× on image decode; Daft takes multi-joins by 2–12×.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Spark
:link: vs-spark
:link-type: doc
Architecture only. No head-to-head numbers are published yet.
:::
::::

And one page per workload family.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` TPC-H
:link: tpch
:link-type: doc
All 22 queries, including the ones we lose.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal ingest
:link: multimodal-ingest
:link-type: doc
Images, point clouds, audio, video.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Scaling out
:link: scaling
:link-type: doc
The distributed runs, and a measured negative result.
:::
::::

## See also

- [Performance tuning](../user-guide/performance.md): making *your* query faster, with the
  levers these numbers come from.
- [Morsel parallelism](../deep-dives/morsel-parallelism.md) and
  [JIT compilation](../deep-dives/jit-compilation.md): the two mechanisms behind most of the
  operator wins.
- [Mergeable algebra](../deep-dives/mergeable-algebra.md): why the distributed result is
  bit-identical to the single-node one.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): the thing no number
  on this page can show, because DuckDB and Spark have nothing to compare it against.
- [Optimizing a slow query](../tutorials/optimizing-a-slow-query.md): the diagnosis loop.

```{toctree}
:hidden:

ai-and-gpu
analytics
vs-duckdb
vs-polars
vs-ray-data
vs-daft
vs-spark
tpch
multimodal-ingest
scaling
methodology
```
