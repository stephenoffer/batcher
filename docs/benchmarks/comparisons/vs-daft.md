# vs Daft

This page compares Batcher against Daft on single-node and distributed work.

Daft is a mature, fast, multi-core Rust engine, roughly DuckDB-class, with about 4 ms of fixed overhead. It is the closest competitor Batcher has on single-node compute. Batcher takes multimodal ingest and top-N by large margins, takes the distributed join by 1.7x to 2.2x, ties on aggregation and single-stage expression ETL, and on 96 cores takes 18 of the 19 TPC-H queries both engines answer.

:::{important}
Daft computes TPC-H **q6 wrong**: it folds `0.06 + 0.01` in IEEE double to `0.06999999999999999`, dropping every `l_discount = 0.07` row, and returns 75.2M where
the correct revenue is 123.1M. The harness declines to time a query whose result does not
match the oracle, so Daft gets no number on q6 rather than a fast one. Every timing on this
page passed that gate first.
:::

## Scorecard

Each row is one workload shape, with the engine that won it and by how much. Read the
ratios alongside the methodology above, not on their own:

| Shape | Winner |
|---|---|
| Image decode → tensor | Batcher, 1.9×–2.4× (machine-dependent) |
| Image curation / augmentation | Batcher, 5.7× — Daft has no native equivalent |
| Audio preprocessing | Batcher — Daft has no native audio surface |
| Top-N / sort-limit | Batcher, 8x to 10x |
| In-memory filter / sum kernels | Batcher, 6.7× / 18× |
| Global aggregate, group-by, expression ETL | Tie |
| Distributed join (sf1 to sf100) | Batcher, 1.7x to 2.2x |
| Distributed `filter → count` (sf10, sf100) | Daft, 0.84x to 0.92x |
| TPC-H multi-join queries | Depends on cores: **Daft up to 2x** at 16, **Batcher on 18 of 19** at 96 |
| Per-batch Python UDF | **Daft, ~2×** |

:::{note}
This page mixes three machines. The multimodal table below is a 96-core node, the kernel and
TPC-H tables are a 16-core node, and the distributed table is a 128-CPU Ray cluster. Compare
engines within a table, never a number from one table against a number from another. {doc}`/benchmarks/methodology` has the full list.
:::

## Multimodal ingest

One 96-core node, best-of-3 warm, gated on identical frame counts and output shapes. 2,000
JPEG frames, 640×480 → 224×224:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|:---:|
| **Batcher** | 351 ms | 5,693 img/s | baseline |
| Daft | 838 ms | 2,388 img/s | **2.4x** |

This path started at about 350 img/s, and five fixes took it to 5,693. The one that matters
for the comparison is the media-decode throttle: the per-row decode kernels ran serially, and
the parallel executor capped its rayon pool to the morsel count. A small-JPEG corpus is a
single morsel, so the entire decode ran on one core. See {doc}`/benchmarks/results/multimodal-ingest` for the rest.

A later run of the same benchmark on a **busy** 96-core node (18 to 22 competing test
processes, load average 13) measured 3,041 to 3,427 img/s against Daft's 2,272 to 2,625, so
**1.3x**. Both numbers are real: contention costs the wider engine more than the narrower
one, so read the lower figure as a floor.

Profiling that 1.3x found that most of the remaining gap was not decode. Two things on the
*read* side cost more than the kernels they fed: the per-file header parse ran regardless of
whether the projection asked for it, and local file reads were fanned across a thread pool,
which is right for an object store and **2.5x slower** than a serial loop for a syscall on
page cache (2,000 local JPEGs: 52 ms serial, 118 ms on 8 threads, 130 ms on 64). Both are
fixed. On the same busy node, load 13 to 17:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|:---:|
| **Batcher** | 418-430 ms | 4,649-4,788 img/s | baseline |
| Daft | 780-845 ms | 2,368-2,565 img/s | **1.9x** |

Take the ratio you can reproduce on your own hardware rather than any of ours.

## Curation and augmentation

The comparison changes shape once the pipeline moves past decode. Daft has no native entropy
measure, perceptual hash, or photometric adjustment, so the screening-and-augmentation pass a
corpus needs is a per-row PIL UDF for a Daft user. Against that baseline — the same three
measures, on the same bytes — Batcher's native expressions run **5.7x** faster
(1,298 ms against 7,361 ms for 2,000 frames):

```bash
python benchmarks/scenarios/image_decode.py --suite curate
```

This is a vocabulary difference before it is a speed difference. `entropy`, `phash`,
`ahash`, `colorfulness`, `mean_color`, `is_grayscale`, the eleven photometric adjustments
and the geometry family are engine expressions here and user code there.

## In-memory kernels

`microbench.py` loads roughly 60M TPC-H rows into Arrow once and times each engine's
kernels with no I/O in the way. Single node, 16 cores:

| Operator | Batcher | Daft | Batcher's lead |
|---|---:|---:|:---:|
| filter | 28 ms | 188 ms | **6.7x** |
| sum | 10 ms | 181 ms | **18x** |
| group-by | 359 ms | 487 ms | **1.4x** |

Hold on to this table. It is what makes the next one interesting.

The full 11-case operator mix goes the same way: the latest sweep in `benchmarks/BENCHMARK_RESULTS.md` has **Batcher ahead of Daft on all 11**, and Daft unable to finish `op-window-rank` at all, where `RANK` over roughly 1.5M partitions hangs and Batcher returns in about 148 ms.

## Multi-join SQL

A re-run at scale factor 1 on a 96-core node put **Batcher ahead on 18 of the 19 queries both engines answer**, including q20 at 0.80x, q3 at 0.52x, q17 at 0.20x and q5 at 0.80x. The ratio is `batcher / daft`, so below 1 means Batcher is faster. The join result is a function of core count, so quote the machine with the number. {doc}`/benchmarks/results/tpch` carries the full per-query table.

The parallel radix join and the whole-partition window kernel are what moved this shape, taking q3 to 1.55x at 16 cores and to 0.52x at 96.

One thing did move by a plan change rather than a kernel change. Kyber's build-side
selection used to check broadcast eligibility only on the join's *right* input, so when the
small side arrived on the left the join fell back to shuffling a 6M-row build. Broadcast is
now decided from `min(left_bytes, right_bytes)`. q3 went from 7.7x to 3.8x, and the q5 `orders` to `lineitem` join from 419 ms to 175 ms.

## Where Batcher wins: top-N

`ORDER BY ... DESC LIMIT 20` over sf1 takes 15 ms against Daft's 121 ms, an **8.1x** lead. A fused top-N heap keeps the running best *k* rows, where Daft sorts the relation and then takes twenty. Sort-limit runs 8x to 10x ahead across the shapes tested.

## Distributed

Both engines on the same live Ray cluster (16 worker nodes x 8 CPUs, 128 CPUs, plus a
0-CPU head), each reading TPC-H parquet directly from S3, so the distributed read is part of
the measured work. Daft runs its Ray runner (flotilla), not its local engine. Ratio is
`daft_ms / batcher_ms`, so **above 1 means Batcher is faster**.

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162×** | **208×** | **250×** |
| `join` | **2.23×** | **1.73×** | **1.72×** |
| `groupby` | 1.03× | **1.18×** | **1.30×** |
| `filter_count` | 1.18× | 0.92× | 0.84× |

Batcher takes the join, the group-by, and the metadata-only count. `filter_count` is the
most purely S3-bound shape in the grid: scan one column, filter, count. Both engines read the
same bytes from the same store, so that row measures object-store read throughput rather than
execution, and neither engine can pull far ahead once both are near the network's line rate.
The margins that matter here are on the compute-bound shapes, where the join and the top-N
results sit.

:::{dropdown} The superseded diagnosis, and why it was wrong
An earlier round of this benchmark had Batcher ~10× *behind* Daft at sf100 and diagnosed it
as a distributed-scan throughput ceiling. That diagnosis was wrong about the depth of the
problem. The dominant cause was a control-plane bug: the cluster-fill fan-out was dead, so
any query that ran with Ray already initialized used 2 of 16 workers. Fixing it, with five
other data-movement bugs, produced the table above. The superseded section is kept in
`benchmarks/BENCHMARK_RESULTS.md`. {doc}`/benchmarks/results/scaling` tells the whole story.
:::

## Correctness

The harness declines to time a query whose result does not match DuckDB, so a wrong answer
gets no number rather than a fast one. At sf1 that gate catches Daft on five of the 22:

| Query | What Daft does |
|---|---|
| q6 | Folds `0.06 + 0.01` in IEEE double to `0.06999999999999999`, dropping every `l_discount = 0.07` row: returns 75.2M where the correct revenue is 123.1M |
| q15 | Returns 0 rows where DuckDB returns 1 |
| q18 | Returns `l_quantity` where the query asks for `sum(l_quantity)` |
| q21 | `DaftError::InternalError: Outer reference columns cannot be bound` |
| q22 | Cannot parse `SUBSTRING(x FROM a FOR b)` |

Daft also disagrees with SQL on window frames outside TPC-H: `sum(x) OVER (PARTITION BY k
ORDER BY o)` returns the whole-partition sum where the default frame is `RANGE UNBOUNDED
PRECEDING TO CURRENT ROW`, so on `v = [10, 20, 30]` it gives `60, 60, 60` against DuckDB's
and Batcher's `10, 30, 60`.

Batcher matches DuckDB on all 22, and on the window frame. Where this page reports a speed
loss it means it; correctness is a separate axis and Batcher does not lose on it.

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,daft
python benchmarks/scenarios/image_decode.py
python benchmarks/cluster/vs_ray_daft.py 10
```

## See also

- {doc}`/benchmarks/results/tpch` for the query-by-query picture.
- {doc}`/benchmarks/results/multimodal-ingest` for the image and point-cloud pipelines.
- {doc}`/benchmarks/results/scaling` for the full distributed cluster runs.
- {doc}`/benchmarks/comparisons/vs-duckdb` for the scorecard against the other native engine.
- {doc}`/architecture/deep-dives/operators/join-algorithms` and {doc}`/architecture/deep-dives/operators/morsel-parallelism` for the two mechanisms the loss lives in.
- {doc}`/architecture/deep-dives/adaptive/cost-model` for the build-side selection that took q3 from 7.7x to 3.8x.
- {doc}`/user-guide/transform/columns/udfs` for why a per-batch Python callback costs what it costs.
