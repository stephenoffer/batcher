# vs Daft

This page compares Batcher against Daft on single-node and distributed work.

Daft is a mature, fast, multi-core Rust engine, roughly DuckDB-class, with about 4 ms of fixed overhead. It's the closest competitor Batcher has on single-node compute, and the result is genuinely mixed. Batcher takes multimodal ingest and top-N by large margins, ties on aggregation and single-stage expression ETL, and still trails on the join-heavy TPC-H queries.

:::{important}
Daft computes TPC-H **q6 wrong**: it folds `0.06 + 0.01` in IEEE double to `0.06999999999999999`, dropping every `l_discount = 0.07` row, and returns 75.2M where
the correct revenue is 123.1M. The harness declines to time a query whose result does not
match the oracle, so Daft gets no number on q6 rather than a fast one. Every timing on this
page passed that gate first.
:::

## Scorecard

| Shape | Winner |
|---|---|
| Image decode → tensor | Batcher, 2.4× |
| Top-N / sort-limit | Batcher, 8x to 10x |
| In-memory filter / sum kernels | Batcher, 6.7× / 18× |
| Global aggregate, group-by, expression ETL | Tie |
| Distributed join (sf1 to sf100) | Batcher, 1.7x to 2.2x |
| Distributed `filter → count` (sf10, sf100) | Daft, 0.84x to 0.92x |
| TPC-H multi-join queries | **Daft, up to 2x** |
| Per-batch Python UDF | **Daft, ~2×** |

:::{note}
This page mixes three machines. The multimodal table below is a 96-core node, the kernel and
TPC-H tables are a 16-core node, and the distributed table is a 128-CPU Ray cluster. Compare
engines within a table, never a number from one table against a number from another. {doc}`methodology` has the full list.
:::

## Multimodal ingest

One 96-core node, best-of-3 warm, gated on identical frame counts and output shapes. 2,000
JPEG frames, 640×480 → 224×224:

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|:---:|
| **Batcher** | 351 ms | 5,693 img/s | baseline |
| Daft | 838 ms | 2,388 img/s | **2.4×** |
| Ray Data | 2,136 ms | 936 img/s | **6.1×** |

This started the session at ~350 img/s, *losing to both*. Five fixes took it to 5,693. The
one that matters for the comparison is the media-decode throttle: the per-row decode kernels
ran serially, and the parallel executor capped its rayon pool to the morsel count. A
small-JPEG corpus is a single morsel, so the entire decode ran on one core. See {doc}`multimodal-ingest` for the rest.

## In-memory kernels

`microbench.py` loads roughly 60M TPC-H rows into Arrow once and times each engine's
kernels with no I/O in the way. Single node, 16 cores:

| Operator | Batcher | Daft | Batcher's lead |
|---|---:|---:|:---:|
| filter | 28 ms | 188 ms | **6.7×** |
| sum | 10 ms | 181 ms | **18×** |
| group-by | 359 ms | 487 ms | **1.4×** |

Hold on to this table. It is what makes the next one interesting.

## Where Daft wins: multi-join SQL

:::{warning}
**Daft is faster on the join-heavy TPC-H queries**, led by q20 at 2.03x, q3 at 1.55x, and q4 at 1.51x. It's also about **2x faster on a per-batch Python UDF**, where a numpy `map_batches` reduce takes 85 ms against Daft's 41 ms. Parity on global aggregation, group-by, and single-stage expression ETL doesn't offset that.
:::

TPC-H at scale factor 1, 16 cores, re-measured 2026-07-18. The ratio is `batcher / daft`, so **above 1 means Daft is faster**. Batcher is faster on 11 of the 18 queries Daft answers correctly. {doc}`tpch` carries the full per-query table.

| Query | vs Daft |
|---|---:|
| q20 | 2.03x |
| q3 | 1.55x |
| q4 | 1.51x |
| q17 | 1.35x |
| q5 | 1.13x |

Earlier revisions of this page reported Daft 4x to 12x ahead here. That gap is largely gone. The parallel radix join and the whole-partition window kernel landed since, taking q3 from 3.8x behind to 1.55x.

Given the kernel table above, the remaining join gap isn't the filter and isn't the aggregate. Single-node parallelism plateaus after about 8 cores where Daft uses effectively all 16, and Batcher does roughly 2x more CPU work per query. That isn't reachable by configuration. It's a runtime-parallelism and kernel-efficiency effort, and it's the top open lever in `benchmarks/BENCHMARK_RESULTS.md`.

One thing did move by a plan change rather than a kernel change. Kyber's build-side
selection used to check broadcast eligibility only on the join's *right* input, so when the
small side arrived on the left the join fell back to shuffling a 6M-row build. Broadcast is
now decided from `min(left_bytes, right_bytes)`. q3 went from 7.7x to 3.8x, and the q5 `orders` to `lineitem` join from 419 ms to 175 ms.

## Where Batcher wins: top-N

`ORDER BY ... DESC LIMIT 20` over sf1 takes 15 ms against Daft's 121 ms, an **8.1x** lead. A fused top-N heap keeps the running best *k* rows, where Daft sorts the relation and then takes twenty. Sort-limit runs 8x to 10x ahead across the shapes tested.

## Distributed

All three engines on the same live Ray cluster (16 worker nodes × 8 CPUs, 128 CPUs, plus a
0-CPU head), each reading TPC-H parquet directly from S3, so the distributed read is part of
the measured work. Daft runs its Ray runner (flotilla), not its local engine. Ratio is
`daft_ms / batcher_ms`, so **above 1 means Batcher is faster**.

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162×** | **208×** | **250×** |
| `join` | **2.23×** | **1.73×** | **1.72×** |
| `groupby` | 1.03× | **1.18×** | **1.30×** |
| `filter_count` | 1.18× | 0.92× | 0.84× |

Batcher wins the join, the group-by, and the metadata-only count, and **loses
`filter_count` at sf10 and sf100**. That loss is the most purely S3-bound shape there is:
scan one column, filter, count. Both engines read the same bytes from the same store, and the
difference is object-store read throughput, not execution.

Be clear about the ceiling here: the 10× bar Batcher clears against Ray Data is *not*
attainable against Daft on these shapes. Daft is also a native Rust engine reading the same
parquet from the same bucket. On an I/O-bound scan, no execution engine can be 10× faster
than another that is already at a similar fraction of the network's line rate.

:::{dropdown} The superseded diagnosis, and why it was wrong
An earlier round of this benchmark had Batcher ~10× *behind* Daft at sf100 and diagnosed it
as a distributed-scan throughput ceiling. That diagnosis was wrong about the depth of the
problem. The dominant cause was a control-plane bug: the cluster-fill fan-out was dead, so
any query that ran with Ray already initialized used 2 of 16 workers. Fixing it, with five
other data-movement bugs, produced the table above. The superseded section is kept in
`benchmarks/BENCHMARK_RESULTS.md`. {doc}`scaling` tells the whole story.
:::

## Correctness

Daft computes TPC-H **q6 wrong**. It folds `0.06 + 0.01` in IEEE double to `0.06999999999999999`, dropping every `l_discount = 0.07` row, and returns 75.2M where
the correct revenue is 123.1M. It also cannot parse the `SUBSTRING(x FROM a FOR b)` in q22.
The harness declines to time a query whose result does not match, so Daft gets no number on
q6 rather than a fast one.

Batcher matches DuckDB on all 22. The gap to Daft is purely speed, never correctness, and this page
would rather say that than pretend the speed gap does not exist.

## Reproduce

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,daft
python benchmarks/scenarios/image_decode.py
python benchmarks/cluster/vs_ray_daft.py 10
```

## See also

- {doc}`tpch` for the query-by-query picture.
- {doc}`multimodal-ingest` for the image and point-cloud pipelines.
- {doc}`scaling` for the full distributed cluster runs.
- {doc}`vs-duckdb` for the same join gap against the other native engine.
- {doc}`../deep-dives/join-algorithms` and {doc}`../deep-dives/morsel-parallelism` for the two mechanisms the loss lives in.
- {doc}`../deep-dives/cost-model` for the build-side selection that took q3 from 7.7x to 3.8x.
- {doc}`../user-guide/udfs` for why a per-batch Python callback costs what it costs.
