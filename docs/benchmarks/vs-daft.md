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

Each row is one workload shape, with the engine that won it and by how much. Read the
ratios alongside the methodology above, not on their own:

| Shape | Winner |
|---|---|
| Image decode → tensor | Batcher, 2.4× |
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
engines within a table, never a number from one table against a number from another. {doc}`methodology` has the full list.
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
single morsel, so the entire decode ran on one core. See {doc}`multimodal-ingest` for the rest.

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

## Multi-join SQL: it depends on the machine

:::{warning}
**On 16 cores Daft is faster on the join-heavy TPC-H queries**, led by q20 at 2.03x, q3 at 1.55x, and q4 at 1.51x. It's also about **2x faster on a per-batch Python UDF**, where a numpy `map_batches` reduce takes 85 ms against Daft's 41 ms.

**On 96 cores that reverses.** A re-run at sf1 on a 96-core node put Batcher ahead on 18 of the 19 queries both engines answer, including q20 at 0.80x, q3 at 0.52x, q17 at 0.20x and q5 at 0.80x. q4 is the one that stays Daft's, at 1.41x rather than 1.51x. Both measurements are real; the join result is a function of core count, so quote the machine with the number.
:::

That reversal also puts a question against the explanation below. If the gap were purely that Batcher's single-node parallelism plateaus around 8 cores while Daft uses all 16, more cores should widen it rather than close it on four of five queries. Either work landed since this section was written, or the diagnosis is incomplete. It is left standing, with this note, rather than quietly rewritten.

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

Batcher takes the join, the group-by, and the metadata-only count, and trails on
`filter_count` at sf10 and sf100 by 8% to 16%. That shape is the most purely S3-bound in the
grid: scan one column, filter, count. Both engines read the same bytes from the same store,
and the difference is object-store read throughput rather than execution.

That also sets the ceiling for this page. Daft is a native Rust engine reading the same
parquet from the same bucket, so on an I/O-bound scan neither engine can pull far ahead of
the other once both are near the network's line rate. The margins that matter here are on
the compute-bound shapes, where the join and the top-N results sit.

:::{dropdown} The superseded diagnosis, and why it was wrong
An earlier round of this benchmark had Batcher ~10× *behind* Daft at sf100 and diagnosed it
as a distributed-scan throughput ceiling. That diagnosis was wrong about the depth of the
problem. The dominant cause was a control-plane bug: the cluster-fill fan-out was dead, so
any query that ran with Ray already initialized used 2 of 16 workers. Fixing it, with five
other data-movement bugs, produced the table above. The superseded section is kept in
`benchmarks/BENCHMARK_RESULTS.md`. {doc}`scaling` tells the whole story.
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

- {doc}`tpch` for the query-by-query picture.
- {doc}`multimodal-ingest` for the image and point-cloud pipelines.
- {doc}`scaling` for the full distributed cluster runs.
- {doc}`vs-duckdb` for the same join gap against the other native engine.
- {doc}`/deep-dives/operators/join-algorithms` and {doc}`/deep-dives/operators/morsel-parallelism` for the two mechanisms the loss lives in.
- {doc}`/deep-dives/adaptive/cost-model` for the build-side selection that took q3 from 7.7x to 3.8x.
- {doc}`/user-guide/transform/udfs` for why a per-batch Python callback costs what it costs.
