# vs Daft

This page compares Batcher with Daft on single-node analytics, multimodal ingest and distributed pipelines.

Daft is a fast multi-core Rust engine with a strong multimodal story, which makes it Batcher's closest peer on AI data work. Batcher is faster on every analytical suite measured against it, often by 5x to 10x, takes image decode by about 2x, and takes the distributed join by 1.7x to 2.2x on the same Ray cluster.

:::{important}
Every timing passed the correctness gate first, and the gate catches Daft on several TPC-H queries. Daft returns the wrong revenue on q6, folding `0.06 + 0.01` in IEEE double to `0.06999999999999999` and dropping every `l_discount = 0.07` row, and returns 75.2M where the correct answer is 123.1M. A wrong answer gets no ratio, so Daft's geomeans below cover only the queries it answers correctly.
:::

## The suites

The five-engine board of 2026-08-28 ran on a 92-core box, best of five at sf1 and best of three at sf10, one process per suite. Each cell is a suite geomean of `batcher_ms / daft_ms`, so **below 1.00 means Batcher is faster**:

| Suite | vs Daft |
|---|---:|
| Semi-structured JSON | **0.04** |
| Operator mix | **0.07** |
| ClickBench | **0.11** |
| TPC-H sf10 | **0.17** |
| TPC-H sf1 | **0.21** |
| H2O.ai `join` | **0.34** |
| H2O.ai `groupby` | **0.38** |

The 2026-07-28 TPC-H board, on a c5d.24xlarge with 96 vCPU, read the same way per query: Batcher faster on 17 of the 18 queries Daft answers correctly at sf1. An earlier operator sweep put Batcher ahead of Daft on all 11 operators, and Daft couldn't complete any of the four window operators on `lineitem`, where `RANK` over about 1.5M partitions hangs and Batcher returns in about 148 ms.

## Multimodal ingest

The benchmark decodes 2,000 JPEG frames and resizes them from 640x480 to 224x224, gated on identical frame counts and output shapes. On an idle 96-core node, best of three warm (2026-07-11):

| Engine | Time | Throughput | Batcher's lead |
|---|---:|---:|---:|
| **Batcher** | 351 ms | 5,693 img/s | |
| Daft | 838 ms | 2,388 img/s | **2.4x** |

A later run on the same node under load from other sessions, after two read-side fixes, measured Batcher at 4,649 to 4,788 img/s against Daft 0.7.23 at 2,368 to 2,565, a lead of **1.87x to 1.96x**. Contention costs the wider engine more, so read the busy-node figure as a floor. {doc}`/benchmarks/results/multimodal-ingest` has both runs and the fixes behind them.

Past decode the comparison changes shape. Daft has no native entropy measure, perceptual hash or photometric adjustment, so screening and augmenting a corpus is a per-row Pillow UDF for a Daft user. Batcher's native expressions ran the same three measures on the same 2,000 frames **5.7x** faster than a per-row Pillow loop. `entropy`, `phash`, `ahash`, `colorfulness`, `mean_color`, `is_grayscale`, the photometric adjustments and the geometry family are engine expressions in Batcher and user code in Daft.

## Top-N

`ORDER BY ... LIMIT` is Batcher's widest single-operator margin over Daft. A fused top-N heap keeps only the running best rows, where Daft sorts the relation and then takes the head. Sort-limit ran 8x to 10x ahead at TPC-H sf1 in the record's Daft comparison.

## Distributed

Both engines attach to the same live Ray cluster, 16 worker nodes of 8 CPUs each (128 CPUs) plus a head node with no CPUs, and read TPC-H Parquet directly from S3, so the distributed read is part of the measured work. Daft runs its Ray runner rather than its local engine (2026-07-12). Ratios here are `daft_ms / batcher_ms`, so **above 1 means Batcher is faster**:

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162x** | **208x** | **250x** |
| `join` | **2.23x** | **1.73x** | **1.72x** |
| `groupby` | 1.03x | **1.18x** | **1.30x** |
| `filter_count` | **1.18x** | 0.92x | 0.84x |

Batcher takes the join at every scale, the group-by lead widens with the data, and the metadata count never scans at all. `filter_count` is the most purely S3-bound pipeline in the grid, so that row measures object-store read throughput rather than execution.

GPU inference is measured end to end on a cluster too. Scoring 100,000 images on six single-T4 nodes with the identical seeded network, Batcher finished in 18.72 s against Daft's 101.10 s, **5.40x** faster, with matching checksums (2026-09-06).

:::{dropdown} An earlier diagnosis, and why it was wrong
An earlier round of the distributed benchmark put Batcher about 10x behind Daft at sf100 and blamed distributed-scan throughput. The dominant cause was a control-plane bug: the cluster-fill fan-out was dead, so any query that ran with Ray already initialized used 2 of 16 workers. Fixing it, with several data-movement bugs, produced the table above. {doc}`/benchmarks/results/scaling` tells the whole story.
:::

## Correctness

At sf1 the gate catches Daft on five of the 22 TPC-H queries. The following table lists what it does:

| Query | What Daft does |
|---|---|
| q6 | Folds `0.06 + 0.01` in IEEE double, dropping every `l_discount = 0.07` row: 75.2M where the answer is 123.1M |
| q15 | Returns 0 rows where the answer has 1 |
| q18 | Returns `l_quantity` where the query asks for `sum(l_quantity)` |
| q21 | Can't plan the correlated subquery: `Outer reference columns cannot be bound` |
| q22 | Can't parse `SUBSTRING(x FROM a FOR b)` |

Batcher matches DuckDB on all 22.

## Requirements and limitations

The following results are where Daft leads or where a figure needs its context:

- **Distributed `filter_count`** at sf10 and sf100 reads 0.92x and 0.84x, on the shape bound by object-store reads.
- **A per-batch Python UDF** ran about 2x faster on Daft in the record's single-node comparison.
- **Multimodal ratios depend on the machine and its load**, from 1.9x on a busy node to 2.4x on an idle one. Reproduce the ratio on your own hardware before quoting one.

## Reproduce

The following commands rerun each result. `vs_ray_daft.py` takes one scale factor per run:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,daft
python benchmarks/run.py --benchmark operators --tier multi
python benchmarks/scenarios/image_decode.py
python benchmarks/scenarios/image_decode.py --suite curate
python benchmarks/cluster/vs_ray_daft.py 10
python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
```

## See also

- {doc}`/benchmarks/results/tpch` for the query-by-query picture.
- {doc}`/benchmarks/results/multimodal-ingest` for the image, point-cloud, audio and video pipelines.
- {doc}`/benchmarks/results/scaling` for the full distributed runs.
- {doc}`/benchmarks/comparisons/vs-duckdb` for the scorecard against the other native engine.
- {doc}`/architecture/deep-dives/operators/sort-internals` for the fused top-N heap.
- {doc}`/user-guide/transform/columns/udfs` for why a per-batch Python callback costs what it costs.
