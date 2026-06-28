# Batcher vs Ray Data vs Daft — CPU benchmark results

Measured on the Anyscale cluster (9 nodes, 128 CPUs) that hosts this workspace.
**Batcher** runs single-node in-process (its low-overhead strength); **Ray Data**
attaches to the live cluster (`ray.init(address="auto")` — its distributed home
turf); **Daft** runs its native multithreaded local engine (`DAFT_RUNNER=native`).
Every workload is **correctness-gated** (all engines must agree as a sorted row
multiset within float tolerance) before any timing is trusted.

Data: TPC-H `s3://ray-benchmark-data/tpch/parquet/sf1` (lineitem = 6,001,215 rows),
read once into Arrow and shared. Reproduce:

```bash
export PATH=/home/ray/anaconda3/bin:$PATH; unset VIRTUAL_ENV
export BENCH_S3_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2 DAFT_RUNNER=native
python benchmarks/run.py --benchmark operators --tier multi      # batcher/ray/daft operator-mix
python benchmarks/run.py --benchmark tpch --engines batcher,daft # SQL (Ray Data has no SQL)
python benchmarks/scenarios/strength_bench.py                    # representative strength workloads
python benchmarks/scenarios/dist_bench.py --workers 4            # distributed batcher on the cluster
```

## Headline: vs Ray Data, batcher is 50–450× faster (>> the 10× bar)

Ray Data carries a large fixed per-operation cost (task scheduling + block/pandas
bridge), ~300–4500 ms even on the cluster. Batcher, in-process and native, pays
none of it.

**Operator-mix** (`run.py --benchmark operators --tier multi`), `b/ray` = batcher_ms / ray_ms:

| op | batcher_ms | ray_ms | b/ray |
|----|-----------:|-------:|------:|
| groupby-sum   | 14.4 | 1824 | **0.01× (127×)** |
| global-sum    |  4.1 | 1804 | **0.00× (440×)** |
| filter-count  |  6.7 |  310 | **0.02× (46×)** |

**Representative "strength" workloads** (`strength_bench.py`), ratio = engine_ms / batcher_ms (>1 ⇒ batcher faster):

| workload | batcher_ms | ray_ms | daft_ms | vs Ray | vs Daft |
|----------|-----------:|-------:|--------:|-------:|--------:|
| `udf-map` (per-batch numpy UDF + reduce — Ray Data's signature `map_batches`) | 85 | 4283 | 41 | **50×** | 0.5× |
| `expr-etl` (derived cols → filter → 2-agg group-by — Daft's lazy-DF strength)  | 27 | 3490 | 26 | **131×** | 1.0× |
| `top-n` (`ORDER BY … DESC LIMIT 20`)                                            | 15 | 4569 | 121 | **306×** | **8.1×** |

Batcher beats Ray Data **50× even on Ray Data's own `map_batches` pattern**. This is
the structural, reliable 10×+ win.

## vs Daft: competitive — wins top-N, parity on agg/expr, trails on multi-joins

Daft is a mature, fast, multi-core Rust engine (~DuckDB class, ~4 ms fixed overhead).
The honest picture, TPC-H sf1, `b/daft` = batcher_ms / daft_ms (<1 ⇒ batcher faster):

- **Batcher wins:** top-N / sort-limit ~8–10× (fused top-N heap vs full sort).
- **Parity:** global agg, group-by, single-stage expression ETL (~1.0×).
- **Batcher trails:** join-heavy queries `b/daft` 2–12× (q5 9.6×, q7 12×, q9 5.9×,
  q17 6.7×, q20 8.6×); per-batch Python UDF ~2×.

Root cause of the join gap: single-node parallelism is ~1.7–3.8× on 16 cores (vs
Daft ≈ all cores) **and** batcher does ~2× more CPU work per query. Closing it is a
runtime-parallelism + kernel-efficiency effort (see "Improvements" / open levers
below), not a tuning knob — 10×-better-than-Daft on compute-bound single-node is not
reachable by configuration.

Correctness note: batcher matches DuckDB on all 22 queries. Daft computes **q6 wrong**
(mishandles `interval '1' year`: 75.2 M vs the correct 123.1 M) and cannot parse
`SUBSTRING(x FROM a FOR b)` (q22). So the gap to Daft is purely speed, never correctness.

## Distributed batcher on the cluster (distributed-vs-distributed)

`scenarios/dist_bench.py` runs batcher's **distributed** path on the live cluster
(udf-map workload, sf1). Batcher auto-ships its package + native extension to worker
nodes via Ray `runtime_env` py_modules (see "Improvement landed" below), so it "just
works" with `ray_address="auto"`.

| engine | ms |
|--------|---:|
| batcher single-node          | 86 |
| batcher distributed (4 workers) | 92 |
| ray data (cluster)           | 4284 |

The distributed result is **bit-identical to single-node** (correctness gate passes).
At sf1 (6M rows) the data is too small for distribution to win — single-node's
near-zero overhead beats the network shuffle + actor startup, and distributed batcher
is within ~7% rather than paying a large penalty. The point is the path **works,
is correct, and is efficient on the cluster** — and even distributed-vs-distributed,
batcher is **46× faster than Ray Data**. Distribution is for scale-out / larger-than-
memory; at small scale, batcher's single-node mode is the right (and faster) choice.

## Improvement landed this round

**Kyber build-side selection now broadcasts the smaller side of *either* input**
(`kyber/rules/selection.py`). Previously broadcast eligibility was checked only on the
*right* input, so when the cost-delta swap failed to fire and the small side was the
left/probe, the join fell back to shuffling the 6 M-row build. Now broadcast is decided
from `min(left_bytes, right_bytes) ≤ broadcast_max_bytes`, swapping the small side to
build. Effect: TPC-H q3 `b/daft` 7.7× → 3.8×; the q5 orders⋈lineitem join 419 ms → 175 ms.
Verified: 846 differential + 97 join/selection unit tests pass.

**Distributed batcher auto-ships to workers** (`dist/executors/ray_runtime/lifecycle.py`).
When attaching to a cluster, batcher now uploads its own package + abi3 native extension
to worker nodes via Ray `runtime_env` py_modules if it is a source/editable install (a
no-op for a site-packages install the worker image already carries). Before this, the
flight-worker actors died with `ModuleNotFoundError: batcher` on any cluster whose image
didn't pre-install batcher — the distributed path was unusable on a fresh Anyscale
cluster. Verified: distributed == single-node on the live cluster; 5 new unit tests.

## Distributed scale-out (sf10/sf100) — bringing the cluster to bear

The head node has **0 schedulable task CPUs** (Anyscale reserves it), so Daft-native and
batcher-single-node run on the head's 16 physical cores while distributed work uses the
**8 worker nodes = 128 CPUs**. `scenarios/scale_bench.py` reads TPC-H lineitem directly
from S3 at scale and runs a scan-heavy aggregation.

**Distributed-scan read-path fixes (this round):**
- **Per-worker parallel split reads** (`dist/executors/partition_io.py`,
  `_prefetch_split_reads`, `BATCHER_SCAN_PREFETCH=8`). Workers read row-group splits one
  at a time over a single S3 connection (~27 MB/s); now they read N splits ahead on a
  thread pool, overlapping object-store I/O with the map-side fold. **sf10 batcher-dist:
  65 s → 16.6 s (3.9×).**
- **Parquet footer cache** (`io/splits.py`, `_parquet_footer`). Each row-group split
  re-opened the file and re-read the footer from S3 (~100 ms each); a worker reads many
  splits of one file, so the footer is now read once and passed to `ParquetFile(metadata=…)`
  — a warm split drops 268 ms → ~90 ms.
- **S3 trailing-slash bug** (`io/filesystem.py`). `s3://bucket/dir/` (trailing slash)
  failed with "does not exist" — `from_uri` strips the slash from the in-path but the
  prefix math didn't, corrupting the scheme prefix (`s3://` → `s3://r`). Fixed + tested.

**Distribution is even, not skewed:** parquet `splits()` returns one split per row-group
(~60 for sf10), greedily LPT-bin-packed across workers by row count — so the scan load is
balanced. (Join/high-card-group-by skew is handled separately by salting in `par.rs`.)

**Honest measured scale numbers (fair cold reads — fresh frame each run so neither engine
caches):**

| workload (lineitem) | batcher-distributed (8 workers) | Daft native (16 cores) |
|---------------------|--------------------------------:|-----------------------:|
| sf10 (60M rows)     | 16.6 s (was 65 s pre-prefetch)  | ~2–10 s                |
| sf100 (600M rows)   | ~150 s                          | **~10 s (cold)**       |

**The "beat Daft 2×" target is NOT met — Daft is ~10× faster at scale, and this is real
(Daft does not cache: cold ≈ warm ≈ 10 s).** Diagnosis, with what was ruled out:
- **Not CPU-bound.** Giving each worker a full node's 16 cores (`SchedulingEnvelope(num_cpus=16)`)
  left sf10 at ~20 s — same as 1 core. So the gap is *not* parallelizable compute.
- **Not skew.** Scan splits are LPT-balanced; per-worker loads are even.
- **Not memory.** The low-cardinality agg is streaming-bounded; no spill thrash.
- **It is distributed-scan throughput / overhead** (~90 MB/s aggregate across 8 workers,
  roughly constant sf10→sf100). Daft drives far more parallel, coalesced S3 range reads per
  node. Two concrete follow-ups: (1) the prefetch pool isn't delivering its 8× concurrency
  on workers — worth profiling; (2) **default `num_workers` is the driver's `os.cpu_count()`
  (16), not the cluster's 128** (`dist/executor.py`), so distributed batcher under-fans-out
  by default — fix: default the fan-out to `cluster_topology()` CPUs.

This is the straight picture: the read-path work landed here is real and verified, but
closing the remaining ~10× to Daft at scale is a deeper distributed-throughput effort, not
a tuning knob.

## Open levers (next, highest-leverage first)

1. **Build-once broadcast** — the parallel `broadcast_join` rebuilds the build-side
   hash table in every probe chunk; build once and share it (`bc_runtime::join`).
2. **Parallelize the shuffle path** — `key_indices` / `partition_by_keys` over the 6 M
   probe side run serially before the per-bucket join (caps parallelism at ~3.8×).
3. **Source-side NDV sketches** — cold-start join cardinality falls back to
   `max(left,right)` (assumes many-to-one), estimating many-to-many low-NDV joins 64–80×
   low and steering join order into 12–18 M-row intermediates (q5 cold 7115 ms vs warm
   300 ms). Feed HLL NDV on base join keys as `SourceStatistics`.
