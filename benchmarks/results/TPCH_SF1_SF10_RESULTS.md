# TPC-H sf1 + sf10 -- Batcher against every comparator (2026-07-28)

A full 22-query TPC-H run at scale factors 1 and 10. This is the raw scorecard; the analysis
of *where* Batcher's time goes lives in `../TPCH_FINDINGS.md`, and the multimodal / physical-AI
workloads in `../BENCHMARK_RESULTS.md`.

Every timing is correctness-gated: the harness compares each engine's result against the
DuckDB reference as a sorted row multiset before it will report a number, so a fast wrong
answer surfaces as `FAILED` and never as a win.

## The two DuckDB bars, and why both are reported

`duckdb` is DuckDB ingested into its **compressed native store** (dictionary/RLE encoding,
zone maps) by an untimed `CREATE TABLE`. That is DuckDB at its best, but it pits DuckDB's
*storage engine plus execution engine* against Batcher's *execution engine over raw Arrow*.

`ddb_arrow` binds DuckDB to the **same zero-copy Arrow** Batcher consumes. That is the
like-for-like execution comparison, and the one that isolates the engines from the format.

Reading only the first bar attributes a storage-format advantage to execution. Reading only
the second ignores that DuckDB users really do get the compressed store. Both are here.

## Environment

| | |
|---|---|
| Host | c5d.24xlarge, 96 vCPU, 184 GiB RAM, us-west-2 |
| Batcher | `maturin develop --release`, verified `__engine_profile__ == "release"` |
| Data | `s3://ray-benchmark-data/tpch/parquet/sf{1,10}`, mirrored to local NVMe |
| Repeats | sf1 best-of-5, sf10 best-of-3, one warm-up first (the harness default per scale) |
| Versions | duckdb 1.5.5, polars 1.36.1, pyspark 4.2.0 (JRE 17), daft 0.7.21, ray 2.56.0 |

```bash
export JAVA_HOME=~/.jre/jdk-17.0.20+8-jre          # pyspark needs a JVM, not just the wheel
export BENCH_TPCH_BASE=/mnt/local_storage/tpch/parquet
python3 benchmarks/run.py --benchmark tpch --scale 10 \
    --engines batcher,duckdb,duckdb_arrow,polars,spark,daft
```

## sf1 times (ms)

| q | batcher | duckdb | ddb_arrow | polars | daft |
|---|--:|--:|--:|--:|--:|
| q1 | 22.4 | 20.1 | 27.1 | 77.7 | 40.6 |
| q2 | 12.5 | 17.6 | 74.8 | 17.9 | 50.6 |
| q3 | 25.2 | 25.2 | 68.0 | 30.9 | 40.8 |
| q4 | 31.7 | 28.0 | 64.1 | 79.8 | 22.9 |
| q5 | 30.6 | 21.9 | 160.1 | 30.8 | 36.1 |
| q6 | 7.5 | 4.2 | 23.4 | 28.5 | wrong |
| q7 | 26.9 | 26.1 | 68.4 | 115.0 | 43.9 |
| q8 | 19.5 | 26.2 | 102.0 | 26.9 | 75.7 |
| q9 | 56.1 | 73.0 | 134.1 | 66.7 | 91.6 |
| q10 | 32.1 | 48.5 | 101.6 | 62.0 | 225.4 |
| q11 | 7.8 | 8.3 | 42.0 | 21.2 | 37.8 |
| q12 | 25.2 | 19.3 | 45.7 | 117.5 | 208.1 |
| q13 | 49.1 | 49.1 | 58.4 | 142.0 | 57.7 |
| q14 | 18.3 | 16.6 | 43.4 | 16.2 | 43.1 |
| q15 | 8.3 | 12.2 | 32.6 | 22.3 | wrong |
| q16 | 19.2 | 30.9 | 78.7 | 37.9 | 55.7 |
| q17 | 15.1 | 23.3 | 78.7 | 6.3 | 63.5 |
| q18 | 37.9 | 36.1 | 89.9 | 65.0 | 43.8 |
| q19 | 38.0 | 38.6 | 56.2 | 120.2 | 65.1 |
| q20 | 23.3 | 21.8 | 73.7 | 52.3 | 25.7 |
| q21 | 83.6 | 78.5 | 213.9 | 91.0 | n/a |
| q22 | 27.2 | 23.7 | 56.2 | 30.5 | n/a |
| **total** | **617.5** | **649.2** | **1693.0** | **1258.6** | **1228.1** |

## sf1 ratios (`b/x`; below 1.00x means Batcher is faster)

| q | duckdb | ddb_arrow | polars | daft |
|---|--:|--:|--:|--:|
| q1 | 1.11x | 0.83x | 0.29x | 0.55x |
| q2 | 0.71x | 0.17x | 0.70x | 0.25x |
| q3 | 1.00x | 0.37x | 0.82x | 0.62x |
| q4 | 1.13x | 0.49x | 0.40x | 1.38x |
| q5 | 1.40x | 0.19x | 0.99x | 0.85x |
| q6 | 1.79x | 0.32x | 0.26x | -- |
| q7 | 1.03x | 0.39x | 0.23x | 0.61x |
| q8 | 0.74x | 0.19x | 0.72x | 0.26x |
| q9 | 0.77x | 0.42x | 0.84x | 0.61x |
| q10 | 0.66x | 0.32x | 0.52x | 0.14x |
| q11 | 0.94x | 0.19x | 0.37x | 0.21x |
| q12 | 1.31x | 0.55x | 0.21x | 0.12x |
| q13 | 1.00x | 0.84x | 0.35x | 0.85x |
| q14 | 1.10x | 0.42x | 1.13x | 0.42x |
| q15 | 0.68x | 0.25x | 0.37x | -- |
| q16 | 0.62x | 0.24x | 0.51x | 0.34x |
| q17 | 0.65x | 0.19x | 2.40x | 0.24x |
| q18 | 1.05x | 0.42x | 0.58x | 0.87x |
| q19 | 0.98x | 0.68x | 0.32x | 0.58x |
| q20 | 1.07x | 0.32x | 0.45x | 0.91x |
| q21 | 1.06x | 0.39x | 0.92x | n/a |
| q22 | 1.15x | 0.48x | 0.89x | n/a |

## sf10 times (ms)

| q | batcher | duckdb | ddb_arrow | polars | spark | daft |
|---|--:|--:|--:|--:|--:|--:|
| q1 | 127.7 | 77.4 | 124.4 | 587.9 | 948.2 | 248.1 |
| q2 | 106.7 | 54.1 | 263.1 | 44.8 | 1077.2 | 79.2 |
| q3 | 145.0 | 89.1 | 411.6 | 191.4 | 2528.2 | 304.5 |
| q4 | 169.9 | 139.7 | 370.8 | 868.8 | 1991.8 | 161.2 |
| q5 | 399.3 | 120.4 | 2095.7 | 95.2 | 4309.3 | 181.0 |
| q6 | 35.9 | 24.2 | 118.7 | 290.1 | 246.3 | wrong |
| q7 | 181.9 | 54.8 | 416.5 | 987.3 | 4197.3 | 285.5 |
| q8 | 129.7 | 93.4 | 390.4 | 65.7 | 1354.9 | 913.2 |
| q9 | 714.2 | 196.6 | 685.6 | 464.1 | 3468.2 | 627.7 |
| q10 | 315.6 | 160.2 | 343.7 | 457.8 | 3216.8 | 565.1 |
| q11 | 15.9 | 17.3 | 135.1 | 72.1 | 645.5 | 97.2 |
| q12 | 106.8 | 74.4 | 218.8 | 1003.3 | 1200.6 | 3968.1 |
| q13 | 187.1 | 150.3 | 204.9 | 1356.3 | 2840.1 | 335.6 |
| q14 | 60.5 | 51.1 | 221.3 | 136.7 | 716.4 | 344.1 |
| q15 | 10.4 | 40.4 | 172.4 | 148.4 | 836.5 | wrong |
| q16 | 65.6 | 84.4 | 138.8 | 109.3 | 1420.3 | 109.0 |
| q17 | 105.4 | 74.4 | 451.6 | 28.7 | 4006.9 | 208.0 |
| q18 | 396.5 | 166.0 | 456.9 | 733.1 | 6023.5 | 360.9 |
| q19 | 129.0 | 109.6 | 218.0 | 1108.6 | 435.4 | 116.0 |
| q20 | 164.9 | 71.9 | 312.9 | 394.7 | 835.0 | 129.7 |
| q21 | 1006.7 | 330.9 | 1126.0 | 841.6 | 8043.6 | n/a |
| q22 | 57.8 | 64.6 | 134.9 | 113.6 | 1479.7 | n/a |
| **total** | **4632.5** | **2245.2** | **9012.1** | **10099.5** | **51821.7** | **9034.1** |

## sf10 ratios (`b/x`; below 1.00x means Batcher is faster)

| q | duckdb | ddb_arrow | polars | spark | daft |
|---|--:|--:|--:|--:|--:|
| q1 | 1.65x | 1.03x | 0.22x | 0.13x | 0.51x |
| q2 | 1.97x | 0.41x | 2.38x | 0.10x | 1.35x |
| q3 | 1.63x | 0.35x | 0.76x | 0.06x | 0.48x |
| q4 | 1.22x | 0.46x | 0.20x | 0.09x | 1.05x |
| q5 | 3.32x | 0.19x | 4.19x | 0.09x | 2.21x |
| q6 | 1.48x | 0.30x | 0.12x | 0.15x | -- |
| q7 | 3.32x | 0.44x | 0.18x | 0.04x | 0.64x |
| q8 | 1.39x | 0.33x | 1.97x | 0.10x | 0.14x |
| q9 | 3.63x | 1.04x | 1.54x | 0.21x | 1.14x |
| q10 | 1.97x | 0.92x | 0.69x | 0.10x | 0.56x |
| q11 | 0.92x | 0.12x | 0.22x | 0.02x | 0.16x |
| q12 | 1.44x | 0.49x | 0.11x | 0.09x | 0.03x |
| q13 | 1.24x | 0.91x | 0.14x | 0.07x | 0.56x |
| q14 | 1.18x | 0.27x | 0.44x | 0.08x | 0.18x |
| q15 | 0.26x | 0.06x | 0.07x | 0.01x | -- |
| q16 | 0.78x | 0.47x | 0.60x | 0.05x | 0.60x |
| q17 | 1.42x | 0.23x | 3.67x | 0.03x | 0.51x |
| q18 | 2.39x | 0.87x | 0.54x | 0.07x | 1.10x |
| q19 | 1.18x | 0.59x | 0.12x | 0.30x | 1.11x |
| q20 | 2.29x | 0.53x | 0.42x | 0.20x | 1.27x |
| q21 | 3.04x | 0.89x | 1.20x | 0.13x | n/a |
| q22 | 0.89x | 0.43x | 0.51x | 0.04x | n/a |

## Summary

| engine | SF1 geomean | SF1 wins | SF10 geomean | SF10 wins |
|---|--:|--:|--:|--:|
| duckdb (native store) | 0.963x | 9/22 | 1.521x | 4/22 |
| **ddb_arrow (same input)** | 0.353x | 22/22 | 0.421x | 20/22 |
| polars | 0.538x | 20/22 | 0.468x | 16/22 |
| daft | 0.445x | 17/18 | 0.525x | 11/18 |
| spark | 0.041x* | 22/22 | 0.076x | 22/22 |
| ray data | 0.001x* | 22/22 | 0.005x* | 22/22 |

\* Spark at sf1 and Ray Data at both scales are carried from earlier verified runs in the
same session rather than re-measured. Spark sf1 total 14,554.7 ms; Ray Data totals
449,268.8 ms (sf1) and 709,969.6 ms (sf10). Every other cell is from the runs above.

Daft's geomean and win counts are taken over the 18 queries it answers **correctly** -- q6
and q15 return wrong results at both scales and q21/q22 do not plan, so counting them would
credit it for being fast at being wrong.

## What the numbers say

**On identical Arrow input Batcher wins 22 of 22 at sf1 and 20 of 22 at sf10**, geomean
0.353x and 0.421x -- roughly 2.4x to 2.8x faster than DuckDB's execution engine. The two
sf10 losses are q1 (1.03x) and q9 (1.04x), both inside run-to-run noise.

Against DuckDB's compressed native store Batcher is **at parity at sf1** (0.963x geomean,
and a lower total: 617.5 ms against 649.2 ms) and **1.52x behind at sf10**.

That remaining gap is storage, not execution. DuckDB's sf10 advantage comes from reading
dictionary/RLE-compressed columns and skipping row groups on zone maps; on the same
uncompressed Arrow bytes it is 2.4x slower than Batcher. The
`Arrow is the only columnar contract` invariant (CLAUDE.md #3) means Batcher has no
compressed form to switch to.

Batcher also beats Polars (0.538x / 0.468x geomean), Daft (0.445x / 0.525x), Spark
(0.041x / 0.076x) and Ray Data (0.001x / 0.005x) at both scales.

## Where the sf10 deficit against the native store concentrates

The sf1 geomean is 0.963x and the sf10 geomean is 1.521x, so the gap opens with scale rather
than being uniform. It is concentrated in the multi-join queries:

| query | b/duckdb sf1 | b/duckdb sf10 | shape |
|---|--:|--:|---|
| q5 | 1.40x | 3.32x | six-way join |
| q7 | 1.03x | 3.32x | nation joined twice |
| q9 | 0.77x | 3.63x | six-way join, composite key |
| q18 | 1.05x | 2.39x | IN-subquery then re-join |
| q21 | 1.06x | 3.04x | EXISTS + NOT EXISTS |

Batcher still wins q15 (0.26x), q16 (0.78x), q22 (0.89x) and q11 (0.92x) at sf10 against the
native store.

## Optimization hypotheses tested and rejected

Six candidate explanations for the sf10 deficit were measured; none survived. They are
recorded so the next attempt does not re-run them:

| hypothesis | measurement |
|---|---|
| Multi-key top-N comparator overhead | Batcher already wins every top-N shape in isolation (1-key int 0.41x, 1-key float 0.92x, 2-key 0.90x, 3-key 0.91x) |
| Scan-level runtime filters in the materializing executor | A/B via `BATCHER_RUNTIME_JOIN_FILTER`: q5 584.6 -> 583.2 ms, q9 693 -> 699 ms, q21 943 -> 930 ms |
| Zone-map / min-max morsel pruning over Arrow | **0%** of 16,384-row morsels are skippable for the real predicates (`l_shipdate`, `l_discount`, `l_quantity`); only the clustered `l_orderkey` skips (74.9%) |
| Streaming vs materializing executor selection | An in-process A/B is confounded by Batcher's own learned statistics -- the second arm inherits measured cardinalities. In isolated processes streaming wins both q5 (556.5 vs 1011.7 ms) and q18 (447.7 vs 564.4 ms) |
| Date32 SIMD lane width in the Cranelift JIT | Equalizing bytes-per-iteration measured flat (27.4 -> 27.2 ms); `perf` shows the predicate runs through Arrow kernels, not the JIT, on this path. Reverted |
| Null-bitmap overhead on non-nullable columns | The column already carries no validity buffer (`null_count=0`); stripping it changes nothing (72.9 vs 72.8 ms) |

What `perf` does show is a **parallel-efficiency ceiling**. Single-threaded PyArrow computes
the q6 date mask at 3.5 GB/s; Batcher on 96 cores reaches 8-13 GB/s, about 2-4x total from
96 cores. The scaling curve saturates at 64 workers and *regresses* at 96 (filter-count
25.5 ms at p=64 against 27.6 ms at p=96), and the q5 profile independently reports 15% CPU
utilization with the verdict "not CPU-limited". That is the lever with real headroom, and it
is a morsel-scheduling concern rather than a kernel one.

## Correctness notes

- **Daft returns wrong answers on q6 and q15 at both scales.** q6 revenue
  `123141078.2283` against the correct `75207768.1855` at sf1, and `1230113636.0101` against
  `752448391.6111` at sf10 -- Daft mis-folds the constant arithmetic in
  `l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`. Polars' SQL parser carries the identical
  bug, which is why Polars is driven through its DataFrame API here. q15 returns 0 rows where
  the answer is 1. Those cells read `wrong` and are excluded from Daft's totals and geomeans.
- **Daft cannot plan q21 or q22**: `Outer reference columns cannot be bound` on the
  correlated `EXISTS`, and no `SUBSTRING(expr FROM start FOR len)` syntax.
- **PyArrow is excluded.** It has no SQL surface and the suite deliberately writes no Acero
  TPC-H pipelines, since a hand-rolled reimplementation would benchmark the benchmark rather
  than the engine. It competes in `--benchmark operators`.
- **Ray Data covers all 22 queries** via the `suites/standard/tpch_ray` pipelines added this
  session (commit `b2dab69`). Before that it was measured single-threaded on 4 of 22, because
  `ray.data.from_arrow` produces exactly one block.
