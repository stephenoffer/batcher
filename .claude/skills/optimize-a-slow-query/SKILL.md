---
name: optimize-a-slow-query
description: Make a correct-but-slow Batcher query fast — measure before guessing, read the plan, then work an ordered checklist of the highest-leverage fixes (pushdown, join order and build side, streaming instead of collect, caching, file layout, spill, batch size, parallelism, adaptive re-optimization, distribution), and benchmark honestly against DuckDB/Polars. Invoke when a query returns the right answer too slowly.
---

# Optimize a slow query

Order matters. Every step below is cheaper and higher-yield than the one after it, and
step 1 is not optional — most "optimizations" applied without a measurement move nothing
or make things worse.

If the query is *wrong*, stop: use `debug-a-batcher-query`. A fast wrong answer is a bug.

## 0. Rule out the trivial causes

- **Are you on a release build?** A debug build is roughly 10x slower.
  `just build-release` (`maturin develop --release`). The benchmark harness warns about
  this automatically; nothing else does.
- **Are you timing the plan or the work?** `Dataset` is lazy — transformations cost
  nothing and terminal ops cost everything. Time the terminal op.
- **Is the wall clock dominated by a warm-up?** Time a second run too.

## 1. Measure — never guess

```python
print(ds.explain(analyze=True))   # executes, then annotates every op
stats = ds.stats()                # RunStats(ops, total_ms, rows)
print(stats.bottleneck, stats.bottleneck_summary(), stats.spilled)
```

`explain(analyze=True)` gives, per operator: `est≈N actual=M (Kx)`, elapsed ms and its
share of wall time, `cpu=%`, output bytes, spill, and the backend (`interp` or `jit`),
followed by the bottleneck line, a CPU-utilization line, and a `decisions:` block naming
what Kyber and Carbonite chose. `RunStats` exposes `ops` (`OpStat` with `rows_in`,
`rows_out`, `elapsed_ms`, `spilled`, `backend`, `est_rows`, `selectivity`, `est_error`),
`bottleneck`, and `spilled`.

Read three things before changing anything:

1. **Where the time is** — `bottleneck`. Optimize that op or nothing.
2. **Where the estimates are wrong** — a large `(Kx)` means Kyber planned on a bad
   cardinality, so the join order or build side may be wrong. That is a *planning*
   problem, not an execution one.
3. **CPU utilization** — the target is >90% of cores. Low CPU with high wall time means
   I/O-, launch-, or GPU-dispatch-bound; adding parallelism will not help.

Note `ds.profile()` is a **data-quality** column profiler, not a performance profiler.
Machine-readable form: `json.loads(ds.explain(analyze=True, format="json"))`.

## 2. The checklist, in order

### a. Projection and predicate pushdown
Select only the columns you need and filter as early as you can express it. Kyber pushes
both down, but it **cannot push through what it cannot see through**: a `map_batches` or
UDF boundary is opaque, so a filter written after one stays after it. Move filters and
column pruning *above* any UDF in your chain. Confirm in `explain()` that the filter sits
directly above the scan.

### b. Join order and build side
The smaller side should be the build side. Check the `decisions:` block for the
`[kyber/selection] join build side: ...` line and its `[exact|learned|default]`
provenance — `default` means Kyber guessed. Kyber reorders joins by DP up to
`optimizer.join_dp_max_tables` (12) and greedily up to `greedy_max_tables` (25); past 25
there is no reordering at all, so hand-order the joins yourself. Broadcast kicks in under
`optimizer.broadcast_max_bytes` (4 MiB).

### c. Do not `collect()` a large result
`collect()` materializes the whole table. If you are writing or reducing, stream:

```python
for batch in ds.iter_batches(batch_size=65_536):   # bounded memory
    consume(batch)
ds.write(...)                                       # streams; never materializes
```

Pushing an aggregate or `limit` into the plan beats collecting and post-processing in
Python — Python must never touch rows in the hot path.

### d. Cache a reused result
`ds.cache()` marks *this* result for a process-wide, memory-bounded LRU keyed by plan and
inputs; the first terminal op computes, later equivalent ones do not. Bounded by
`memory.result_cache_max_bytes` (256 MiB). Caveats worth knowing: it does not survive the
process, a further transform is a new uncached result, and it covers **single-node
relational results only** — a `map_batches`/ML or distributed pipeline is silently not
cached. For cross-process reuse, write Parquet instead.

### e. File layout and scan pushdown
The cheapest row is the one never read. Partition on your common filter columns so
partition pruning fires, prefer Parquet with useful row-group statistics, and avoid many
tiny files (per-file overhead) and single huge unsplittable ones.
`execution.split_bytes` (128 MiB) is the file-split target.

### f. Spill only when you need it
Spill is **off by default** and costs I/O. Turn it on when a stateful op would otherwise
OOM, not preemptively:

```python
out = ds.collect(spill=True)
```

If `stats.spilled` is true and memory is available, raise the budget instead —
`memory.max_memory_bytes` (auto-sensed by default), effective budget
`max_memory_bytes x memory.hard_limit` (0.90). `memory.spill_compression` defaults to
`"auto"`; `"lz4"` trades CPU for I/O, `"zstd"` the reverse.

### g. Batch size and parallelism
Defaults are good; change them only against a measurement.

```python
from batcher.config import Config, ExecutionConfig, config_context
with config_context(Config().replace(execution=ExecutionConfig(morsel_rows=65_536, parallelism=16))):
    out = ds.collect()
```

`execution.morsel_rows` 16,384 (the batch size), `morsel_bytes` 1 MiB,
`parallelism` 0 = all cores. `adaptive_morsel_sizing` (True) already retunes batch size at
runtime via the PID controller — turning it off is rarely right. Env-var form is
`BATCHER_<SECTION>_<FIELD>`, e.g. `BATCHER_EXECUTION_MORSEL_ROWS=65536`.

### h. Adaptive re-optimization
Stage-boundary re-planning on *measured* cardinalities — the moat, and the fix when
`explain(analyze=True)` shows a large `(Kx)` estimate error feeding a join.

```python
out = ds.collect(adaptive=True)     # "auto" (default) | True | False
```

Under `adaptive="auto"` it engages only when a join has a breaker-produced operand whose
size is merely *guessed* **and** total input rows clear an internal 20,000,000-row gate
(`_ADAPTIVE_MIN_INPUT_ROWS` in `python/batcher/api/adaptive.py` — a private constant, not
a config field; it is not settable, so do not look for a knob). Below that, one-shot
planning is already fast and re-planning is pure overhead (~20–40 ms of control plane per
stage). `adaptive=True` bypasses the gate entirely — the right thing to try on a
mid-sized query with badly mis-estimated joins. The related tunable that *is* real is
`optimizer.reoptimize_error` (2.0), the estimate-error factor that triggers a re-plan.

### i. Distribution — last, not first
Distribution adds shuffle, serialization and scheduling. It pays only when the work is
genuinely too large for one node, not because a query feels slow.

```python
out = ds.collect(distributed=True, num_workers=8, num_partitions=64)
```

`distributed` defaults to `"auto"`; `distributed.distribute_min_rows` is 1,000,000. Prove
the result is unchanged (`assert_tables_equal` against `distributed=False`) before you
believe the timing. See `run-a-distributed-job`.

## 3. Benchmark honestly

```bash
just bench                     # TPC-H vs DuckDB/Polars   (python benchmarks/run.py --benchmark tpch)
just bench-ops                 # operator mix             (--benchmark operators)
just bench-tpch                # alias of `just bench`
just bench-dist                # single-node == many-partition equivalence + timing
just bench-list                # every registered benchmark
just bench --scale 10 --engines batcher,duckdb,polars
```

The harness runs every engine once, checks correctness **before** reporting, and picks the
oracle from `duckdb, polars, spark, daft, pyarrow` in that order — Batcher is never the
reference, because the system under test cannot grade itself. A mismatch marks the row
`FAILED` and `run.py` exits non-zero; a `FAILED` timing is not a result, whatever number
it prints. Timings are best-of-N after a warm-up.

Rules: compare against DuckDB/Polars, not only your own last commit; a worsened ratio is a
**blocking failure**, not a note; and never report a speedup on a path whose correctness
you have not just proven.

## 4. Signs the bottleneck is the engine, not your query

- The plan is already minimal — pushdown fired, join order is right, estimates are
  accurate (`(1.0x)`) — and it is still slow.
- CPU utilization is high but throughput is poor on a simple shape: an operator's inner
  loop is the cost.
- `backend` reads `interp` on a hot expression the JIT should have compiled — the JIT fell
  back. It must fall back *silently and correctly*, but a fallback on a supported shape is
  a gap worth closing.
- A shape where Batcher already loses to DuckDB/Polars in `just bench-ops`.
- Time sits in shuffle or spill rather than compute.

Then it is engine work: `add-relational-operator`, `add-expression-or-function`,
`add-kyber-optimizer-pass`, `add-distributed-operator` — each with the differential test
and benchmark those skills require, verified through `/run-quality-gate`.

## See also

- `.claude/rules/performance.md` — the competitive mandate and how we win
- `docs/tutorials/optimizing-a-slow-query.md`, `docs/user-guide/performance.md`
- `docs/user-guide/explain-plans.md`, `docs/user-guide/caching.md`, `docs/user-guide/best-practices.md`
- `docs/internals/kyber.md`, `docs/architecture/optimization.md`
- `debug-a-batcher-query` — when the answer is wrong, not merely slow
