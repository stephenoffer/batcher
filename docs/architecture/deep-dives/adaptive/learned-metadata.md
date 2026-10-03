# Learned metadata

This page describes what Batcher measures on every run, where it keeps it, and how the optimizer, the resource manager and the scheduler read it back on the next run.

A query that has run once isn't the same query as one that has never run. The engine knows how many rows that join really produced, how much memory that aggregate really peaked at, and which build side really won. The `MetadataHub` is where that is kept, and the rule around it is one sentence: **Core measures, Kyber decides, Carbonite protects.** They meet in the hub.

Run a query twice and the estimates change provenance:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 50 for i in range(8000)], "x": [float(i) for i in range(8000)]})
q = ds.filter(bt.col("x") > 500).group_by("g").agg(s=bt.sum("x"))
print(q.explain())  # filter est≈2,667 (default)
for _ in range(2):
    q.collect()
print(q.explain())  # filter est≈... (learned), from the measured 7,499 rows
```

The first plan guesses a range selectivity of one third. After two runs the estimate is tagged `learned`, folded from the 7,499 rows the filter actually kept.

![The Kyber-Carbonite-Core feedback loop: Kyber decides and emits a plan with estimated cost, Carbonite protects by granting allocations, Core executes, and measured cardinalities and peak memory flow back to Kyber.](/_static/diagrams/carbonite_loop.svg)

```text
              ┌───────────────────────────────────────────────────────────┐
              │                      MetadataHub                          │
              │   op_stats                     learned_params             │
              │   (pruned at 65,536 rows)      (keyed by plan signature)  │
              └───▲───────────────────────────────────┬──────────────────┘
                  │  hub.record(feedback)             │  read
                  │  WRITE ONLY                       │  READ ONLY
                  │                                   │
        ┌─────────┴──────┐   ┌────────────────────────┴────┬─────────────┬────────────┐
        │      CORE      │   │           KYBER             │  CARBONITE  │    DIST    │
        │    measures    │   │          decides            │   protects  │  schedules │
        ├────────────────┤   ├─────────────────────────────┼─────────────┼────────────┤
        │  ExecMetrics   │   │  q-error correction         │ bytes per   │ partition  │
        │  from Rust,    │   │  cost coefficients          │ input row,  │ counts     │
        │  per op_id     │   │  the join bandit (UCB1)     │ per family  │ actor pool │
        │                │   │  broadcast/sort-merge       │ credit      │ hot keys   │
        │  never reads   │   │  crossovers                 │ window      │            │
        └────────────────┘   └─────────────────────────────┴─────────────┴────────────┘

   None of these four subsystems can import another. The hub is where they meet.
```

A join's chosen strategy and its wall time are recorded by the conductor in `api/tuning/decisions.py`, which folds the outcome into the bandit through `kyber/learned_tuning/`. The conductor is the layer allowed to see both the plan and the run, so no Kyber pass ever observes an execution.

![The loop that outlives one query. In run N, Kyber plans on whatever it knows now, Core executes it and measures rows, times and column sketches, and writes them to the MetadataHub, keyed by plan signature and, for anything in machine units, by hardware fingerprint. In run N plus 1, which is a later query in another process, minutes or days on, Kyber reads the hub before planning and plans on measured numbers, and Core measures and records again. What travels through the hub: measured cardinalities, operator wall times, column sketches, fitted cost coefficients and bandit arm rewards. Core writes after every run; Kyber reads before every plan, and never the other way round. The horizontal axis is the difference worth claiming: this is the same stage-boundary mechanism Spark AQE uses, but AQE keeps nothing once the query finishes, so it re-learns the same shape every time.](/_static/diagrams/cross_run_learning.svg)

## What Core measures

`bc-interp` returns per-operator metrics alongside the result batches from `execute_plan_metered`. {py:meth}`Dataset.stats() <batcher.Dataset.stats>` shows them:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 50 for i in range(8000)], "x": [float(i) for i in range(8000)]})
print(ds.filter(bt.col("x") > 500).group_by("g").agg(s=bt.sum("x")).stats())
```

```text
OP  KIND       ROWS IN  ROWS OUT   TIME  OP SHARE           OUT  BACKEND
────────────────────────────────────────────────────────────────────────
 0  aggregate    7,499        50  116µs  ████▊░  78%      800 B  interp
 1  filter       8,000     7,499   32µs  █▍░░░░  22%  117.2 KiB  interp
 2  scan         8,000     8,000    1µs  ░░░░░░  <1%  125.0 KiB  interp
────────────────────────────────────────────────────────────────────────
```

Timings move run to run. The row counts and `BACKEND`, the execution tier each operator ran on, do not. `BACKEND` is what `jit_speedup` calibration fits against. A query this small runs on the sequential executor, which never calls the JIT. See {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`.

:::{dropdown} The full per-operator record, and how it is matched to estimates
```rust
// crates/bc-interp/src/metrics.rs
pub struct OpMetric {
    pub op_id: u32,          // pre-order DFS index, matching kyber.annotate
    pub kind: &'static str,
    pub rows_in: u64,        // probe side only, for a join
    pub rows_build: u64,
    pub rows_out: u64,
    pub elapsed_ns: u64,
    pub wall_span_ns: u64,
    pub cpu_ns: u64,
    pub threads: u32,
    pub peak_bytes: u64,     // input held + result being built
    pub result_bytes: u64,
    pub spilled: bool,
    pub spill_bytes: u64,
    pub peak_rss_bytes: u64,
    pub backend: &'static str, // "interp" | "jit" | "interp+jit"
    pub hw: HwCounters,
}
```

`core/executor.py::_record_op_feedback` transcribes each into an `OperatorFeedback` and calls `hub.record(feedback)`. That is all of Core's involvement, and it never reads anything back.

- `n_estimated` is the rows Kyber predicted *before* any learned correction, so pairing it with `n_actual` measures the structural estimator's own error.
- `expr_factor` is the per-row cost of the operator's expressions, which calibration divides back out.
- Operators between a runtime join filter and its join are listed as `runtime_filtered` and recorded with `n_estimated = 0`, because their counts depend on the plan's filters rather than on the data.
- `op_id` is a pre-order index on both sides: Kyber's `annotate_ops` and the Rust executor number the same tree the same way, so a measured `rows_out` matches the estimate that predicted it.
:::

## The hub

[`python/batcher/metadata/hub.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/metadata/hub.py) holds two logical tables, `op_stats` and `learned_params`, behind a small API.

:::{dropdown} The whole `MetadataHub` surface
```python
# docs: skip
hub.record(feedback)  # the FeedbackSink: Core's only entry point
hub.version  # monotonic counter; the cache-invalidation signal
hub.op_stats_by_kind()  # bucketed by operator kind, for cost calibration
hub.op_stats_with_signature()  # oldest-first, for the q-error correction
hub.load_keyed_params(namespace)  # per-key learned scalars
hub.get_keyed_param(namespace, key)
hub.put_keyed_param(namespace, key, value)
```

`put_keyed_param` writes per key, so two writers learning about different query shapes don't clobber each other. Two writers folding into the *same* key at the same moment can lose one observation, since there is no compare-and-swap. The store stays consistent either way.

The derived views are maintained incrementally and bounded at 4,096 rows each, and the `op_stats` table beneath them is pruned to 65,536 rows on backends that support deletes.
:::

## Signatures

Learned values are keyed by plan shape, not query text: `kyber/signature.py::plan_signature` is a 16-hex-character SHA-1 of the normalized plan structure.

- A range bound's literal is normalized away, so `x > 5` and `x > 6` share a signature and a dashboard's daily query accumulates evidence.
- An equality, a membership list, and a string pattern keep their values, because on a skewed column `= '[ru]'` and `= '[us]'` select very different fractions.
- A `Scan` signs as its source's identity, and column statistics are keyed by `source\x1fcolumn`, so one table's `id` never answers for another's.

A 64-bit collision across a million shapes has odds of about one in 37 million, and would cost estimate quality, never a result.

## Backends

`MetadataBackend` is a four-method Protocol (`get`, `put`, `scan`, `batch_put`) with six implementations:

| Backend | Storage | Use |
|---|---|---|
| `in_process` | nested dicts | **the default**; learns within a session, forgets on exit |
| `sqlite` | one `kv` table, commit per put | carry learning across restarts |
| `rocksdb` | one embedded LSM tree, `table\x00key` per entry | the same, under a heavy write rate |
| `redis` | one hash per table | share learning across drivers |
| `object_storage` | one fsspec object per key | shared, durable, slow |
| `layered` | in-process cache over a durable store | the practical shared setup |

Cross-run learning is one config line. `backend="sqlite"` with no `uri` persists to `$BATCHER_HOME`, defaulting to `~/.batcher/metadata.db`:

```python
# docs: skip
import batcher as bt

durable = bt.Config().replace(metadata=bt.MetadataConfig(backend="sqlite", require_durable=True))
bt.set_config(durable)
```

The hub degrades to `in_process` with a warning if the configured backend can't be built, so a misconfigured Redis costs learning, not your query. `require_durable=True` turns that into a `ConfigError`. RocksDB locks its directory, so share across drivers with `redis` or `object_storage`. A `layered` store sees other drivers' learning after `refresh()`, which nothing calls on a schedule.

## What is learned

Four families, across three subsystems, all reading the same hub.

- **Kyber, cardinality and cost.** The per-signature q-error correction, per-column NDV, quantiles, most-common-values and average byte width, and recalibrated cost coefficients. See {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`.
- **Kyber, physical strategy** (`kyber/learned_tuning/`). A UCB1 bandit over `("hash", "broadcast", "sort_merge")`, plus OLS crossover fits for `broadcast_max_bytes` and the sort-merge threshold, learned build sides, and a verdict on whether partial pre-aggregation pays.
- **Carbonite, memory and flow control.** `LearnedMemoryModel` fits bytes per input row per operator family from measured peaks, which feeds admission, `should_spill`, spill partitioning and compression, and the morsel row cap. The converged shuffle credit window is persisted per channel, so a recurring shuffle skips slow-start.
- **Dist, scheduling.** Partition rows, actor-pool size, per-task CPU weight, shuffle fan-out, the straggler-speculation factor, and join hot keys.

The bandit's reward is **milliseconds per million input rows**, not wall time, because the same signature may run over 1M rows today and 50M tomorrow. Selection is a deterministic lower confidence bound scaled by the arm's measured spread, so a plan is reproducible.

![The learned-tuning bandit: its arms, its reward, and the bound on exploration. Two arm sets are learned: join strategy over hash, broadcast and sort_merge, and execution route over one_shot and staged. The reward is a measured latency in milliseconds, and the bandit minimizes it. Every arm emits the same relation, so a wrong pick costs throughput and never correctness, which is what makes exploring safe at all. Selection runs three ways. Under three observations the bandit returns nothing and the cost model decides. An arm never tried is given exactly one turn. Otherwise the lowest bound wins, computed as the mean minus 1.0 times the arm's own spread times the square root of 2 ln N over n, with ties broken by arm name and no RNG anywhere, so a plan is reproducible. Evidence decays at 0.975 per observation, so an arm that got faster is asked again rather than frozen out by a confidence radius that has shrunk to nothing. Statistics are kept per plan signature, and anything in machine units is additionally scoped by hardware fingerprint, so unlike machines never blend.](/_static/diagrams/bandit_tuning.svg)

:::{important}
Every learned value changes *how* a query runs, never *what* it returns. That is a tested property, since a tuned run must equal an untuned one, and it's what makes it safe to learn aggressively. A cold hub returns `None` from every reader and the first run is byte-for-byte the pre-learning path.
:::

## Practical limits

- The default `in_process` backend forgets on exit. Configure a durable backend for learning to carry across processes.
- Nothing expires. Recency comes from smoothing: a per-signature EWMA with step `max(learned_scalar_alpha_floor, 1/(n_obs+1))` and an 8-sample window on cardinality corrections.
- The join bandit learns only from single-join plans, where the whole query's wall time is attributable to that join.
- Distributed workers' feedback carries no signature, so it feeds per-kind cost calibration but not cardinality correction.
- `learned_broadcast_max_bytes` trains only on the distributed path.

:::{dropdown} Code map
| Concern | File |
|---|---|
| The hub | [`python/batcher/metadata/hub.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/metadata/hub.py) |
| Backends | [`python/batcher/metadata/backends/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/metadata/backends) |
| Metric transcription (Core) | [`python/batcher/core/executor.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/executor.py) |
| Per-operator metrics (Rust) | [`crates/bc-interp/src/metrics.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/metrics.rs) |
| Plan/column signatures | `python/batcher/kyber/{signature,learning}.py` |
| The join bandit and OLS crossovers | [`python/batcher/kyber/learned_tuning/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/kyber/learned_tuning) |
| Learned memory model | [`python/batcher/carbonite/memory/learned.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/carbonite/memory/learned.py) |
| Learned distributed sizing | [`python/batcher/dist/adaptive_sizing/sizing.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/adaptive_sizing/sizing.py) |
:::

## See also

- {doc}`Architecture </architecture/index>`: the contract loop, and why the subsystems meet only here.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the biggest reader.
- {doc}`Carbonite </architecture/internals/carbonite>`: the second-biggest.
- {doc}`Configuration options </configuration/options>`: the `metadata.*` backend settings.
- {doc}`Adaptive execution </getting-started/concepts/adaptive>`: what a user sees from this.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: cold versus warm, measured.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: the biggest consumer.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: coefficient calibration.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: the within-query half of the loop.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: what the learned memory model sizes.
