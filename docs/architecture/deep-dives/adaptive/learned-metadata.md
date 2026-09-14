# Learned metadata

A query that has run once is not the same query as one that has never run. The engine
knows how many rows that join really produced, how much memory that aggregate really
peaked at, which build side really won, how fast that source really reads. A static
optimizer throws all of it away and re-derives the same wrong estimates next time.

The `MetadataHub` is where it is kept. The rule around it is one sentence, and every
subsystem respects it: **Core measures, Kyber decides, Carbonite protects.** Core never
optimizes. Kyber never executes. Carbonite never rewrites a plan. They meet in the hub.

That division is the same contract loop the whole engine runs on:

![The Kyber-Carbonite-Core feedback loop: Kyber decides and emits a plan with estimated cost, Carbonite protects by granting allocations, Core executes, and measured cardinalities and peak memory flow back to Kyber.](/_static/diagrams/carbonite_loop.svg)

The hub is what the return arrow is made of. The rest of this page is what travels along it.

```text
              ┌───────────────────────────────────────────────────────────┐
              │                      MetadataHub                          │
              │   op_stats                     learned_params             │
              │   (bounded at 4,096 rows)      (keyed by plan signature)  │
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

One refinement the diagram flattens. Core is the only subsystem that reports *operator* metrics,
and Kyber only ever reads them. A join's chosen strategy and its wall time are recorded
separately, by the conductor in `api/tuning/decisions.py`, which calls into
`kyber/learned_tuning/` to fold the outcome into the bandit. The conductor is the layer allowed
to see both the plan it chose and the run that followed, so no Kyber pass has to observe an
execution to close that loop.

That loop turns once per query. The hub's other axis runs across runs, and it is the one with no DuckDB or Spark equivalent:

![The loop that outlives one query. In run N, Kyber plans on whatever it knows now, Core executes it and measures rows, times and column sketches, and writes them to the MetadataHub, keyed by plan signature and, for anything in machine units, by hardware fingerprint. In run N plus 1, which is a later query in another process, minutes or days on, Kyber reads the hub before planning and plans on measured numbers, and Core measures and records again. What travels through the hub: measured cardinalities, operator wall times, column sketches, fitted cost coefficients and bandit arm rewards. Core writes after every run; Kyber reads before every plan, and never the other way round. The horizontal axis is the difference worth claiming: this is the same stage-boundary mechanism Spark AQE uses, but AQE keeps nothing once the query finishes, so it re-learns the same shape every time.](/_static/diagrams/cross_run_learning.svg)

## What Core measures

`bc-interp` returns metrics alongside the result batches from `execute_plan_metered`. Per
operator (`crates/bc-interp/src/metrics.rs`):

```rust
pub struct OpMetric {
    pub op_id: u32,          // pre-order DFS index, matching kyber.annotate
    pub kind: &'static str,
    pub rows_in: u64,        // probe side only, for a join
    pub rows_build: u64,
    pub rows_out: u64,
    pub elapsed_ns: u64,
    pub cpu_ns: u64,
    pub threads: u32,
    pub peak_bytes: u64,     // input held + result being built
    pub result_bytes: u64,
    pub spilled: bool,
    pub spill_bytes: u64,
    pub peak_rss_bytes: u64,
    pub backend: &'static str, // "interp" | "jit" | "interp+jit"
}
```

`core/executor.py::_record_op_feedback` transcribes each into an `OperatorFeedback`, carrying
`n_actual`, `n_input`, `t_op_ms`, `m_peak_bytes`, `selectivity`, `signature`, `n_estimated`,
and `expr_factor`, then calls `hub.record(feedback)`. That is the whole of Core's involvement.
It does not read anything back.

Two of those fields carry the loop's precision. `n_estimated` is the rows Kyber predicted
*before* applying any learned correction, so pairing it with `n_actual` measures the structural
estimator's own error. Reporting the already-corrected estimate would make a converged
correction look error-free and decay it back to 1.0. `expr_factor` is the per-row cost of the
expressions the operator evaluated, which calibration divides back out so a fitted coefficient
describes the engine rather than the workload's expressions.

The `op_id` correspondence is what makes the loop close: Kyber's `annotate_ops` numbers the
plan in pre-order and stamps each node's estimate and signature onto it; the Rust executor
numbers the same tree the same way; so a measured `rows_out` can be matched to the estimate
that predicted it.

You can see the whole record for a query:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 50 for i in range(8000)], "x": [float(i) for i in range(8000)]})
print(ds.filter(bt.col("x") > 500).group_by("g").agg(s=bt.sum("x")).stats())
```

```text
OP  KIND       ROWS IN  ROWS OUT   TIME  OP SHARE          OUT  BACKEND
───────────────────────────────────────────────────────────────────────
 0  aggregate    7,499        50  434µs  ████▎░  71%      800 B  interp
 1  filter       8,000     7,499  180µs  █▉░░░░  29%  117.2 KiB  interp
 2  scan         8,000     8,000    1µs  ░░░░░░  <1%  125.0 KiB  interp
───────────────────────────────────────────────────────────────────────
```

The timings move run to run; the row counts and the `BACKEND` column do not. That column is
the execution tier each operator ran on, one of `interp`, `jit`, or `interp+jit` for an
operator that fell back part way, and it is what `jit_speedup` calibration fits against. It
reads `interp` throughout here because a query this small runs on the sequential executor,
which never calls the JIT at all. See
{doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`.

## The hub

`python/batcher/metadata/hub.py`. Two logical tables (`op_stats` and `learned_params`) and a
deliberately small API.

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

`put_keyed_param` rather than a whole-blob write, because two concurrent writers learning
about *different* query shapes must not clobber each other.

The derived views (`_by_kind`, `_signed`) are maintained incrementally and bounded at 4,096
rows each, so the backend is scanned exactly once per view per process.
:::

## Signatures

Learned values are keyed by plan shape, not by query text.

```python
# docs: skip
# python/batcher/kyber/signature.py
def plan_signature(node) -> str:
    payload = json.dumps(_struct(node), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
```

`_struct` normalizes literals away: every `{"e": "lit"}` becomes a bare `{"e": "lit"}`, so
`x > 5` and `x > 6` share a signature. That is the point: you want a dashboard's daily
query to accumulate evidence, not to start cold every time the date literal moves.

Column statistics are keyed differently, and the difference is a bug that was caught:
`plan_signature` degenerates a `Scan` to the bare token `["scan"]` with no source identity.
So column stats are qualified by source instead:

```python
# docs: skip
def qualify(source_key: str, column: str) -> str:
    return f"{source_key}\x1f{column}"
```

Without it, one table's `id` column answered for another's.

## Backends

`MetadataBackend` is a four-method Protocol (`get`, `put`, `scan`, `batch_put`) with six
implementations:

| Backend | Storage | Use |
|---|---|---|
| `in_process` | nested dicts | **the default**; learns within a session, forgets on exit |
| `sqlite` | one `kv` table, commit per put | carry learning across restarts |
| `rocksdb` | one embedded LSM tree, `table\x00key` per entry | the same, under a heavy write rate |
| `redis` | one hash per table | share learning across drivers |
| `object_storage` | one fsspec object per key | shared, durable, slow |
| `layered` | in-process cache over a durable store | the practical shared setup |

`backend="sqlite"` with no `uri` persists to `$BATCHER_HOME`, defaulting to
`~/.batcher/metadata.db`, so cross-run learning is one line with no path to manage.

`rocksdb` and `sqlite` answer the same question on different storage engines, and the write
rate is what separates them. Core records feedback after every query and Kyber reads it back
before every plan, so a busy single node presents this store with a sustained stream of small
writes. SQLite answers each one with a B-tree update and a journal write; RocksDB appends to a
memtable and merges later, which is what a log-structured merge tree is for. The encoding is
identical either way, because `encode_key` is prefix-preserving and orders the way a byte
comparison does, so a prefix `scan` is a seek and a walk on both. RocksDB locks its directory,
so only one process may hold it open; use `redis` or `object_storage` to share across drivers.
`LayeredBackend` writes durable-first then caches, and
reads cache-first with fall-through. Its `refresh()` drops the cache entirely, which is the
cross-driver freshness hook.

The hub is a process-wide singleton built by `core/runtime.py::default_hub()`, and it
**degrades to `in_process` on any construction failure** with a warning. A misconfigured
Redis costs you learning, not your query.

## What is learned

Four families, across three subsystems, all reading the same hub.

**Kyber: cardinality and cost.** The per-signature q-error correction
(`__cardinality_correction__`), plus per-column NDV, quantiles, most-common-values, and
average byte width. Cost coefficients are recalibrated from measured operator times. See
{doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`.

**Kyber: physical strategy** (`kyber/learned_tuning/`). A UCB1 bandit over the three
equivalent join algorithms:

```python
# docs: skip
_JOIN_ARMS = ("hash", "broadcast", "sort_merge")
```

The reward is the subtle part. It is **milliseconds per million input rows**, not wall time:

```python
# docs: skip
mrows = max(0.0, input_rows) / 1e6
reward = wall_ms / mrows if mrows > 0.0 else wall_ms
```

Raw wall time is non-stationary (the same signature runs over 1M rows today and 50M
tomorrow), so it would permanently condemn whichever arm happened to draw the large input.
Selection is a deterministic lower-confidence-bound (the bandit *minimizes*), with the
exploration radius scaled by the pooled standard deviation recovered from the stored
`sumsq`. Textbook UCB1 assumes rewards in [0,1]; a bare radius against a 500 ms mean is a
0.2% nudge and collapses to greedy.

The bound on exploration is the half of a bandit that usually goes undrawn, and it is what keeps a cold signature from paying for the search:

![The learned-tuning bandit: its arms, its reward, and the bound on exploration. Two arm sets are learned: join strategy over hash, broadcast and sort_merge, and execution route over one_shot and staged. The reward is a measured latency in milliseconds, and the bandit minimizes it. Every arm emits the same relation, so a wrong pick costs throughput and never correctness, which is what makes exploring safe at all. Selection runs three ways. Under three observations the bandit returns nothing and the cost model decides. An arm never tried is given exactly one turn. Otherwise the lowest bound wins, computed as the mean minus 1.0 times the arm's own spread times the square root of 2 ln N over n, with ties broken by arm name and no RNG anywhere, so a plan is reproducible. Evidence decays at 0.975 per observation, so an arm that got faster is asked again rather than frozen out by a confidence radius that has shrunk to nothing. Statistics are kept per plan signature, and anything in machine units is additionally scoped by hardware fingerprint, so unlike machines never blend.](/_static/diagrams/bandit_tuning.svg)

Alongside it sit OLS two-line crossover fits for `broadcast_max_bytes` and the sort-merge
row threshold, learned build sides, and a learned verdict on whether partial
pre-aggregation pays off.

**Carbonite: memory and flow control.** The `LearnedMemoryModel` fits a per-operator-family
**bytes-per-input-row** from measured `m_peak_bytes`. A ratio, not an absolute peak, so it is
size-general: a 1M-row aggregate and a 10-row one share one coefficient. Every sizing
decision then blends the plan's byte estimate toward the measured figure, clamped so a noisy
sample cannot wildly move sizing. It feeds admission, `should_spill`, spill partition count,
spill compression, and the morsel row cap. And the converged shuffle credit window is
persisted per channel signature, so a recurring shuffle skips slow-start.

**Dist: scheduling.** Learned partition rows, actor-pool size, per-task CPU weight, shuffle
fan-out, straggler-speculation factor, and join hot keys.

## Everything here is result-invariant

:::{important}
Every learned value changes *how* a query runs: how much memory it reserves, when it spills, how
big a morsel is, which of three equivalent join algorithms it picks, how many workers it fans
out to. None of them changes *what* it returns. That is not a convention but a tested property,
since a tuned run must equal an untuned one, and it is what makes it safe to learn
aggressively. The worst a stale learned value can do is cost you throughput.
:::

::::{tab-set}
:::{tab-item} A cold hub
Every reader returns `None` when it has no evidence, and every caller then uses its static
default. A first run is byte-for-byte the pre-learning path.
:::

:::{tab-item} A warm hub
Corrections, coefficients, bandit arms, memory ratios, and partition counts all have values, and
the plan is different. The *result* is not.
:::
::::

## Limits worth knowing

Start with the one that governs everything else: the default backend is `in_process`, so a
fresh interpreter starts cold and everything on this page is learned and then discarded within
a single session. Cross-run learning is real but opt-in, and until you configure a durable
backend the warm numbers quoted here describe a repeated query inside one process rather than
the behavior of a new one.

:::{warning}
**Nothing expires.** `metadata.decay_per_day` is declared and validated and has no reader. There
is no TTL and no aging on any backend. What provides recency is smoothing, not expiry: a
per-signature EWMA with step `max(learned_scalar_alpha_floor, 1/(n_obs+1))`, a running mean while
evidence is thin and then a roughly 10-observation memory, plus the 8-sample window on
cardinality corrections.
:::

Three more, each a real hole rather than a rough edge:

| Limit | Consequence |
|---|---|
| The join bandit only learns from single-join plans (`record_join_outcomes` bails above one join, because whole-query wall time must be unambiguously attributable to *that* join) | a TPC-H query with five joins contributes nothing to the bandit |
| Distributed workers' feedback carries no signature, because a worker's `op_id`s address its own sub-plan and cannot be correlated with the driver's tree | those rows feed per-kind cost calibration but are excluded from cardinality correction |
| `learned_broadcast_max_bytes` only trains on the distributed path | on a single-node deployment it returns `None` forever and the static threshold never moves |

## Code map

Each stage of the learning loop lives in one file. Read them in this order to follow a
measurement from recording to reuse:

| Concern | File |
|---|---|
| The hub | `python/batcher/metadata/hub.py` |
| Backends | `python/batcher/metadata/backends/` |
| Metric transcription (Core) | `python/batcher/core/executor.py` |
| Per-operator metrics (Rust) | `crates/bc-interp/src/metrics.rs` |
| Plan/column signatures | `python/batcher/kyber/{signature,learning}.py` |
| The join bandit and OLS crossovers | `python/batcher/kyber/learned_tuning/` |
| Learned memory model | `python/batcher/carbonite/memory/learned.py` |
| Learned distributed sizing | `python/batcher/dist/adaptive_sizing/sizing.py` |

## See also

- {doc}`Architecture </architecture/index>`: the contract loop, and why the subsystems meet only here.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the biggest reader.
- {doc}`Carbonite </architecture/internals/carbonite>`: the second-biggest.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): UCB1, the shrinkage estimator, the EWMA.
- {doc}`Configuration options </configuration/options>`: the `metadata.*` backend settings.
- {doc}`Adaptive execution </getting-started/concepts/adaptive>`: what a user actually sees from this.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: cold versus warm, measured.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: the biggest consumer.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: coefficient calibration.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: the within-query half of the loop.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: what the learned memory model sizes.
