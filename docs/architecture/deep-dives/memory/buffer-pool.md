# The buffer pool

The *buffer pool* is the process-wide account of how many bytes the engine has outstanding. This page describes how Batcher reserves against it, how pressure is read from it, and how concurrent queries share it.

Two operators that each decide independently they have room will together exceed the machine. One shared counter is the answer. Batcher's is `MemoryPool`, in [`crates/bc-resource/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-resource/src/lib.rs), deliberately the smallest crate at the bottom of the DAG (`std` plus `thiserror`), so `bc-runtime` and `bc-transport` draw on the same envelope without depending on each other. The design is DataFusion's greedy `MemoryPool` with RAII reservations plus Spark's cooperative-spilling `MemoryConsumer` model.

## Reserve before you allocate

:::{important}
A caller reserves bytes *before* it allocates them, and a reservation that would push the pool past its limit fails. An operator that allocates first and reserves afterwards has already put the process over the line by the time the pool hears about it.
:::

```rust
// crates/bc-resource/src/lib.rs
pub fn try_reserve_bytes(&self, bytes: usize) -> ResourceResult<()>
pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> ResourceResult<MemoryReservation>
pub fn release_bytes(&self, bytes: usize)
```

`try_reserve_bytes` is a compare-and-swap loop on an `AtomicUsize` that returns `ResourceError::Exhausted { requested, available, limit }` without mutating on failure. `release_bytes` clamps at zero. `MemoryReservation` is the RAII handle, with `try_grow()`, `shrink()`, `free()` and a `Drop` that releases whatever remains, so an operator that panics doesn't leak its budget. The pool itself is policy-free: it accounts and admits, and every decision about a refusal lives above it.

## Two pools, read side by side

One `MemoryPool` type has two live instances:

- The **engine pool** is created inside `execute_plan`, sized from `EngineConfig.memory_budget_bytes`. Operator state reserves against it, and on any real query this is where the bytes are.
- The **control-plane pool** is created by Carbonite from its memory envelope and carries the coarse per-query reservation, so concurrent queries admit against one budget.

They are deliberately separate counters. Carbonite reserves a plan's *estimated* peak and the engine then reserves the operator's *actual* bytes, so one account would double-count every query. Both are reported in each query's Carbonite resource decision:

```python
import json
import batcher as bt

ds = bt.from_pydict({"g": [i % 100 for i in range(5000)], "x": [1.0] * 5000})
report = json.loads(ds.group_by("g").agg(n=bt.count()).explain(analyze=True, format="json"))
detail = next(d for d in report["decisions"] if d["category"] == "resources")["detail"]
print(sorted(k for k in detail if k.endswith("pool")))  # ['engine_pool', 'pool']
print(detail["memory_share"], detail["engine_pool"]["over_released_bytes"])  # 1.0 0
```

Reading the pair is a diagnosis. A query that spilled with Carbonite's pool nearly empty and the engine's at its limit was bound by an estimate that was too low, not by the box. The reverse means the estimate was too high.

## Pressure

`used / limit` is coarsened into levels, and this one signal feeds every backpressure mechanism: the plan-time decision to go out of core, morsel admission, and the shuffle credit window.

```text
   memory.max_memory_bytes: auto-sensed once at the terminal op, cgroup-aware,
                             then frozen for the query
   ┌────────────────────────────────────────────────────────────────────┐  100%
   ├─ memory.hard_limit   0.90  ───────────────────────────────────────►│  CRITICAL
   │     a new reservation succeeds only after something spills         │
   ├─ memory.soft_limit   0.85  ───────────────────────────────────────►│  SPILL
   │     new plans go out of core; AIMD reads its congestion signal here│
   ├─ soft_limit × 0.9    0.765 ───────────────────────────────────────►│  ELEVATED
   │     trim the result cache; narrow the in-flight window             │
   │     NORMAL: no throttling                                          │
   └────────────────────────────────────────────────────────────────────┘  0%

   the fraction being classified is  max( pool.used / pool.limit ,
                                          process_footprint / total )
```

```python
import batcher as bt

mem = bt.Config().memory
print(mem.soft_limit, mem.hard_limit)  # 0.85 0.9
```

The classified fraction is the **maximum** of the control-plane pool's `used / limit`, the engine pool's, and `process_footprint / total`, where the footprint prefers the cgroup's `memory.current` over RSS. A pyarrow buffer or a UDF's tensors are real memory the pool never hears about, and the footprint floor is what stops the monitor reporting NORMAL while the kernel OOM-kills you.

:::{dropdown} The Rust levels and the Python monitor
The Rust pool reports `Pressure { Nominal, Elevated, Critical }`. `Critical` is `used >= limit`; `Elevated` is `used >= limit * soft_bps / 10_000`, with `DEFAULT_SOFT_BPS` of 8000 on an unconfigured pool. On every query `execute_plan` moves that line to `min(memory.soft_limit, memory.hard_limit)`, so the pool's `Elevated` begins at the same byte as the monitor's `SPILL`. The level reaches the control plane as the `pressure` field of `engine_pool_stats()`; nothing in the data plane acts on it.

The finer ladder lives in `carbonite/memory/pressure.py`:

| `PressureLevel` | Trigger (fraction of budget) | Default |
|---|---|---|
| `NORMAL` | below everything | |
| `ELEVATED` | `soft_limit * 0.9` | 0.765 |
| `SPILL` | `memory.soft_limit` | 0.85 |
| `CRITICAL` | `memory.hard_limit` | 0.90 |

`PressureMonitor.level()` classifies on `max(raw, previous_ewma)`, so pressure escalates instantly and de-escalates as the EWMA relaxes, which keeps the morsel size and credit window from flapping. Readers that must not advance the EWMA, such as morsel sizing and the cache trim, call `classify()` instead.
:::

## Cooperative spilling

A plain refusal means "you can't have this memory", which is unhelpful when a *different* operator is sitting on the budget and could spill. `try_reserve_cooperative` asks registered consumers to give some back:

```rust
pub trait Spillable: Send + Sync {
    fn spill(&self, target: usize) -> usize;   // bytes actually freed
    fn spillable_bytes(&self) -> usize;        // orders the victims
}
```

On a failed reservation the pool computes the shortfall, asks live consumers to spill largest-first outside the registry lock, and retries. A pass that frees nothing ends the loop. With no registered consumers this is exactly `try_reserve`.

The registered consumer is `ShuffleSpiller` in [`crates/bc-py/src/flight.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/flight.rs): the published shuffle store, finished output waiting to be collected, which costs one re-read to spill. So a distributed worker gets cooperative spilling, and on a single node a breaker that cannot reserve spills its own state.

![The Rust buffer pool as a single gauge with one soft line, drawn at memory.soft_limit of the limit on every query and at 80% on a pool nobody configured. Below that line the pool is nominal and nothing throttles; above it the pool reports elevated to the control plane, and nothing in the data plane acts on that report. Critical is used == limit rather than a band, because growth past the limit is refused outright. The value moves right as operators reserve and back as they release: the pool counts bytes, it never allocates them. A try_reserve(n) that still fits under the limit is granted as an RAII guard, and every byte returns when the guard drops, on a panic as much as on a clean finish. One that does not fit is refused with the denial counted and used untouched, the pool then asks the largest other registered consumer to spill and re-reserves, for at most 32 rounds and stopping the moment a round frees nothing. A caller still short after that spills itself, the refusal being the signal. Carbonite's pool and the engine's pool count different bytes and are read side by side, never summed.](/_static/diagrams/buffer_pool_zones.svg)

## Where the limit comes from

`memory.max_memory_bytes` is `None` by default, and `api` auto-senses it once at the terminal op from host RAM, honoring a cgroup limit, then freezes it for the query. The budget shipped to Rust is `cap * memory.hard_limit`. Pin a smaller envelope by deriving a config:

```python
import dataclasses
import batcher as bt
from batcher import Config

base = Config()
cfg = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=512 << 20))
print(cfg.spill_budget_bytes())  # 483183820

with bt.config_context(cfg):
    ds = bt.from_pydict({"g": [i % 1000 for i in range(10_000)], "x": [1.0] * 10_000})
    print(ds.group_by("g").agg(s=bt.sum("x")).collect().num_rows)  # 1000
```

That's `512 MiB * 0.90`, and any stateful operator whose estimate exceeds it goes out of core instead of OOMing. A budget of `0`, which `memory.unbounded_memory = True` asks for, attaches no engine pool and no spill path: the query neither spills nor fails early.

:::{dropdown} What is frozen and what is live
The cap is sensed per terminal op, so the next query picks up a new `max_memory_bytes`. What it is sensed *from* (host RAM, cgroup caps, the scheduler grant, `RLIMIT_AS`) is memoized for the process, so a container resized in place keeps the old ceiling until restart. The pressure monitor re-samples the cgroup's usage, PSI stall share and OOM history on short TTLs, and a `SPILL` reading sends the next plan out of core however large the frozen cap is. The engine's process-wide pool keeps the largest budget any query has shipped, so a smaller budget cannot strand a concurrent query's granted reservation. In a container the monitor reads cgroup v2 `memory.max`, falling back to v1 `memory.limit_in_bytes`; where it can't, set `max_memory_bytes` yourself.
:::

## What the kernel says

The pool answers how much the engine reserved. Three cgroup v2 signals answer whether the kernel is coping, each read only from the container's own slice:

- **`memory.high`.** Kubernetes derives it from a pod's *request*. Past it the kernel throttles allocations in direct reclaim rather than failing them. With `memory.respect_cgroup_high` on, the lower of `memory.high` and `memory.max` is the ceiling.
- **Pressure Stall Information.** The `full` stall share is a floor on the pressure level, capped at `SPILL`, so a thrashing cgroup spills before it is killed. A host-wide reading is ignored.
- **`memory.events`.** A recorded OOM kill scales a restarted worker's envelope by `memory.oom_kill_backoff`.

The resource decision carries all of it in a `kernel` block, absent on a host with no cgroups:

```python
print("was_oom_killed" in detail["kernel"])  # True
```

## Concurrent queries divide the envelope

A query decides whether to go out of core before it reserves anything, so several queries planning at the same moment would each see an empty pool. When `execution.max_concurrent_queries` is set, a query admitted while N are running plans against `1/N` of the envelope, Spark's `ExecutionMemoryPool` rule at query granularity. The share moves only the spill threshold; the pool keeps the whole envelope so no granted reservation becomes unaffordable. The default is `0`, unbounded concurrency, where the share is exactly 1:

```python
import batcher as bt

print(bt.Config().execution.max_concurrent_queries)  # 0
```

A process that serves queries side by side should set it; see {doc}`Hardening </user-guide/trust/hardening>`. `BudgetingAdmission` also subtracts what concurrent queries have already reserved, so the share covers only the window before reservations happen.

![How Carbonite sizes the envelope one query may plan against, and what it does when the plan will not fit inside it. A join's peak is the larger of its build subtree's peak and the resident build table plus its probe subtree's peak, because the build table stays resident while the probe runs; on one worked bushy plan the largest single operator reads 18.2 MB where that concurrent figure is 27.4 MB, a 1.5x under-count. The envelope starts at the process envelope, or total RAM when none is set, is multiplied by memory.soft_limit, has what concurrent work already holds subtracted, and is divided by the query's share of 1 / min(active, slots), which is exactly 1 when concurrency is unbounded. It is floored at one morsel, so a streaming plan is never refused for a budget smaller than a single batch. A plan whose peak fits is admitted with no bound imposed; one that does not is admitted anyway with m_max_bytes set to the envelope, so the binding join, aggregate or sort goes out of core. The verdict names that operator, and where it rests on a guess it is advisory: it routes, it never fails.](/_static/diagrams/memory_envelope.svg)

A nested query, such as a `collect()` inside a `map_batches` UDF, takes no admission slot. On a thread it reserves against the same engine pool; in a child process it has its own pools.

## Storage yields to execution

`ResourceManager.reserve` in `carbonite/manager.py` implements Spark's unified memory model. The result cache behind {py:meth}`Dataset.cache() <batcher.Dataset.cache>`, bounded by `memory.result_cache_max_bytes`, is *storage*; an operator building a hash table is *execution*, and execution wins. Before reserving, the manager trims the cache to three-quarters of its budget at `ELEVATED`, half at `SPILL` and nothing at `CRITICAL`, then evicts any remaining deficit lowest-value first. Under the default `MEMORY_AND_DISK` level an evicted result is demoted to the disk tier (`memory.result_cache_disk_max_bytes`) and promoted back on a later read. Evicting costs at most a recompute, so it never changes an answer.

## Practical limits

- **Contention.** The pool is one atomic counter, so reservations are per operator rather than per morsel.
- **Coverage.** Python-side pyarrow buffers and UDF tensors are invisible to `used`. The footprint floor covers them, and the `memory_ledger` block reports `unaccounted_bytes`, the resident bytes above the accounted figure.
- **Estimates.** The stateful breakers reserve once, at admission, and don't grow the reservation with their state. The learned memory model in {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>` corrects the estimate across runs.
- **Over-release.** A release past `used` is clamped and counted as `over_released_bytes`, which is `0` on a run whose reservations balance.

## See also

- {doc}`Architecture </architecture/index>`: Carbonite's lane, where it protects but never decides or executes.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager that drives the pool.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page). It is the v1-era design paper with an errata list at its top, and where it and the code differ the code decides. It covers the control theory behind the hysteresis.
- {doc}`Configuration options </configuration/options>`: every `memory.*` knob named here.
- {doc}`Performance </user-guide/operate/tuning/performance>`: setting an envelope on purpose.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what bounded memory buys under load.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what happens when a reservation cannot be granted.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: the same envelope, applied to the network.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: what the bytes being counted actually are.
