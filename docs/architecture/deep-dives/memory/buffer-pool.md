# The buffer pool

The *buffer pool* is the process-wide account of how many bytes the engine has
outstanding. This page describes how Batcher reserves against it, how pressure is read from
it, and how concurrent queries share it.

Two operators, each estimating its own memory, each deciding independently that it has
room, will together exceed the machine. That's the whole problem, and one shared counter
is the answer to it.

Batcher's is `MemoryPool`, in `crates/bc-resource/src/lib.rs`. It is deliberately the
smallest crate at the bottom of the DAG (`std` plus `thiserror`, no Arrow, no IR), so
that `bc-runtime` and `bc-transport` can both draw on the same envelope without either
depending on the other. The design is DataFusion's `MemoryPool` / `MemoryReservation`
(a greedy pool with RAII reservations) plus Spark's cooperative-spilling
`MemoryConsumer` model, adopted rather than re-derived.

## Reserve before you allocate

:::{important}
A caller reserves bytes *before* it allocates them, and a reservation that would push the pool
past its limit fails. That's the whole contract, and it only works if everything of consequence
honors it. An operator that allocates first and reserves afterwards has already put the process
over the line by the time the pool hears about it.
:::

```rust
// crates/bc-resource/src/lib.rs
pub fn try_reserve_bytes(&self, bytes: usize) -> ResourceResult<()>
pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> ResourceResult<MemoryReservation>
pub fn release_bytes(&self, bytes: usize)
```

`try_reserve_bytes` is a compare-and-swap loop on an `AtomicUsize`, with a
`saturating_add`, and it returns `ResourceError::Exhausted { requested, available, limit }`
*without mutating* on failure. `release_bytes` clamps at zero, so a double release can't
underflow the counter into a pool that thinks it has 18 exabytes free.

`MemoryReservation` is the RAII handle. It offers `size()`, `try_grow()` which leaves the
reservation unchanged when it can't grow, `shrink()`, `free()`, and a `Drop` that releases
whatever remains. An operator that panics doesn't leak its budget.

The pool itself is policy-free. It accounts and it admits. Every decision about what to
*do* when a reservation fails lives above it.

## There are two pools, and they are different budgets

One `MemoryPool` type, two live instances, and knowing which one a number came from is
what makes the number mean anything.

The **engine pool** is created inside `execute_plan` and sized from
`EngineConfig.memory_budget_bytes`. Operator state reserves against it, and the Flight
shuffle store registers with it as a spillable consumer. On any real query this is where
the bytes are.

The **control-plane pool** is created by Carbonite and sized from its memory envelope. It
carries the coarse per-query reservation Carbonite takes for the duration of execution, so
concurrent queries admit against one budget.

They are deliberately not one counter. Carbonite reserves a plan's *estimated* peak and
the engine then reserves the same operator's *actual* bytes, so charging both to one
account would double-count every query and push it out of core at half its envelope.

What they do share is a reader. `engine_pool_stats()` in `carbonite/memory/pool.py` reads
the engine's pool, the pressure monitor classifies against whichever of the two is fuller,
and `ResourceManager.stats()` reports both side by side. Before that, the control plane
could only infer the engine's memory from process RSS, which lags a reservation by however
long the operator takes to fill the state it reserved.

Reading the pair is a diagnosis. A query that spilled with Carbonite's pool nearly empty
and the engine's at its limit was bound by an estimate that was too low, not by the box.
The reverse means the estimate was too high and the query spilled needlessly.

## Pressure

`used / limit` is coarsened into levels, and this is the one signal every backpressure
mechanism reads: proactive spill, the morsel-admission gate, and the shuffle credit window.
They can't invent disagreeing thresholds.

```text
   memory.max_memory_bytes: auto-sensed once at the terminal op, cgroup-aware,
                             then frozen for the query
   ┌────────────────────────────────────────────────────────────────────┐  100%
   │                                                                    │
   ├─ memory.hard_limit   0.90  ───────────────────────────────────────►│  CRITICAL
   │     a new reservation succeeds only after something spills         │
   │                                                                    │
   ├─ memory.soft_limit   0.85  ───────────────────────────────────────►│  SPILL
   │     spill proactively; AIMD reads its congestion signal here       │
   │                                                                    │
   ├─ soft_limit × 0.9    0.765 ───────────────────────────────────────►│  ELEVATED
   │     trim the result cache; narrow the in-flight window             │
   │                                                                    │
   │     NORMAL: no throttling                                          │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘  0%

   the fraction being classified is  max( pool.used / pool.limit ,
                                          process_footprint / total )
```

::::{tab-set}
:::{tab-item} The Rust pool
```rust
pub enum Pressure { Nominal, Elevated, Critical }
```

`Critical` is `used >= limit`. `Elevated` is `used >= limit * soft_bps / 10_000`, where
`soft_bps` is seeded to `DEFAULT_SOFT_BPS` of 8000, meaning 80%. `Elevated` exists so an
operator can spill *proactively*, before the hard cap forces a stall.

The pool exposes `set_soft_fraction` to move that line, but nothing outside the crate's own
tests calls it, so the Rust soft line sits at 80% and is independent of
`memory.soft_limit`. The finer Python ladder is the one that reads the configured limits.
:::

:::{tab-item} The Python monitor
The finer ladder lives in `carbonite/memory/pressure.py`:

| `PressureLevel` | Trigger (fraction of budget) | Default |
|---|---|---|
| `NORMAL` | below everything | |
| `ELEVATED` | `soft_limit * 0.9` | 0.765 |
| `SPILL` | `memory.soft_limit` | 0.85 |
| `CRITICAL` | `memory.hard_limit` | 0.90 |

`PressureMonitor.level()` samples with asymmetric hysteresis. It classifies on
`max(raw, previous_ewma)`, so pressure escalates instantly and de-escalates only as the EWMA
relaxes. A monitor that flapped between NORMAL and SPILL would flap the morsel size and the
credit window with it. Readers that must not advance the EWMA, such as morsel sizing and the
cache trim, call `PressureMonitor.classify()` instead. Exactly one component per round may call `level()`.
:::
::::

The fraction it classifies is the **maximum** over three readings: the control-plane
pool's `used / limit`, the engine pool's, and `process_footprint / total`, where the
footprint prefers the cgroup's `memory.current` over RSS. Both pools, because the engine's
is the one holding operator state, and reading only the control plane's classified a query
holding 90% of the engine's envelope as `NORMAL` until RSS caught up.

:::{warning}
A pyarrow buffer allocated on the Python side, or a UDF's tensors, is real memory the pool has never
heard of. Taking the maximum against the process footprint is what stops the monitor reporting
NORMAL while the kernel OOM-kills you.
:::

## Cooperative spilling

The interesting method is `try_reserve_cooperative`. A plain reservation failure means
"you can't have this memory", which is unhelpful when the reason is that a *different*
operator is sitting on the budget and could spill.

```rust
pub trait Spillable: Send + Sync {
    fn spill(&self, target: usize) -> usize;   // bytes actually freed
    fn spillable_bytes(&self) -> usize;        // orders the victims
}
```

Consumers register with `register_consumer`, held as `Weak` so a dead operator is swept
rather than leaked. On a failed reservation, the pool computes the shortfall, snapshots
the live consumers, sorts them largest-first, and asks each to spill *outside* the
registry lock, because `spill()` must not re-enter the pool. If a full pass frees nothing,
it breaks, which is the termination guarantee. Then it retries.

The requester is deliberately not registered yet, because it reserves before it builds
state, so every victim is a different operator or a concurrent query. That's the point. A
small aggregate no longer dies while a large neighboring join sits on the whole budget.

With no registered consumers this is exactly `try_reserve`, so nothing pays for machinery
it doesn't use.

One consumer registers today, and which one it is decides where the mechanism applies.
`ShuffleSpiller` in `crates/bc-py/src/flight.rs` puts the published shuffle store in the
registry when a Flight server binds, and keeps a pool reservation equal to the store's
resident bytes so spilling it hands real credit back. Published output is finished work
waiting to be collected, so writing it out stalls nobody and costs one re-read.

That means a distributed worker gets cooperative spilling and a **single-node** query does
not: no Flight server exists there, the registry is empty, and a breaker that cannot
reserve is always the one that spills. Closing that half needs a `Spillable` on the
operators that own in-progress state, which the pool may call from another thread while
the owning operator is reading it.

The pool's whole behavior fits in one picture: one soft line, one limit, and the two ways
a refused reservation can end.

![The Rust buffer pool as a single gauge with one soft line at 80% of the limit. Below that line the pool is nominal and nothing throttles; above it the pool is elevated and operators spill early. Critical is used == limit rather than a band, because growth past the limit is refused outright. The value moves right as operators reserve and back as they release: the pool counts bytes, it never allocates them. A try_reserve(n) that still fits under the limit is granted as an RAII guard, and every byte returns when the guard drops, on a panic as much as on a clean finish. One that does not fit is refused with the denial counted and used untouched, the pool then asks the largest other registered consumer to spill and re-reserves, for at most 32 rounds and stopping the moment a round frees nothing. A caller still short after that spills itself, the refusal being the signal. Carbonite's pool and the engine's pool count different bytes and are read side by side, never summed.](/_static/diagrams/buffer_pool_zones.svg)

## Where the limit comes from

`memory.max_memory_bytes` is `None` by default, and `api` auto-senses it once at the
terminal-op boundary from the live envelope (host RAM, honoring a cgroup limit), then
freezes it for the query. The data-plane budget shipped to Rust is
`cap * memory.hard_limit`, as `EngineConfig.memory_budget_bytes`.

A `memory_budget_bytes` of `0` means unbounded. `ExecOptions.agg_spill` stays `None` and
the engine runs fully in memory with zero spill machinery. Set
`memory.unbounded_memory = True` to ask for that explicitly. A query then fails fast rather
than spilling.

In a container the OS often reports the *host's* RAM rather than the cgroup limit, which
is why the pressure monitor reads cgroup v2 `memory.max` and falls back to v1
`memory.limit_in_bytes`. Where it can't, set `max_memory_bytes` yourself.

You can see the budget the engine actually ran under:

```python
import json
import batcher as bt

ds = bt.from_pydict({"g": [i % 100 for i in range(5000)], "x": [1.0] * 5000})
report = json.loads(ds.group_by("g").agg(n=bt.count()).explain(analyze=True, format="json"))
print("budget bytes:", report["memory_budget_bytes"])
print("peak rss:", report["peak_rss_bytes"])
print("spilled:", report["spilled"])
```

To pin a smaller envelope, derive a config:

```python
import dataclasses
import batcher as bt
from batcher import Config

base = Config()
cfg = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=512 << 20))

with bt.config_context(cfg):
    ds = bt.from_pydict({"g": [i % 1000 for i in range(10_000)], "x": [1.0] * 10_000})
    out = ds.group_by("g").agg(s=bt.sum("x")).collect()
    print(out.num_rows)
```

The data-plane budget under that context is `512 MiB * 0.90`, about 461 MiB, and any
stateful operator whose estimated footprint exceeds it goes out of core instead of OOMing.

## What the kernel says

The pool's accounting answers how much the engine reserved. It cannot answer whether the kernel is coping, and the two disagree in exactly the cases that kill a container. Three cgroup v2 signals close that gap, each read only from the container's own cgroup slice.

`memory.high` is the threshold Kubernetes memory QoS derives from a pod's *request*, while `memory.max` comes from its *limit*. Past `memory.high` the kernel does not fail an allocation. It puts every allocating task to sleep in direct reclaim. A query sized against `memory.max` then spends its whole life being throttled while every counter reports success. With `memory.respect_cgroup_high` on, the engine treats the lower of the two as the ceiling, and the pressure fraction is measured against it.

Pressure Stall Information answers the coping question directly. A cgroup can sit at 70% of its limit and still spend most of every second in reclaim, because the limit being defended is `memory.high`, or because its resident set is nearly all anonymous and there is no cache left to drop. Batcher reads the `full` share, meaning the fraction of the window in which every runnable task was stalled, and uses it only as a floor on the pressure level, capped at `SPILL`. A stall share is a rate rather than a headroom figure, so it must never be the thing that halts a query with gigabytes free. A host-wide reading is deliberately ignored: acting on it would make one container spill because a different one is thrashing.

`memory.events` records whether this cgroup has already been OOM-killed. That is evidence rather than a forecast, so a restarted worker scales its envelope by `memory.oom_kill_backoff` instead of re-deriving the number that got it killed, and an un-sized plan spills rather than repeating the kill.

Read what the kernel published through {py:meth}`Dataset.explain(analyze=True) <batcher.Dataset.explain>`, whose Carbonite resource decision carries a `kernel` block. The block is absent on a host with no cgroups, which is deliberately distinct from a block of zeros.

## Concurrent queries divide the envelope

The pool is one envelope for the whole process, so a second query's reservations are
visible to the first. What is *not* visible that way is a plan that has not reserved yet.
A query decides whether to go out of core by comparing its estimated peak against the
budget, and that comparison happens before any reservation. When several queries reach it
at the same moment, each one sees an empty pool, each concludes that a plan needing most of
RAM fits, and all of them take the in-memory path.

Batcher divides the budget the same way it divides the cores. When
`execution.max_concurrent_queries` is set, a query admitted while N are running plans
against `1/N` of the envelope, which is Apache Spark's `ExecutionMemoryPool` rule applied
at query granularity. A plan that fits on an idle machine goes out of core on a busy one,
which is the correct outcome: spilling costs a disk round trip, and the alternative costs
the process.

Three things bound the division:

- The share only moves the *spill threshold*. The pool keeps the whole envelope, because
  shrinking it would make a concurrent query's already-granted reservation retroactively
  unaffordable.
- A nested query, such as a `collect()` inside a `map_batches` UDF, takes no admission slot
  and so does not raise the occupancy. The outer query already paid for the machine.
- The default is unbounded concurrency (`max_concurrent_queries = 0`), where the share is
  exactly 1 and no budget changes.

The share is reactive as well as proactive. `BudgetingAdmission` subtracts what concurrent
queries have already reserved, and a reservation that does not fit routes the query out of
core. Those two see reservations that have happened; the share covers the window before
they do.

Admission applies those reductions in a fixed order, and what it holds the result against is
the plan's estimated peak rather than a count of live hash tables.

![How Carbonite sizes the envelope one query may plan against, and what it does when the plan will not fit inside it. A join's peak is the larger of its build subtree's peak and the resident build table plus its probe subtree's peak, because the build table stays resident while the probe runs; on one worked bushy plan the largest single operator reads 18.2 MB where that concurrent figure is 27.4 MB, a 1.5x under-count. The envelope starts at the process envelope, or total RAM when none is set, is multiplied by memory.soft_limit, has what concurrent work already holds subtracted, and is divided by the query's share of 1 / min(active, slots), which is exactly 1 when concurrency is unbounded. It is floored at one morsel, so a streaming plan is never refused for a budget smaller than a single batch. A plan whose peak fits is admitted with no bound imposed; one that does not is admitted anyway with m_max_bytes set to the envelope, so the binding join, aggregate or sort goes out of core. The verdict names that operator, and where it rests on a guess it is advisory: it routes, it never fails.](/_static/diagrams/memory_envelope.svg)

Read the division back from the Carbonite resource decision:

```python
import json
import batcher as bt

ds = bt.from_pydict({"g": [i % 100 for i in range(5000)], "x": [1.0] * 5000})
report = json.loads(ds.group_by("g").agg(n=bt.count()).explain(analyze=True, format="json"))
resources = [d for d in report["decisions"] if d["category"] == "resources"]
detail = resources[0]["detail"]
print("share:", detail["memory_share"])
print("this query's budget:", detail["hard_budget_bytes"])
print("queries admitted:", detail["admission"]["active"])
```

On an idle process the share is `1.0` and `hard_budget_bytes` is the whole envelope times
`memory.hard_limit`.

## Storage yields to execution

`ResourceManager.reserve` in `carbonite/manager.py` implements Spark's unified memory
model, in two steps. The result cache behind {py:meth}`Dataset.cache() <batcher.Dataset.cache>`, bounded by
`memory.result_cache_max_bytes`, is *storage*. An operator building a hash table is
*execution*. Execution wins.

Before reserving, the manager calls `CacheStore.on_pressure` to trim the cache against the current pressure level: to three-quarters of its budget at `ELEVATED`, half at `SPILL`, and nothing at `CRITICAL`. If a shortfall remains, it evicts exactly the deficit, lowest-value entries first. Only then does it reserve.

What the trim sheds isn't necessarily lost. Under the default `MEMORY_AND_DISK` storage level an evicted result is demoted to the cache's disk tier (`carbonite/cache_disk.py`, bounded by `memory.result_cache_disk_max_bytes`), which writes through the same tiered spill store the operators use, and a later read promotes it back. `CRITICAL` is the one rung that clears without demoting, because encoding hundreds of megabytes of Arrow is the wrong use of memory the process doesn't have. The trim reads `classify()` rather than `level()`, so it doesn't consume the AIMD round's sample. Evicting a cache entry costs at most a recompute, so none of this can change an answer.

## Costs and limits

The pool is a single atomic counter with a CAS loop, so it's cheap but it's a process-wide
contention point at high reservation rates. That's why reservations are per-*operator*,
not per-morsel: one reserve for a hash table build, not one per batch.

It only accounts what's routed through it. Arrow buffers allocated by pyarrow on the
Python side and a UDF's torch tensors are invisible to `used`. The Flight partition store is visible only where `ShuffleSpiller` registers it, which is a distributed worker with a bound Flight server. The pressure monitor's RSS and cgroup floor is the mitigation, and
it's a floor, not a ledger.

A reservation is also an *estimate* accepted in advance. The pool can't tell you that the
hash table you reserved 100 MB for will actually take 400. What corrects that is the
learned memory model described in {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`, which fits a
measured bytes-per-input-row figure per operator family from `m_peak_bytes` and blends the
plan's estimate toward it.

## See also

- {doc}`Architecture </architecture/index>`: Carbonite's lane, where it protects but never decides or executes.
- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager that drives the pool.
- `docs/architecture/internals/mathematical_foundations.md` (in the repo, not a site page): the control theory behind the hysteresis.
- {doc}`Configuration options </configuration/options>`: every `memory.*` knob named here.
- {doc}`Performance </user-guide/operate/tuning/performance>`: setting an envelope on purpose.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what bounded memory buys under load.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what happens when a reservation cannot be granted.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: the same envelope, applied to the network.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: what the bytes being counted actually are.
