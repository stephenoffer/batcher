# The buffer pool

The *buffer pool* is the single process-wide account of how many bytes the engine has
outstanding. Every allocation of consequence reserves against it before allocating. This
page describes the reservation contract, the pressure levels every backpressure mechanism
reads, cooperative spilling, and where the pool's limit comes from.

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
| `ELEVATED` | `soft_limit × 0.9` | 0.765 |
| `SPILL` | `memory.soft_limit` | 0.85 |
| `CRITICAL` | `memory.hard_limit` | 0.90 |

`PressureMonitor.level()` samples with asymmetric hysteresis. It classifies on
`max(raw, previous_ewma)`, so pressure escalates instantly and de-escalates only as the EWMA
relaxes. A monitor that flapped between NORMAL and SPILL would flap the morsel size and the
credit window with it. Readers that must not advance the EWMA, such as morsel sizing and the
cache trim, call `classify()` instead. Exactly one component per round may call `level()`.
:::
::::

The fraction it classifies is the **maximum** of `pool.used / pool.limit` and
`process_footprint / total`, where the footprint prefers the cgroup's `memory.current`
over RSS.

:::{warning}
The Flight `PartitionStore` and any off-pool pyarrow buffer are real memory the pool has never
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

## Where the limit comes from

`memory.max_memory_bytes` is `None` by default, and `api` auto-senses it once at the
terminal-op boundary from the live envelope (host RAM, honoring a cgroup limit), then
freezes it for the query. The data-plane budget shipped to Rust is
`cap × memory.hard_limit`, as `EngineConfig.memory_budget_bytes`.

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

The data-plane budget under that context is `512 MiB × 0.90`, about 461 MiB, and any
stateful operator whose estimated footprint exceeds it goes out of core instead of OOMing.

## Storage yields to execution

`ResourceManager.reserve` in `carbonite/manager.py` implements Spark's unified memory
model, in two steps. The result cache behind `Dataset.cache()`, bounded by
`memory.result_cache_max_bytes`, is *storage*. An operator building a hash table is
*execution*. Execution wins.

Before reserving, the manager calls `CacheStore.on_pressure` to trim the cache against the
current pressure level: three-quarters of the budget at `ELEVATED`, half at `SPILL`, and
everything at `CRITICAL`. If a shortfall remains, it evicts exactly the deficit,
lowest-value entries first. Only then does it reserve. Caching therefore can't grow the
process without bound, and it hands RAM back rather than pushing a query out to disk. The
trim reads `classify()` rather than `level()`, so it doesn't consume the AIMD round's
sample. Evicting a cache only costs a recompute, so none of this can change an answer.

## Costs and limits

The pool is a single atomic counter with a CAS loop, so it's cheap but it's a process-wide
contention point at high reservation rates. That's why reservations are per-*operator*,
not per-morsel: one reserve for a hash table build, not one per batch.

It only accounts what's routed through it. Arrow buffers allocated by pyarrow on the
Python side, the Flight in-memory partition store, and a UDF's torch tensors are all
invisible to `used`. The pressure monitor's RSS and cgroup floor is the mitigation, and
it's a floor, not a ledger.

A reservation is also an *estimate* accepted in advance. The pool can't tell you that the
hash table you reserved 100 MB for will actually take 400. What corrects that is the
learned memory model described in {doc}`Learned metadata </deep-dives/adaptive/learned-metadata>`, which fits a
measured bytes-per-input-row figure per operator family from `m_peak_bytes` and blends the
plan's estimate toward it.

## See also

- {doc}`Architecture </architecture/index>`: Carbonite's lane, where it protects but never decides or executes.
- {doc}`Carbonite </internals/carbonite>`: the resource manager that drives the pool.
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the control theory behind the hysteresis.
- {doc}`Configuration options </configuration/options>`: every `memory.*` knob named here.
- {doc}`Performance </user-guide/operate/performance>`: setting an envelope on purpose.
- {doc}`Scaling benchmarks </benchmarks/scaling>`: what bounded memory buys under load.
- {doc}`Spilling </deep-dives/memory/spilling>`: what happens when a reservation cannot be granted.
- {doc}`Credit-based flow control </deep-dives/distribution/credit-flow-control>`: the same envelope, applied to the network.
- {doc}`Arrow and memory </deep-dives/memory/arrow-memory>`: what the bytes being counted actually are.
