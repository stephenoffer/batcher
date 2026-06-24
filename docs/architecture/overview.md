# Architecture overview

Batcher splits in two. Python is the control plane: it builds a query plan,
optimizes it, and decides how much it should cost — but it never touches a row of
data. Rust is the data plane: every per-row and per-batch computation runs there,
over Apache Arrow. The two meet at one boundary, a JSON plan plus zero-copy Arrow
batches, and nothing else crosses it. That single split is what lets the optimizer
be written in clean, malleable Python while the hot path runs at native speed.

The API is lazy. An operation does not compute anything; it returns a new plan. Work
begins only at a terminal call like `collect`, and by then the optimizer can see the
whole computation at once — which is what makes whole-query optimization, and
re-optimization mid-query, possible.

## The two planes

![Batcher's two planes: a Python control plane (Dataset/SQL, Kyber, Carbonite, Core) handing a JSON IR plus Arrow batches to the Rust data plane (bc-py, bc-interp, bc-runtime, bc-codegen, bc-sketches, bc-transport).](../_static/diagrams/two_planes.png)

The Python side carries the plan as JSON IR across the FFI boundary in `bc-py`; the
data comes back as Arrow `RecordBatch`es with no copy and no serialization. The Rust
crates form a one-way dependency chain, and only `bc-py` links Python — every other
crate is pure Rust.

## The three control-plane subsystems

The Python control plane is three independent subsystems plus a neutral contract
layer. They do not import one another; only the conductor wires them together.

- **Kyber decides.** The optimizer rewrites plans and chooses physical strategies —
  join order, build side, what to prune — using cardinality and cost. It never makes
  execution happen.
- **Carbonite protects.** The resource manager checks whether a plan fits, hands out
  memory reservations and shuffle credits, and decides when to spill. It never
  rewrites a plan or computes a result.
- **Core measures.** The executor drives the engine through `bc-py`, runs the
  adaptive re-optimization loop, and records what actually happened — real row
  counts, operator times, peak memory.

`plan` is the neutral layer they all share (the logical and physical plan nodes, the
expression IR, the JSON wire format); it depends on none of them. `api` is the only
conductor, and the only place that imports all three. The verbs stay in their lanes:
Core measures, Kyber decides, Carbonite protects. Keeping them separate is what makes
the feedback loop below stable — each side has exactly one job.

## How a query runs

```python
import batcher as bt

ds = bt.read("events.parquet")
result = (
    ds.filter(bt.col("status") == "active")
    .group_by("region")
    .agg(total=bt.col("amount").sum())
    .collect()
)
```

1. **Build.** Each operation returns a new `Dataset` wrapping a `LogicalPlan`.
   Nothing executes; the plan accumulates.
2. **Optimize.** On `collect`, Kyber rewrites the logical plan — predicate and
   projection pushdown, join reordering, fusion — and lowers it to a physical plan
   tagged with estimated resource bounds.
3. **Admit.** Carbonite checks the plan against the memory envelope. If it does not
   fit, it returns a counter-offer (lower parallelism, a smaller credit window) that
   Kyber re-plans around.
4. **Execute.** Core ships the physical plan as JSON IR to the Rust engine, which
   runs it over Arrow batches: pipelines stream through filters and projections,
   breakers materialize for joins, aggregates, and sorts.
5. **Adapt.** At each breaker the engine has *measured* the real data size. When an
   estimate was badly wrong, Kyber re-plans the rest of the query on the measured
   numbers before continuing.
6. **Return.** Results come back as a PyArrow `Table` (`collect`), a Python dict
   (`to_pydict`), a stream of batches (`iter_batches`), or are written to files.

Step 5 is the part static optimizers cannot do. DuckDB optimizes once before it runs;
Spark AQE adapts only at stage boundaries; Batcher re-optimizes at every breaker, on
numbers it measured rather than guessed.

## One algebra, single node to cluster

Stateful operators live in `bc-runtime` as mergeable primitives — `partial`,
`combine`, `finalize`, with `combine` associative and commutative. The same
implementation runs sequentially on one core, in parallel across many (morselize and
merge), and across machines (the distributed path composes the identical primitives
over Ray workers). A result is the same whether it ran on a laptop or a cluster,
because there is no second distributed code path with its own semantics.

## Distribution

Ray is an optional dependency, used for task and actor scheduling and control-plane
metadata only. Single-node execution never loads it. On a cluster, each worker hosts
the same in-process Rust engine, and bulk Arrow batches move between workers over
Arrow Flight (`bc-transport`) with credit-based backpressure — they do not pass
through the Ray object store, which is where the serialization overhead and OOM risk
of object-store shuffles would otherwise come from. The single-node out-of-core
machinery (radix-partition and spill) is the same machinery that becomes the
distributed shuffle; disk and network are just two sinks for it.

## Where to read next

- [Execution engine](../internals/execution.md) — pipelines, morsels, and the tiers
- [Kyber optimizer](../internals/kyber.md) — the passes and the re-optimization loop
- [Carbonite](../internals/carbonite.md) — the memory envelope and flow control
