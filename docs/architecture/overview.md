# Architecture overview

This page describes how Batcher splits into a control plane and a data plane, and what
each one is responsible for.

Batcher splits in two. Python is the control plane. It builds a query plan, optimizes
it, and decides how much it should cost, but it never touches a row of data. Rust is
the data plane, where every per-row and per-batch computation runs over Apache Arrow.
The two meet at one boundary, a JSON plan plus zero-copy Arrow batches, and nothing
else crosses it. The optimizer stays easy-to-change Python. The hot path runs at native
speed.

The API is lazy. An operation doesn't compute anything. It returns a new plan, and work
begins only at a terminal call such as `collect`. By then the optimizer sees the whole
computation at once, which is the precondition for optimizing a query as a whole and for
revising the plan mid-query.

## The two planes

![Batcher's two planes: a Python control plane (Dataset/SQL, Kyber, Carbonite, Core) handing a JSON IR plus Arrow batches to the Rust data plane (bc-py, bc-interp, bc-runtime, bc-codegen, bc-sketches, bc-transport).](/_static/diagrams/two_planes.svg)

The Python side carries the plan as JSON IR across the FFI boundary in `bc-py`, and the
data comes back as Arrow `RecordBatch`es with no copy and no serialization. Only
`bc-py` links Python. Every other crate is pure Rust and builds without an interpreter.

The Rust crates form a directed acyclic graph whose edges point one way only. `bc-arrow`
sits at the bottom and feeds `bc-expr`, which is the single scalar expression type.
`bc-expr` in turn feeds two independent branches: `bc-ir` (the single relational plan
type) leading to `bc-runtime`, and `bc-codegen` (the Cranelift JIT, which compiles
scalar expressions and so has no dependency on `bc-ir`). Both branches converge on
`bc-interp`, the interpreter and its parallel and distributed drivers. `bc-py` caps the
graph, but it isn't a thin cap on a single chain: it depends directly on `bc-sketches`,
`bc-transport`, `bc-io`, and `bc-resource` as well, which makes it a second assembly
point.

## The four control-plane subsystems

The Python control plane is four independent subsystems plus a neutral contract layer.
They don't import one another, and only the conductor wires them together.

Kyber is the optimizer. It rewrites plans and chooses physical strategies such as join
order, build side, and what to prune, using cardinality and cost, and it never makes
execution happen. Carbonite is the resource manager: it checks whether a plan fits,
hands out memory reservations and shuffle credits, and decides when to spill, without
ever rewriting a plan or computing a result. Core is the executor. It drives the engine
through `bc-py`, runs the adaptive re-optimization loop, and records what actually
happened, meaning real row counts, operator times, and peak memory. Governance applies
row filters and column masks as a pure plan rewrite, alongside column-level lineage.

`plan` is the neutral layer they all share, holding the logical and physical plan nodes,
the expression IR, and the JSON wire format, and it depends on none of them. `api` is
the only conductor, and the only place that imports every subsystem. The verbs stay in
their lanes: Core measures, Kyber decides, Carbonite protects. The separation keeps the
feedback loop below stable, because each side has exactly one job.

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

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](/_static/diagrams/lifecycle.svg)

1. **Build.** Each operation returns a new {py:class}`Dataset <batcher.Dataset>` wrapping a `LogicalPlan`.
   Nothing executes, and the plan accumulates.
1. **Optimize.** On `collect`, Kyber rewrites the logical plan with predicate and
   projection pushdown, join reordering, and fusion, then lowers it to a physical plan
   tagged with estimated resource bounds.
1. **Admit.** Carbonite checks the plan's largest materializing operator against the
   memory envelope. If memory is what doesn't fit, the verdict is a spill-friendly
   counter-offer and the query runs out of core. Any other binding constraint has no
   spill remedy, so the query fails with a `PlanError` before it starts.
1. **Execute.** Core ships the physical plan as JSON IR to the Rust engine, which
   runs it over Arrow batches. Pipelines stream through filters and projections, and
   breakers materialize for joins, aggregates, and sorts.
1. **Adapt.** On a query large enough to qualify, the engine runs stage by stage. At each
   stage boundary it has *measured* the real data size, and when an estimate was badly
   wrong, Kyber re-plans the rest of the query on the measured numbers before continuing.
1. **Return.** Results come back as a PyArrow `Table` from `collect`, a Python dict
   from {py:meth}`to_pydict <batcher.Dataset.to_pydict>`, a stream of batches from {py:meth}`iter_batches <batcher.Dataset.iter_batches>`, or are written to files.

Step 5 is the feedback loop that separates Batcher from a static optimizer. DuckDB
optimizes once, before it runs. Batcher's loop is stage-boundary re-optimization, the
same granularity as Spark AQE, and it runs single-node as well as distributed. Single-node,
it engages on a joined query that clears a floor of 5 million rows or about 320 MB per
breaker it would cut at, so most small queries take the one-shot plan. On top of it sits
a cross-query loop of sketches, calibrated costs and a bandit, so a plan improves the more
a query runs, whatever its size. Both mechanisms are described in
{doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`.

## One algebra, single node to cluster

Stateful operators live in `bc-runtime` as mergeable primitives: `partial`, `combine`,
and `finalize`, with `combine` associative and commutative so partials merge in any
order. The same implementation runs sequentially on one core, in parallel across many
by morselizing and merging, and across machines, where the distributed path composes
the identical primitives over Ray workers. The rows, the column names and the column
types come back the same whether the query ran on a laptop or a cluster. There is no
second distributed code path with its own semantics to diverge from.

![Mergeable algebra: each partition computes a partial state, an associative combine merges them in any order, and finalize produces the result. The same code runs on one core or many machines.](/_static/diagrams/mergeable.svg)

## Distribution

Ray is an optional dependency used for task and actor scheduling and control-plane
metadata only. Single-node execution never loads it. On a cluster, each worker hosts
the same in-process Rust engine, and bulk Arrow batches move between workers over
Arrow Flight (`bc-transport`) with credit-based backpressure. Those batches never pass
through the Ray object store, which is where the serialization overhead and OOM risk of
an object-store shuffle would otherwise come from. Only small control-plane strings
transit Ray. The single-node out-of-core machinery, radix-partition and spill, is the
same machinery that becomes the distributed shuffle, so disk and network are two sinks
for one mechanism.

## See also

- {doc}`Execution model <execution>`: pipelines, breakers, and the execution tiers.
- {doc}`Query optimization <optimization>`: Kyber's passes and cost model.
- {doc}`Fault tolerance <fault-tolerance>`: retries, recompute, and backpressure.
- {doc}`Execution engine </architecture/internals/execution>`: morsels and the tiers in detail.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the passes and the re-optimization loop.
- {doc}`Carbonite </architecture/internals/carbonite>`: the memory envelope and flow control.
- {doc}`What makes Batcher different <differentiators>`: the design decisions this layout supports.
- {doc}`Deep dives </architecture/deep-dives/index>`: one mechanism per page, starting with the query lifecycle.
