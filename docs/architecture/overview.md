# Architecture overview

This page describes how Batcher splits into a control plane and a data plane, and what each one is responsible for.

Python is the control plane. It builds a query plan, optimizes it and decides what it should cost, but it never touches a row. Rust is the data plane: every per-row and per-batch computation runs there, over Apache Arrow. The two meet at one boundary, a JSON plan plus zero-copy Arrow batches. Nothing else crosses it.

The optimizer stays easy-to-change Python. The hot path runs at native speed.

## The two planes

![Batcher's two planes: a Python control plane (Dataset/SQL, Kyber, Carbonite, Core) handing a JSON IR plus Arrow batches to the Rust data plane (bc-py, bc-interp, bc-runtime, bc-codegen, bc-sketches, bc-transport).](/_static/diagrams/two_planes.svg)

The plan crosses the FFI boundary in `bc-py` as JSON, and the data comes back as Arrow `RecordBatch`es with no copy and no serialization. Only `bc-py` links Python. Every other crate is pure Rust and builds without an interpreter.

The crates form a graph whose edges point one way. `bc-arrow` sits at the bottom and feeds `bc-expr`, the single scalar expression type. From there two branches split off: `bc-ir`, the single relational plan type, leading to `bc-runtime`; and `bc-codegen`, the Cranelift JIT, which compiles scalar expressions and so never needs `bc-ir`. Both converge on `bc-interp`, the interpreter with its parallel and distributed drivers. `bc-py` sits on top and also depends directly on `bc-runtime`, `bc-sketches`, `bc-transport`, `bc-io` and `bc-resource`, which makes it a second assembly point rather than a thin cap.

## The four control-plane subsystems

Four subsystems share the control plane, and none of them imports another.

Kyber is the optimizer. It rewrites plans and picks physical strategies such as join order, build side and what to prune, and it never makes execution happen. Carbonite manages resources: it checks whether a plan fits, hands out memory reservations and shuffle credits, and decides when to spill, without ever rewriting a plan. Core executes. It drives the engine through `bc-py`, runs the adaptive loop, and records what actually happened, meaning real row counts, operator times and peak memory. Governance applies row filters and column masks as a pure plan rewrite and tracks column-level lineage.

`plan` is the neutral layer all four share. It holds the plan nodes, the expression IR and the JSON wire format, and depends on none of them. `api` is the only conductor, the one place that imports every subsystem. Core measures, Kyber decides, Carbonite protects, and each side keeps exactly one job.

## How a query runs

The API is lazy, so each operation returns a new plan and work begins only at a terminal call such as `collect`. By then the optimizer sees the whole computation:

```python
# docs: run
import batcher as bt

ds = bt.from_pydict({
    "region": ["eu", "us", "eu", "us"],
    "status": ["active", "active", "closed", "active"],
    "amount": [10, 20, 30, 40],
})
query = ds.filter(bt.col("status") == "active").group_by("region").agg(total=bt.col("amount").sum())
print(query.sort("region").collect().to_pydict())
# {'region': ['eu', 'us'], 'total': [10, 60]}
```

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](/_static/diagrams/lifecycle.svg)

1. Build. Each operation returns a new {py:class}`Dataset <batcher.Dataset>` wrapping a `LogicalPlan`. Nothing executes.
1. Optimize. On `collect`, Kyber applies predicate and projection pushdown, join reordering and fusion, then lowers the plan to a physical one tagged with estimated resource bounds.
1. Admit. Carbonite checks the largest materializing operator against the memory envelope. When memory is the problem, the answer is a spill-friendly counter-offer and the query runs out of core. Any other binding constraint fails the query with a `PlanError` before it starts.
1. Execute. Core ships the plan as JSON IR to the Rust engine. Filters and projections stream through, and the pipeline breakers materialize.
1. Adapt. On a query large enough to qualify, the engine runs stage by stage. At each boundary it has *measured* the real data size, and when an estimate was badly wrong, Kyber re-plans the rest of the query.
1. Return. Results come back as a PyArrow `Table` from `collect`, a dict from {py:meth}`to_pydict <batcher.Dataset.to_pydict>`, a stream from {py:meth}`iter_batches <batcher.Dataset.iter_batches>`, or files.

You can watch steps 2 and 4 from the outside. `explain()` prints the optimized plan, and `explain(analyze=True)` runs it and adds measured row counts:

```python
# docs: run
assert "pushed[status = active]" in query.explain()
assert "actual=" in query.explain(analyze=True)
```

Step 5 is what a static optimizer lacks. DuckDB optimizes once, before it runs. Batcher re-optimizes at stage boundaries, the same granularity as Spark AQE, inside the Python process and on a single node as well as a cluster. Single-node, it engages on a joined query that clears 5 million rows or about 320 MB per breaker it would cut at, so most small queries take the one-shot plan. Above it sits a cross-query loop of sketches, calibrated costs and a bandit, so a plan improves the more a query runs, whatever its size. {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` covers both.

## One algebra, single node to cluster

Stateful operators live in `bc-runtime` as three mergeable primitives: `partial`, `combine` and `finalize`. `combine` is associative and commutative, so partials merge in any order. The same code runs on one core, across many cores by morselizing and merging, and across machines over Ray workers. Rows, column names and column types come back the same on a laptop and on a cluster, because there is no second distributed code path to diverge.

![Mergeable algebra: each partition computes a partial state, an associative combine merges them in any order, and finalize produces the result. The same code runs on one core or many machines.](/_static/diagrams/mergeable.svg)

```python
# docs: skip
# On a Ray cluster: the same plan, four workers, the same rows.
query.collect(distributed=True, num_workers=4)
```

## Distribution

Ray is optional. It schedules tasks and actors and carries control-plane metadata, and single-node execution never loads it. On a cluster every worker hosts the same in-process Rust engine, and bulk Arrow batches travel between workers over Arrow Flight (`bc-transport`) with credit-based backpressure. They never pass through the Ray object store, where an object-store shuffle would pay serialization and risk OOM. Only small control-plane strings transit Ray.

The single-node out-of-core machinery, radix-partition and spill, is the same machinery that becomes the distributed shuffle. Disk and network are two sinks for one mechanism.

## See also

Each page below takes one piece of this layout further.

- {doc}`Execution model <execution>`: pipelines, breakers, and the execution tiers.
- {doc}`Query optimization <optimization>`: Kyber's passes and cost model.
- {doc}`Fault tolerance <fault-tolerance>`: retries, recompute, and backpressure.
- {doc}`Execution engine </architecture/internals/execution>`: morsels and the tiers in detail.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the passes and the re-optimization loop.
- {doc}`Carbonite </architecture/internals/carbonite>`: the memory envelope and flow control.
- {doc}`What makes Batcher different <differentiators>`: the design decisions this layout supports.
- {doc}`Deep dives </architecture/deep-dives/index>`: one mechanism per page, starting with the query lifecycle.
