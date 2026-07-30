# Internals

How Batcher works under the hood: the layered architecture and the components that
turn your query into results. You don't need any of this to use Batcher. It is here
for contributors and the curious.

:::{seealso}
The {doc}`deep dives <../deep-dives/index>` cover the same engine one mechanism at a time,
with worked examples you can run: the {doc}`query lifecycle <../deep-dives/query-lifecycle>`,
{doc}`mergeable algebra <../deep-dives/mergeable-algebra>`,
{doc}`spilling <../deep-dives/spilling>`, and
{doc}`adaptive re-optimization <../deep-dives/adaptive-reoptimization>`.
:::

## Architecture overview

![Batcher's layered architecture from the User API down through the Dataset API, Logical Plan, Kyber optimizer, Physical Plan, Execution Engine, Carbonite, and optional Ray.](../_static/diagrams/layer_stack.svg)

Ray is an optional dependency used only for distributed scheduling. Single-node
execution does not require it, and even on a cluster the data plane moves Arrow
batches over Arrow Flight rather than through the Ray object store.

## Core components

Four subsystems do the work, and each owns one decision. The sections below take them in
the order a query meets them.

### Kyber optimizer

Kyber is Batcher's query optimization engine. It transforms logical plans into efficient physical plans through:

- **Phased rule pipeline** (normalize → pushdown → join-reorder → fusion →
  selection → enforce), with rules grouped by family in `kyber/rules/`
- **Cost-based physical choices**: join build-side swap and hash-vs-broadcast
  selection from sketch/learned cardinality
- **Learned cardinality** that sharpens across runs via the MetadataHub
- **Intra-query adaptive re-optimization** - re-plans at pipeline breakers on
  *measured* sizes, single-node and distributed. This is stage-boundary adaptation,
  the same granularity as Spark AQE. The difference is that Batcher does it on a
  single node too, where DuckDB's static optimizer has no equivalent, and pairs it
  with a cross-query learned-stats loop that neither has

{doc}`Learn more about Kyber <kyber>`

### Carbonite

Carbonite handles memory management and data movement:

- **Memory coordination** across the cluster
- **File-based caching** for intermediate results
- **Spill-to-disk** when memory is constrained
- **Shuffle optimization** for redistributing data
- **Backpressure control** for streaming execution

{doc}`Learn more about Carbonite <carbonite>`

### Execution engine

The execution engine runs the optimized plan in Rust over Arrow batches. It lowers
the plan into pipelines and breakers, schedules the work as 16K-row morsels, and
runs each pipeline through one of three paths that share operator semantics: the
sequential interpreter (the oracle), a rayon-parallel path, and a Cranelift JIT that
falls back to the interpreter on anything it does not support. The same mergeable
primitives run on one core, many cores, or many machines.

{doc}`Learn more about the execution engine <execution>`

## Data flow

A typical query flows through the system:

![The eight-step data flow from user code through logical plan, Kyber optimization, physical plan, execution engine, Carbonite, the Rust data plane, and collected results.](../_static/diagrams/data_flow.svg)

## Key concepts

Three ideas explain most of Batcher's behavior. Each one is why a query does something
that would surprise you in an eager engine.

### Lazy evaluation

Operations build a plan without executing:

```python
# No execution yet - just building plan
ds2 = ds.filter(col("x") > 10)
ds3 = ds2.select("a", "b")

# Execution happens here
result = ds3.collect()
```

### Streaming execution

Data flows through operators in chunks:

```python
# Data streams through pipeline
# Memory footprint stays constant
for batch in ds.iter_batches():
    process(batch)
```

### Adaptive optimization

Kyber learns from execution feedback:

1. Initial plan uses statistics-based estimates
2. Execution reports actual row counts, timing
3. Kyber updates models for future queries

## Performance

Where the speed comes from is worth separating from how fast it is. The first section is
the optimizer's contribution, and the second is what has been measured end to end.

### Optimization impact

The passes compound, so the effect of any one of them is not separable from the rest by
reasoning alone. Where a single planner change has been isolated and measured, the figure
is published with its run: a `filter(...).count()` that took 2,187 ms fell to 47 ms once
`.count()` lowered to a `COUNT(*)` aggregate and projection pushdown pruned the scan to
the predicate's column, a 46x improvement from one change. See
{doc}`../benchmarks/analytics`. No speedup range is quoted here that the harness has not
produced, because the harness refuses to time a query whose result does not match the
oracle.

### Scalability

Batcher's distributed path composes the *same* mergeable primitives
(`partial → combine → finalize`) the single-node path uses, so per-node memory
stays bounded and the shuffle is credit-controlled (data bypasses the Ray object
store via Arrow Flight). This is the design basis for near-linear scaling, but
**published multi-node throughput numbers are not yet measured** - distributed
execution is validated for *correctness* (single-node == multi-worker equivalence)
in CI, and large-cluster benchmarks are pending real multi-host runs. No GB/s-per-node
figure is quoted here until the benchmark harness produces one.

## In this section

```{toctree}
:maxdepth: 1

kyber
carbonite
execution
extending
testing-strategy
```

The formal treatment (cost models, sketch error bounds, control-theory stability
proofs) lives in `internals/mathematical_foundations.md`, which is rendered to PDF
by `internals/generate_pdf.py` rather than as a site page.

## See also

- {doc}`Architecture overview <../architecture/overview>`: high-level design
