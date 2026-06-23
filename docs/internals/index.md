# Internals

This section provides deep technical documentation on Batcher's internal architecture and components.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User API Layer                                  │
│                                                                              │
│    bt.read()   bt.col()   ds.filter()   ds.join()   ds.collect()           │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Dataset API                                     │
│                                                                              │
│    Lazy operations that build logical plans                                  │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Logical Plan                                    │
│                                                                              │
│    Tree of logical operators: Scan, Filter, Project, Join, Aggregate        │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kyber Optimizer                                 │
│                                                                              │
│    Rule + cost-based passes (normalize→pushdown→reorder→selection)           │
│    Learned cardinality; intra-query adaptive re-optimization                 │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Physical Plan                                   │
│                                                                              │
│    Concrete operators: HashJoin (shuffle + broadcast), Aggregate, Sort,     │
│    Window - all mergeable (single-node == distributed)                       │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Execution Engine                                │
│                                                                              │
│    Plan compilation, task scheduling, progress tracking                      │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Carbonite                                       │
│                                                                              │
│    Memory management, caching, spill-to-disk, data transfer                  │
│                                                                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Ray (optional, distributed only)                       │
│                                                                              │
│    Task and actor scheduling plus control-plane metadata. Bulk Arrow batches │
│    bypass the Ray object store and move over Arrow Flight (bc-transport).    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Ray is an optional dependency used only for distributed scheduling. Single-node
execution does not require it, and even on a cluster the data plane moves Arrow
batches over Arrow Flight rather than through the Ray object store.

## Core components

### Kyber optimizer

Kyber is Batcher's query optimization engine. It transforms logical plans into efficient physical plans through:

- **Phased rule pipeline** (normalize → pushdown → join-reorder → fusion →
  selection → enforce), with rules grouped by family in `kyber/rules/`
- **Cost-based physical choices**: join build-side swap and hash-vs-broadcast
  selection from sketch/learned cardinality
- **Learned cardinality** that sharpens across runs via the MetadataHub
- **Intra-query adaptive re-optimization** - re-plans at pipeline breakers on
  *measured* sizes, single-node and distributed (the moat over DuckDB/Spark AQE)

[Learn more about Kyber](kyber.md)

### Carbonite

Carbonite handles memory management and data movement:

- **Memory coordination** across the cluster
- **File-based caching** for intermediate results
- **Spill-to-disk** when memory is constrained
- **Shuffle optimization** for redistributing data
- **Backpressure control** for streaming execution

[Learn more about Carbonite](carbonite.md)

### Execution engine

The execution engine runs the optimized plan in Rust over Arrow batches. It lowers
the plan into pipelines and breakers, schedules the work as 16K-row morsels, and
runs each pipeline through one of three paths that share operator semantics: the
sequential interpreter (the oracle), a rayon-parallel path, and a Cranelift JIT that
falls back to the interpreter on anything it does not support. The same mergeable
primitives run on one core, many cores, or many machines.

[Learn more about the execution engine](execution.md)

## Data flow

A typical query flows through the system:

```
1. User writes: ds.filter(col("x") > 10).select("a", "b")
                                    │
2. Dataset builds logical plan:     │
   LogicalProject(a, b)             │
       └── LogicalFilter(x > 10)    │
               └── LogicalScan(ds)  │
                                    ▼
3. Kyber optimizes:
   - Push filter before project
   - Prune unused columns
   - Select scan strategy
                                    │
4. Physical plan:                   │
   PhysicalProject(a, b)            │
       └── PhysicalFilter(x > 10)   │
               └── PhysicalScan(ds) │
                                    ▼
5. Execution engine:
   - Compile to a task graph
   - Schedule morsels (Ray on a cluster, threads on one node)
   - Stream results
                                    │
6. Carbonite:                       │
   - Manage memory and spill        │
   - Move batches over Arrow Flight │
                                    ▼
7. The Rust data plane executes the operators over Arrow batches
                                    │
8. Results collected                ▼
```

## Key concepts

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

### Optimization impact

| Optimization | Typical speedup |
|--------------|-----------------|
| Predicate pushdown | 2-10x |
| Projection pruning | 1.5-3x |
| Join reordering | 2-100x |
| Operator fusion | 1.2-2x |

### Scalability

Batcher's distributed path composes the *same* mergeable primitives
(`partial → combine → finalize`) the single-node path uses, so per-node memory
stays bounded and the shuffle is credit-controlled (data bypasses the Ray object
store via Arrow Flight). This is the design basis for near-linear scaling, but
**published multi-node throughput numbers are not yet measured** - distributed
execution is validated for *correctness* (single-node == multi-worker equivalence)
in CI; large-cluster benchmarks are pending real multi-host runs. We don't quote a
GB/s-per-node figure until the benchmark harness produces one.

## In this section

```{toctree}
:maxdepth: 1

kyber
carbonite
execution
testing-strategy
```

The formal treatment (cost models, sketch error bounds, control-theory stability
proofs) lives in `internals/mathematical_foundations.md`, which is rendered to PDF
by `internals/generate_pdf.py` rather than as a site page.

## See also

- [Architecture overview](../architecture/overview.md): high-level design
