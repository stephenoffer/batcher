# Deep dives

How the engine actually works, one mechanism at a time.

The [architecture guide](../architecture/index.md) gives the shape of the system. These
pages go a level down: what a mechanism is for, how it works, what it costs, and where the
code lives. Each one names real files, so you can stop reading and go look.

They are written for two people. One is trying to understand why a query behaved the way it
did. The other is about to change the engine and needs to know what they would break.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` The query, end to end
:link: query-lifecycle
:link-type: doc
From `collect()` to Arrow and back: the plan IR, the interpreter, the JIT.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Parallelism and the operator core
:link: mergeable-algebra
:link-type: doc
Morsels, `partial → combine → finalize`, and the four stateful operators.
:::

:::{grid-item-card} {octicon}`database;1.1em` Memory
:link: arrow-memory
:link-type: doc
The one columnar contract everything speaks, and what happens when the data does not fit.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Distribution
:link: shuffle-flight
:link-type: doc
The Flight shuffle, credit-based flow control, Ray scheduling, and the GPU path.
:::

:::{grid-item-card} {octicon}`graph;1.1em` The adaptive layer
:link: adaptive-reoptimization
:link-type: doc
Re-planning mid-query on measured cardinalities, plus the cross-query learned-stats loop.
:::

:::{grid-item-card} {octicon}`book;1.1em` The architecture above it
:link: ../architecture/index
:link-type: doc
The shape of the system, if these pages are already a level too deep.
:::
::::

## The query, end to end

Follow one query from Python down to Arrow and back.

- [Query lifecycle](query-lifecycle.md): what happens between `collect()` and your rows.
- [The plan IR](plan-ir.md): the JSON wire contract between the control plane and the engine.
- [Expression evaluation](expression-evaluation.md): one `Expr`, vectorized over Arrow.
- [JIT compilation](jit-compilation.md): the Cranelift fast path, and why it must fall back rather than diverge.

## Parallelism and the operator core

The first two pages give you the shape shared by every operator. The last four are the four
stateful operators themselves, each one an instance of that shape.

- [Morsel parallelism](morsel-parallelism.md): why work is cut into 16,384-row chunks.
- [Mergeable algebra](mergeable-algebra.md): `partial → combine → finalize`, and why one core and one cluster run the same code.
- [Aggregation internals](aggregation-internals.md): a `group_by().agg()` from the morsel to the output rows, and the decisions made at runtime rather than at plan time.
- [Join algorithms](join-algorithms.md): the one row-index primitive every join type and strategy is built on.
- [Sort internals](sort-internals.md): the only operator whose order is the answer, and why an order-independent test cannot see its bugs.
- [Window internals](window-internals.md): a pipeline breaker that must return every input row, in the original order.

## Memory

Start with the contract, then the accounting, then what happens when the accounting says no.

- [Arrow memory model](arrow-memory.md): the only columnar contract, and what zero-copy really buys.
- [Tensor columns](tensor-columns.md): how an image becomes a column without a Python round trip.
- [The buffer pool](buffer-pool.md): the process-wide byte account every allocation of consequence reserves against.
- [Spilling](spilling.md): staying alive when the data does not fit.

## Distribution

Scaling out is a scheduling concern, not a second engine. These pages cover what moves, what
schedules it, and what keeps a fast producer from burying a slow consumer.

- [Shuffle over Arrow Flight](shuffle-flight.md): why bulk data bypasses the Ray object store.
- [Credit-based flow control](credit-flow-control.md): one credit is one batch slot, and the producer blocks at zero.
- [Distributed scheduling](distributed-scheduling.md): where work runs, how many pieces it runs in, and what does and doesn't travel through Ray.
- [GPU execution](gpu-execution.md): the two paths that run work on a device, and the scheduling that keeps it busy.

## The adaptive layer

Batcher re-optimizes at stage boundaries on measured cardinalities, the same mechanism and the
same granularity as Spark AQE, but available single-node too. It is also off for queries under
20M input rows, so most queries never reach it. What neither DuckDB nor Spark has is the second
half: a sketch-backed *cross-query* learned-stats and bandit loop, so a plan improves the more
a query runs. Read these pages for how both halves work and where each one stops.

- [Adaptive re-optimization](adaptive-reoptimization.md): re-planning mid-query on measured cardinalities.
- [Cardinality estimation](cardinality-estimation.md): how many rows a subtree will produce, how wrong that guess is, and how the engine tracks which.
- [The cost model](cost-model.md): turning row counts into the one comparable number that ranks two plans.
- [Learned metadata](learned-metadata.md): Core measures, Kyber consumes, and the plan improves the more a query runs.

## See also

:::{seealso}
- [Architecture](../architecture/index.md): the shape of the system these pages sit inside
- [Kyber](../internals/kyber.md), [Carbonite](../internals/carbonite.md), [the execution engine](../internals/execution.md): the three subsystems, at design level
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the contracts, the control theory, the sketch error bounds, the regret proofs
- [Performance](../user-guide/performance.md) and [reading a plan](../user-guide/explain-plans.md): where a reader applies all of this
- [Benchmarks](../benchmarks/index.md): the numbers these pages keep quoting
- [Extending the engine](../internals/extending.md): what to read before you change one of these mechanisms
:::

```{toctree}
:hidden:
:caption: The query, end to end

query-lifecycle
plan-ir
expression-evaluation
jit-compilation
```

```{toctree}
:hidden:
:caption: Parallelism and operators

morsel-parallelism
mergeable-algebra
aggregation-internals
join-algorithms
sort-internals
window-internals
```

```{toctree}
:hidden:
:caption: Memory

arrow-memory
tensor-columns
buffer-pool
spilling
```

```{toctree}
:hidden:
:caption: Distribution

shuffle-flight
credit-flow-control
distributed-scheduling
gpu-execution
```

```{toctree}
:hidden:
:caption: The adaptive layer

adaptive-reoptimization
cardinality-estimation
cost-model
learned-metadata
```
