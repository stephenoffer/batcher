# Execution model

This page describes how Batcher turns an optimized plan into results: the lazy API, the
pipeline and breaker model, and the three execution paths.

Batcher runs across two planes. Python is the control plane. It builds a plan,
optimizes it, and decides resource bounds, but never touches a row in the hot path.
Rust is the data plane, where every per-row and per-batch computation runs over
Apache Arrow. The two meet at a single boundary, a JSON plan IR plus zero-copy
Arrow `RecordBatch`es carried over the Arrow C Data Interface, and nothing else
crosses it. The Python entry point is Core handing the plan to the native engine:

```python
out, metrics = _native.execute_plan_metered(plan.to_json(), sources, cfg.engine_config_json())
```

For the contributor's view, which crate runs which scale, the thresholds with their
config names, and the metadata layer that answers some terminals without a scan, see
{doc}`Execution engine <../internals/execution>`.

## Lazy evaluation

The API is lazy and immutable. Each operation returns a new `Dataset` wrapping a
`LogicalPlan`, and nothing computes until a terminal call.

```python
import batcher as bt

ds = bt.read("data.parquet")
filtered = ds.filter(bt.col("x") > 0)
result = filtered.select("x", "y")  # still no execution

rows = result.collect()  # the plan runs here
```

Deferring work until the terminal op is what makes whole-query optimization possible.
By `collect`, the optimizer sees the entire computation and can push predicates and
projections down, fuse operators, and choose join orders before a single batch is read.
It's also what makes adaptive re-optimization possible mid-query, because there is one
plan to revise rather than a sequence of already-executed steps.

The terminal operations are `collect()`, which returns a PyArrow `Table`;
`to_pydict()`; `count()`; `iter_batches()`, which streams a result without
materializing it whole; and the `write` namespace, either `ds.write("out/")` or a typed
form such as `ds.write.parquet(...)`. To see the optimized plan without running it,
call `explain()`:

```python
print(ds.filter(bt.col("x") > 10).select("a", "b").explain())
```

## Pipelines and breakers

Execution lowers a plan into pipelines and breakers. A pipeline is a maximal chain
of operators that streams a batch straight through without materializing, such as
scan, filter, project, and probe. A breaker is an operator that must collect its input
before it can produce output, such as a hash-join build, an aggregate, a sort, a
distinct, or a window.

![A streaming Scan-Filter-Project pipeline feeding two pipeline breakers: the HashJoin build, then the Aggregate.](/_static/diagrams/pipeline_breakers.svg)

Breakers are where the model does its real work. Data materializes there, spills
there under memory pressure, shuffles there when a query is distributed, and gets
re-optimized there once real numbers are known. The unit of work flowing through a
pipeline is the morsel, a `RecordBatch` of 16,384 rows by default, which keeps
scheduling granular and the working set in cache.

## Execution paths

There is one set of operator semantics, exercised by three paths.

The Tier-0 sequential interpreter is the reference. It is deterministic and kept
obviously correct, and the other two paths are tested against it.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

Tier-0 parallel reuses the same operator code and changes only the scheduling. It
morselizes, runs on a rayon thread pool, and hash-shuffles into the breakers,
computing exactly what the sequential path does. Tier-1 is the Cranelift JIT, which
compiles the supported subset of column expressions to machine code once per operator
and reuses that across every morsel. On anything it doesn't support, the JIT falls back
to the interpreter rather than diverge, so it stays bit-for-bit identical to the
interpreter on its subset.

A compiled pipeline can drop back to the interpreter at any breaker, which is what
lets compilation and adaptivity coexist.

## One algebra, single node to cluster

Stateful operators are written once as mergeable primitives: `partial(batch)`
builds a partial state, `combine(states)` merges two of them, and `finalize(state)`
emits rows. Because `combine` is associative and commutative, partials merge in any
order. That single implementation serves one core (the sequential interpreter),
many cores (the parallel path builds partials and combines them), and many machines,
where the distributed path composes the same `partial`, `combine`, and `finalize`.
There is no separate distributed operator with its own semantics, so a
result is identical whether it runs on a laptop or a cluster. CI asserts exactly that.

## Adaptive re-optimization

At a stage boundary the engine has *measured* the data it just processed rather than
estimated it: real row counts, real operator times, and real peak memory. Core records
those numbers, and when an estimate was off by more than `optimizer.reoptimize_error`
(2.0 by default), Kyber re-plans the rest of the query on the measured values before
continuing.

This is stage-boundary re-optimization, the same granularity Spark AQE works at, and
Batcher runs it single-node as well as distributed. DuckDB, by contrast, optimizes once
before it runs. The loop is gated: `adaptive="auto"` is the default, and it turns on
only for plans large enough and uncertain enough that measuring could flip a
downstream decision. A separate cross-query loop feeds sketch-backed statistics from
each run into the next, so estimates sharpen the more a query runs.

## Memory and spilling

Carbonite owns the memory envelope. It throttles new allocations as the budget fills
and begins spilling to disk before the budget is exhausted. Aggregation, join, and sort
all spill, so a query that doesn't fit in memory slows down rather than failing.
Spilling is a property of the runtime primitive rather than a separate operator, so the
plan doesn't change when a query goes out of core.

## Distribution

Ray is an optional dependency used for task and actor scheduling and control-plane
metadata only, and single-node execution never loads it. On a cluster, each worker
hosts the same in-process Rust engine, and bulk Arrow batches move between workers
over Arrow Flight (`bc-transport`) with credit-based flow control. One credit is one
in-flight batch slot, and a producer blocks when its credits reach zero. Those batches
bypass the Ray object store entirely, which is where the serialization overhead and OOM
risk of an object-store shuffle would otherwise come from. The radix-partition-and-spill
machinery that does single-node out-of-core also becomes the distributed shuffle, so
disk and network are two sinks for one mechanism.

## See also

- {doc}`Execution engine <../internals/execution>`: the tiers, the crate map, and the exact thresholds.
- {doc}`Architecture overview <overview>`: the two planes and the control-plane subsystems.
- {doc}`Fault tolerance <fault-tolerance>`: what happens when a worker or a task fails.
- {doc}`Configuration options <../configuration/options>`: every execution knob.
